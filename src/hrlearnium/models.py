import hashlib
import json
import math
import re
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from hrlearnium.config import Settings
from hrlearnium.evaluation import provider_usage
from hrlearnium.policy import (
    EXPLANATION_INSTRUCTIONS,
    SELECTOR_INSTRUCTIONS,
    VERIFICATION_INSTRUCTIONS,
)
from hrlearnium.retrieval import Candidate
from hrlearnium.schemas import Excerpt, Explanation, ExplanationVerification, Selection
from hrlearnium.text import normalize_persian


class ModelUnavailable(Exception):
    """Infrastructure/protocol error, distinct from a valid evidence-based refusal."""


class ModelGateway(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def select(
        self, question: str, previous_questions: list[str], candidates: list[Candidate]
    ) -> Selection: ...
    def explain(
        self, question: str, previous_questions: list[str], excerpts: list[Excerpt]
    ) -> Explanation: ...
    def verify_explanation(
        self,
        question: str,
        previous_questions: list[str],
        excerpts: list[Excerpt],
        explanation: Explanation,
    ) -> bool: ...
    def identity(self) -> str | None: ...
    def readiness(self) -> dict: ...
    def close(self) -> None: ...


StructuredResult = TypeVar("StructuredResult", bound=BaseModel)


def _strict_schema(value):
    """OpenAI strict schemas require every object property, including fields with defaults."""
    if isinstance(value, list):
        return [_strict_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _strict_schema(item) for key, item in value.items() if key != "default"}
    if result.get("type") == "object":
        result["required"] = list(result.get("properties", {}))
        result["additionalProperties"] = False
    return result


def _normalized_vectors(values, count: int, dimensions: int | None):
    if not isinstance(values, list) or len(values) != count:
        raise ModelUnavailable("Invalid embedding count")
    vectors = []
    for vector in values:
        try:
            if (
                not isinstance(vector, list)
                or not 1 <= len(vector) <= 16384
                or any(
                    type(value) not in (float, int) or not math.isfinite(value) for value in vector
                )
            ):
                raise ValueError("Invalid embedding vector")
            scale = max(abs(float(value)) for value in vector)
            if not scale:
                raise ValueError("Zero embedding vector")
            scaled = [float(value) / scale for value in vector]
            norm = math.sqrt(sum(value * value for value in scaled))
        except (ValueError, OverflowError, TypeError):
            raise ModelUnavailable("Invalid embedding vector") from None
        if dimensions is not None and dimensions != len(vector):
            raise ModelUnavailable("Embedding dimensions changed")
        dimensions = len(vector)
        vectors.append([value / norm for value in scaled])
    return vectors, dimensions


def _content_limit(model: type[BaseModel]) -> int:
    return 20000 if model is Explanation else 1000 if model is ExplanationVerification else 12000


def _validate_structured(content: str, model: type[StructuredResult]) -> StructuredResult:
    # Enforce required fields locally too: json_object mode does not enforce a provider schema.
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != set(model.model_fields):
        raise ValueError("Structured response has missing or extra fields")
    return model.model_validate(value)


class _EvidenceGateway:
    settings: Settings

    def _structured(
        self,
        model: type[StructuredResult],
        instructions: str,
        payload: dict,
        schema: dict,
    ) -> StructuredResult:
        raise NotImplementedError

    def select(
        self, question: str, previous_questions: list[str], candidates: list[Candidate]
    ) -> Selection:
        if not candidates:
            raise ModelUnavailable("Evidence selection requires candidate passages")
        schema = Selection.model_json_schema()
        schema["properties"]["passage_ids"]["items"] = {
            "type": "string",
            "enum": [item.passage.id for item in candidates],
        }
        schema["properties"]["passage_ids"]["maxItems"] = self.settings.max_excerpts
        payload = {
            "question": question,
            "previous_questions": previous_questions,
            "passages": [
                {
                    "id": item.passage.id,
                    "text": item.passage.text,
                    "chapter_number": item.passage.chapter_number,
                    "chapter_title": item.passage.chapter_title,
                    "section_kind": item.passage.section_kind,
                }
                for item in candidates
            ],
        }
        return self._structured(Selection, SELECTOR_INSTRUCTIONS, payload, schema)

    @staticmethod
    def _explanation_payload(question: str, previous_questions: list[str], excerpts: list[Excerpt]):
        if not excerpts:
            raise ModelUnavailable("Explanation requires source excerpts")
        return {
            "question": question,
            "previous_questions": previous_questions,
            "excerpts": [excerpt.model_dump() for excerpt in excerpts],
        }

    def explain(
        self, question: str, previous_questions: list[str], excerpts: list[Excerpt]
    ) -> Explanation:
        payload = self._explanation_payload(question, previous_questions, excerpts)
        schema = Explanation.model_json_schema()
        schema["$defs"]["GroundedStatement"]["properties"]["citation_ids"]["items"] = {
            "type": "string",
            "enum": [excerpt.id for excerpt in excerpts],
        }
        return self._structured(Explanation, EXPLANATION_INSTRUCTIONS, payload, schema)

    def verify_explanation(
        self,
        question: str,
        previous_questions: list[str],
        excerpts: list[Excerpt],
        explanation: Explanation,
    ) -> bool:
        payload = self._explanation_payload(question, previous_questions, excerpts)
        payload["explanation"] = explanation.model_dump()
        result = self._structured(
            ExplanationVerification,
            VERIFICATION_INSTRUCTIONS,
            payload,
            ExplanationVerification.model_json_schema(),
        )
        return result.supported


class OllamaGateway(_EvidenceGateway):
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(
            base_url=settings.ollama_base_url.rstrip("/"),
            timeout=settings.model_timeout_seconds,
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self.client.request(method, path, **kwargs)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Expected JSON object")
            request_payload = kwargs.get("json", {})
            if method == "POST":
                usage = {
                    "prompt_tokens": result.get("prompt_eval_count"),
                    "completion_tokens": result.get("eval_count"),
                }
                provider_usage({"usage": usage}, request_payload.get("model", ""), path)
            return result
        except (httpx.HTTPError, ValueError, KeyError):
            # Do not propagate provider response bodies, source text, URLs or credentials to clients/logs.
            raise ModelUnavailable(
                "Model service unavailable or returned an invalid response"
            ) from None

    def _installed_models(self) -> dict[str, str]:
        result = self._request("GET", "/api/tags", timeout=5)
        try:
            return {item["name"]: item["digest"] for item in result["models"]}
        except (KeyError, TypeError) as error:
            raise ModelUnavailable("Invalid model inventory") from error

    def identity(self) -> str:
        installed = self._installed_models()
        name = self.settings.embedding_model
        digest = installed.get(name) or installed.get(name + ":latest")
        if not isinstance(digest, str) or not digest:
            raise ModelUnavailable("Configured embedding model is not installed")
        return f"ollama:{name}@{digest}"

    def readiness(self) -> dict:
        installed = self._installed_models()
        names = [self.settings.selector_model]
        if self.settings.retrieval_mode in {"hybrid", "hybrid_rerank"}:
            names.append(self.settings.embedding_model)
        return {
            "ready": all(name in installed or name + ":latest" in installed for name in names),
            "backend": "ollama",
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        dimensions = None
        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]
            result = self._request(
                "POST",
                "/api/embed",
                json={
                    "model": self.settings.embedding_model,
                    "input": batch,
                    "truncate": False,
                },
            )
            normalized, dimensions = _normalized_vectors(
                result.get("embeddings"), len(batch), dimensions
            )
            vectors.extend(normalized)
        return vectors

    def _structured(
        self,
        model: type[StructuredResult],
        instructions: str,
        payload: dict,
        schema: dict,
    ) -> StructuredResult:
        result = self._request(
            "POST",
            "/api/chat",
            json={
                "model": self.settings.selector_model,
                "stream": False,
                "think": False,
                "messages": [
                    {
                        "role": "system",
                        "content": instructions + "\nSchema:\n" + json.dumps(schema),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "format": schema,
                "options": {
                    "temperature": 0,
                    "num_ctx": 32768,
                    "num_predict": 4096 if model is Explanation else 700,
                },
            },
        )
        try:
            if result.get("done") is not True or result.get("done_reason") == "length":
                raise ValueError("Incomplete selection")
            content = result["message"]["content"]
            if not isinstance(content, str) or len(content) > _content_limit(model):
                raise ValueError("Invalid response length")
            return _validate_structured(content, model)
        except (ValidationError, KeyError, TypeError, ValueError):
            raise ModelUnavailable(
                "Model returned an invalid structured evidence response"
            ) from None

    def close(self) -> None:
        self.client.close()


class OpenAICompatibleGateway(_EvidenceGateway):
    """Explicit Chat Completions/embeddings contract; no provider guessing or fallback."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(
            timeout=settings.model_timeout_seconds,
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )

    def _missing_config(self, *, embeddings: bool = False) -> list[str]:
        missing = [
            name
            for name, value in (
                ("HR_API_BASE_URL", self.settings.api_base_url),
                ("HR_API_KEY", self.settings.api_key.get_secret_value()),
                ("HR_API_MODEL", self.settings.api_model),
            )
            if not value.strip()
        ]
        if embeddings and not self.settings.api_embedding_model.strip():
            missing.append("HR_API_EMBEDDING_MODEL")
        return missing

    def _require_config(self, *, embeddings: bool = False) -> None:
        if self._missing_config(embeddings=embeddings):
            raise ModelUnavailable("Hosted model configuration is incomplete")

    def readiness(self) -> dict:
        missing = self._missing_config(
            embeddings=self.settings.retrieval_mode in {"hybrid", "hybrid_rerank"}
        )
        return {
            "ready": not missing,
            "backend": "openai_compatible",
            "configuration_complete": not missing,
            "provider_verified": False,
            "check": "configuration_only",
            "missing_settings": missing,
        }

    def identity(self) -> str:
        self._require_config(embeddings=True)
        # No key is persisted. Changing the provider URL or embedding model requires reindexing.
        provider = hashlib.sha256(self.settings.api_base_url.rstrip("/").encode()).hexdigest()
        return (
            f"openai_compatible:{provider}:{self.settings.api_embedding_model}"
            f"@{self.settings.api_embedding_revision}"
        )

    def _request(self, path: str, payload: dict) -> dict:
        self._require_config()
        try:
            # An absolute URL preserves prefixes such as /v1 even when a test client has a base_url.
            response = self.client.post(
                self.settings.api_base_url.rstrip("/") + "/" + path,
                headers={"Authorization": "Bearer " + self.settings.api_key.get_secret_value()},
                json=payload,
                follow_redirects=False,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Expected JSON object")
            provider_usage(result, payload.get("model", ""), path)
            return result
        except (httpx.HTTPError, httpx.InvalidURL, ValueError, KeyError):
            raise ModelUnavailable(
                "Model service unavailable or returned an invalid response"
            ) from None

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._require_config(embeddings=True)
        vectors = []
        dimensions = None
        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]
            result = self._request(
                "embeddings",
                {
                    "model": self.settings.api_embedding_model,
                    "input": batch,
                    "encoding_format": "float",
                },
            )
            data = result.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise ModelUnavailable("Invalid embedding count")
            ordered = [None] * len(batch)
            seen = set()
            for item in data:
                if not isinstance(item, dict):
                    raise ModelUnavailable("Invalid embedding response")
                index = item.get("index")
                if type(index) is not int or not 0 <= index < len(batch) or index in seen:
                    raise ModelUnavailable("Invalid embedding index")
                seen.add(index)
                ordered[index] = item.get("embedding")
            normalized, dimensions = _normalized_vectors(ordered, len(batch), dimensions)
            vectors.extend(normalized)
        return vectors

    def _structured(
        self,
        model: type[StructuredResult],
        instructions: str,
        payload: dict,
        schema: dict,
    ) -> StructuredResult:
        schema = _strict_schema(schema)
        response_format = {"type": "json_object"}
        if self.settings.api_structured_output == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": model.__name__, "strict": True, "schema": schema},
            }
        result = self._request(
            "chat/completions",
            {
                "model": self.settings.api_model,
                "stream": False,
                "store": False,
                "max_completion_tokens": self.settings.api_max_completion_tokens,
                "response_format": response_format,
                "messages": [
                    {
                        "role": "system",
                        "content": instructions + "\nJSON schema:\n" + json.dumps(schema),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            },
        )
        try:
            choices = result["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("Invalid completion count")
            choice = choices[0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Incomplete completion")
            message = choice["message"]
            if (
                not isinstance(message, dict)
                or message.get("refusal")
                or message.get("tool_calls")
                or message.get("function_call")
            ):
                raise ValueError("No structured completion")
            content = message["content"]
            if not isinstance(content, str) or len(content) > _content_limit(model):
                raise ValueError("Invalid completion length")
            return _validate_structured(content, model)
        except (ValidationError, KeyError, TypeError, ValueError, AttributeError):
            raise ModelUnavailable(
                "Model returned an invalid structured evidence response"
            ) from None

    def close(self) -> None:
        self.client.close()


class LiteralGateway:
    """Offline diagnostic: only explicitly requested, exact quoted phrases can be looked up.

    This is NOT a replacement for the semantic answerability gate. It makes no claims
    about a passage answering a natural-language question and never guesses relevance.
    """

    def identity(self) -> None:
        return None

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ModelUnavailable("Literal lookup has no semantic model")

    def explain(
        self, question: str, previous_questions: list[str], excerpts: list[Excerpt]
    ) -> Explanation:
        raise ModelUnavailable("Literal lookup cannot generate explanations")

    def verify_explanation(
        self,
        question: str,
        previous_questions: list[str],
        excerpts: list[Excerpt],
        explanation: Explanation,
    ) -> bool:
        raise ModelUnavailable("Literal lookup cannot verify explanations")

    def readiness(self) -> dict:
        return {"ready": True, "backend": "literal", "natural_language_qa": False}

    def select(
        self, question: str, previous_questions: list[str], candidates: list[Candidate]
    ) -> Selection:
        # A full-match grammar prevents a quote from masking an unrelated request or extra advice.
        match = re.fullmatch(
            r'\s*(?:متن دقیق|عبارت دقیق|نقل قول|exact quote)\s*[«"]([^»"\n]{5,300})[»"]\s*[.؟?]?\s*',
            question,
            re.I,
        )
        if not match:
            return Selection(status="refused", passage_ids=[], reason_code="insufficient_evidence")
        phrase = normalize_persian(match.group(1))
        if len(phrase.strip()) < 5 or not any(character.isalnum() for character in phrase):
            return Selection(status="refused", passage_ids=[], reason_code="insufficient_evidence")
        matches = [
            item.passage.id for item in candidates if phrase in normalize_persian(item.passage.text)
        ]
        if len(matches) == 1:
            return Selection(status="answered", passage_ids=matches, reason_code="supported")
        return Selection(
            status="clarification" if matches else "refused",
            passage_ids=[],
            reason_code="ambiguous" if matches else "insufficient_evidence",
        )

    def close(self) -> None:
        pass


def create_gateway(settings: Settings) -> ModelGateway:
    if settings.model_backend == "ollama":
        return OllamaGateway(settings)
    if settings.model_backend == "openai_compatible":
        return OpenAICompatibleGateway(settings)
    if settings.model_backend == "literal":
        return LiteralGateway()
    raise ModelUnavailable("Unsupported model backend")
