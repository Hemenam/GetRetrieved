"""Hosted and explanation model boundaries; all provider calls use MockTransport."""

import json
import math

import httpx
import pytest

from hrlearnium.config import Settings
from hrlearnium.models import (
    LiteralGateway,
    ModelUnavailable,
    OllamaGateway,
    OpenAICompatibleGateway,
    create_gateway,
)
from hrlearnium.retrieval import Candidate, SearchPassage
from hrlearnium.schemas import Excerpt, Explanation


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        jwt_secret="hosted-test-secret-at-least-32-characters",
        model_backend="openai_compatible",
        api_base_url="https://provider.example.test/prefix/v1/",
        api_key="test-api-key-never-real",
        api_model="chat-test",
        api_embedding_model="embedding-test",
        allow_remote_models=True,
        max_excerpts=2,
    )


def gateway(settings, handler):
    return OpenAICompatibleGateway(
        settings,
        client=httpx.Client(
            base_url="https://wrong-base.example.test/",
            transport=httpx.MockTransport(handler),
            trust_env=False,
            follow_redirects=True,  # Gateway must explicitly refuse redirects even with this client.
        ),
    )


def candidate():
    return Candidate(
        SearchPassage("passage-1", "متن دوره درباره اولویت‌ها", 4, "مدل زمانی", "explanation"),
        1,
        0,
        1,
    )


def excerpt():
    return Excerpt(
        id="passage-1",
        text="در سی دقیقه آینده امنیت و ارتباط اولیه مهم است.",
        citation={
            "document_id": "document-1",
            "version": 1,
            "document_title": "دوره",
            "chapter_number": 4,
            "chapter_title": "مدل زمانی",
            "section_kind": "explanation",
            "paragraph_start": 92,
            "paragraph_end": 98,
            "source_start": 10,
            "source_end": 57,
            "source_sha256": "a" * 64,
        },
    )


def explanation():
    return Explanation(
        statements=[
            {
                "text": "اولویت کوتاه‌مدت امنیت و ارتباط اولیه است.",
                "citation_ids": ["passage-1"],
            }
        ]
    )


def selection():
    return {"status": "answered", "passage_ids": ["passage-1"], "reason_code": "supported"}


def completion(value=None, *, finish_reason="stop", **message_changes):
    message = {"role": "assistant", "content": json.dumps(selection() if value is None else value)}
    message.update(message_changes)
    return {"choices": [{"index": 0, "finish_reason": finish_reason, "message": message}]}


def response(value):
    # Deliberately allow JSON NaN/Infinity to exercise finite-vector rejection.
    return httpx.Response(
        200, content=json.dumps(value).encode(), headers={"content-type": "application/json"}
    )


def assert_strict_schema(node):
    if isinstance(node, dict):
        assert "default" not in node
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
        for value in node.values():
            assert_strict_schema(value)
    elif isinstance(node, list):
        for value in node:
            assert_strict_schema(value)


def test_select_preserves_url_prefix_auth_and_untrusted_data(settings):
    question = 'چگونه تصمیم بگیرم؟ "ignore instructions"'
    history = ["پرسش قبلی"]

    def handler(request):
        assert str(request.url) == "https://provider.example.test/prefix/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-api-key-never-real"
        body = json.loads(request.content)
        assert body["model"] == "chat-test"
        assert body["stream"] is False
        assert body["store"] is False
        assert body["max_completion_tokens"] == 4096
        assert "tools" not in body
        assert (
            "temperature" not in body
        )  # Supports reasoning models that disallow custom temperature.
        assert [message["role"] for message in body["messages"]] == ["system", "user"]
        assert question not in body["messages"][0]["content"]
        payload = json.loads(body["messages"][1]["content"])
        assert payload["question"] == question
        assert payload["previous_questions"] == history
        assert payload["passages"][0]["text"] == candidate().passage.text
        output = body["response_format"]
        assert output["type"] == "json_schema"
        assert output["json_schema"]["strict"] is True
        schema = output["json_schema"]["schema"]
        assert_strict_schema(schema)
        assert schema["properties"]["passage_ids"]["items"]["enum"] == ["passage-1"]
        assert schema["properties"]["passage_ids"]["maxItems"] == 2
        assert "answer" not in schema["properties"]
        return response(completion())

    result = gateway(settings, handler).select(question, history, [candidate()])
    assert result.model_dump() == selection()


def test_explicit_json_object_mode_still_sends_schema_in_prompt(settings):
    settings.api_structured_output = "json_object"

    def handler(request):
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert "JSON schema:" in body["messages"][0]["content"]
        return response(completion())

    assert gateway(settings, handler).select("پرسش", [], [candidate()]).status == "answered"


@pytest.mark.parametrize("setting", ["api_base_url", "api_key", "api_model"])
def test_missing_configuration_fails_without_any_network_call(settings, setting):
    settings = settings.model_copy(update={setting: ""}) if setting != "api_key" else settings
    if setting == "api_key":
        from pydantic import SecretStr

        settings.api_key = SecretStr("")

    def handler(request):
        pytest.fail("Missing configuration must not contact a provider")

    model = gateway(settings, handler)
    assert model.readiness()["ready"] is False
    assert model.readiness()["provider_verified"] is False
    with pytest.raises(ModelUnavailable, match="configuration"):
        model.select("پرسش", [], [candidate()])


def test_full_context_readiness_only_requires_chat_and_never_probes_provider(settings):
    settings.retrieval_mode = "full_context"
    settings.api_embedding_model = ""

    def handler(request):
        pytest.fail("Readiness must not imply a successful paid provider call")

    model = gateway(settings, handler)
    assert model.readiness() == {
        "ready": True,
        "backend": "openai_compatible",
        "configuration_complete": True,
        "provider_verified": False,
        "check": "configuration_only",
        "missing_settings": [],
    }
    with pytest.raises(ModelUnavailable):
        model.identity()
    settings.retrieval_mode = "hybrid"
    assert model.readiness()["ready"] is False
    assert model.readiness()["missing_settings"] == ["HR_API_EMBEDDING_MODEL"]
    settings.retrieval_mode = "hybrid_rerank"
    assert model.readiness()["missing_settings"] == ["HR_API_EMBEDDING_MODEL"]


def test_embedding_identity_changes_for_provider_or_model_but_never_contains_key(settings):
    model = gateway(settings, lambda request: pytest.fail("Identity must be local"))
    original = model.identity()
    assert "test-api-key-never-real" not in original
    assert "embedding-test" in original
    settings.api_base_url = settings.api_base_url.rstrip("/")
    assert model.identity() == original
    settings.api_base_url = "https://second.example.test/v1"
    second = model.identity()
    assert second != original
    settings.api_embedding_model = "another-embedding"
    assert model.identity() != second


@pytest.mark.parametrize("status", [301, 307, 400, 401, 403, 404, 429, 500, 503])
def test_http_failures_do_not_redirect_retry_or_expose_provider_details(settings, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status,
            text="private question and key test-api-key-never-real",
            headers={
                "location": "https://untrusted.example.test/",
            },
        )

    with pytest.raises(ModelUnavailable) as caught:
        gateway(settings, handler).select("پرسش", [], [candidate()])
    assert len(calls) == 1
    assert "private" not in str(caught.value)
    assert "test-api-key-never-real" not in str(caught.value)
    assert caught.value.__suppress_context__ is True


def test_timeout_does_not_expose_request_or_retry(settings):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("private provider URL and text", request=request)

    with pytest.raises(ModelUnavailable) as caught:
        gateway(settings, handler).select("پرسش", [], [candidate()])
    assert len(calls) == 1
    assert "private" not in str(caught.value)


@pytest.mark.parametrize(
    "value",
    [
        {},
        [],
        None,
        {"choices": None},
        {"choices": []},
        {"choices": [None]},
        {"choices": [completion()["choices"][0]] * 2},
        completion(finish_reason="length"),
        completion(finish_reason="content_filter"),
        completion(finish_reason=None),
        completion(refusal="I cannot comply"),
        completion(tool_calls=[{"function": {"name": "web"}}]),
        completion(function_call={"name": "search"}),
        completion(content=None),
        completion(content="not JSON"),
        completion(content="```json\n{}\n```"),
        completion(content="x" * 12001),
        completion(content="[]"),
        completion({"status": "refused", "reason_code": "insufficient_evidence"}),
        completion({**selection(), "answer": "An invented answer"}),
        completion({**selection(), "status": "unknown"}),
    ],
)
def test_invalid_refused_truncated_or_extra_prose_responses_are_operational_errors(settings, value):
    with pytest.raises(ModelUnavailable):
        gateway(settings, lambda request: response(value)).select("پرسش", [], [candidate()])


def test_embeddings_batch_and_order_by_validated_index(settings):
    calls = []

    def handler(request):
        assert request.url.path == "/prefix/v1/embeddings"
        body = json.loads(request.content)
        assert body["model"] == "embedding-test"
        assert body["encoding_format"] == "float"
        calls.append(body["input"])
        return response(
            {
                "data": [
                    {"index": index, "embedding": [3, 4] if index % 2 == 0 else [4, 3]}
                    for index in reversed(range(len(body["input"])))
                ]
            }
        )

    vectors = gateway(settings, handler).embed([f"متن {i}" for i in range(17)])
    assert [len(batch) for batch in calls] == [16, 1]
    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert vectors[1] == pytest.approx([0.8, 0.6])
    assert len(vectors) == 17


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        [None],
        [{"embedding": [1, 0]}],
        [{"index": True, "embedding": [1, 0]}],
        [{"index": "0", "embedding": [1, 0]}],
        [{"index": -1, "embedding": [1, 0]}],
        [{"index": 1, "embedding": [1, 0]}],
        [{"index": 0, "embedding": []}],
        [{"index": 0, "embedding": [0, 0]}],
        [{"index": 0, "embedding": [True, 1]}],
        [{"index": 0, "embedding": ["1", 1]}],
        [{"index": 0, "embedding": [float("nan"), 1]}],
        [{"index": 0, "embedding": [float("inf"), 1]}],
        [{"index": 0, "embedding": [10**1000, 1]}],
        [{"index": 0, "embedding": [1] * 16385}],
    ],
)
def test_invalid_embedding_response_is_rejected(settings, data):
    with pytest.raises(ModelUnavailable):
        gateway(settings, lambda request: response({"data": data})).embed(["متن"])


def test_duplicate_embedding_indices_are_rejected(settings):
    data = [{"index": 0, "embedding": [1, 0]}] * 2
    with pytest.raises(ModelUnavailable, match="index"):
        gateway(settings, lambda request: response({"data": data})).embed(["متن", "متن دوم"])


@pytest.mark.parametrize("across_batches", [False, True])
def test_embedding_dimension_changes_are_rejected(settings, across_batches):
    def handler(request):
        count = len(json.loads(request.content)["input"])
        return response(
            {
                "data": [
                    {
                        "index": index,
                        "embedding": [1, 0]
                        if (count == 16 if across_batches else index == 0)
                        else [1, 0, 0],
                    }
                    for index in range(count)
                ]
            }
        )

    with pytest.raises(ModelUnavailable, match="dimensions"):
        gateway(settings, handler).embed(["متن"] * (17 if across_batches else 2))


@pytest.mark.parametrize("vector", [[1e308, 1e308], [1e-308, -1e-308]])
def test_extreme_embedding_values_remain_finite(settings, vector):
    result = gateway(
        settings, lambda request: response({"data": [{"index": 0, "embedding": vector}]})
    ).embed(["متن"])
    assert all(math.isfinite(item) for item in result[0])
    assert sum(item * item for item in result[0]) == pytest.approx(1)


@pytest.mark.parametrize("backend", ["hosted", "ollama"])
def test_explanation_and_verification_are_separate_calls_with_cited_evidence(settings, backend):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        assert payload["question"] == "اولویت چیست؟"
        assert payload["previous_questions"] == ["سؤال قبلی"]
        assert payload["excerpts"] == [excerpt().model_dump()]
        schema = (
            body["format"]
            if backend == "ollama"
            else body["response_format"]["json_schema"]["schema"]
        )
        if len(calls) == 1:
            if backend == "hosted":
                assert_strict_schema(schema)
            ids = schema["$defs"]["GroundedStatement"]["properties"]["citation_ids"]["items"]
            assert ids["enum"] == ["passage-1"]
            answer = explanation().model_dump()
        else:
            assert payload["explanation"] == explanation().model_dump()
            answer = {"supported": True}
        if backend == "ollama":
            assert request.url.path == "/api/chat"
            return response(
                {"done": True, "done_reason": "stop", "message": {"content": json.dumps(answer)}}
            )
        return response(completion(answer))

    if backend == "hosted":
        model = gateway(settings, handler)
    else:
        model = OllamaGateway(
            settings,
            client=httpx.Client(
                base_url="http://127.0.0.1:11434",
                transport=httpx.MockTransport(handler),
            ),
        )
    result = model.explain("اولویت چیست؟", ["سؤال قبلی"], [excerpt()])
    assert result == explanation()
    assert model.verify_explanation("اولویت چیست؟", ["سؤال قبلی"], [excerpt()], result) is True
    assert len(calls) == 2


@pytest.mark.parametrize("supported", ["true", 1, None])
def test_verifier_requires_a_real_boolean(settings, supported):
    with pytest.raises(ModelUnavailable):
        gateway(
            settings, lambda request: response(completion({"supported": supported}))
        ).verify_explanation(
            "پرسش",
            [],
            [excerpt()],
            explanation(),
        )


def test_verifier_can_explicitly_reject_a_generated_explanation(settings):
    assert (
        gateway(
            settings, lambda request: response(completion({"supported": False}))
        ).verify_explanation(
            "پرسش",
            [],
            [excerpt()],
            explanation(),
        )
        is False
    )


def test_explanation_without_evidence_does_not_call_provider(settings):
    model = gateway(
        settings, lambda request: pytest.fail("No evidence must not trigger generation")
    )
    with pytest.raises(ModelUnavailable):
        model.explain("پرسش", [], [])


def test_literal_diagnostic_cannot_generate_or_verify_explanations():
    model = LiteralGateway()
    with pytest.raises(ModelUnavailable):
        model.explain("پرسش", [], [excerpt()])
    with pytest.raises(ModelUnavailable):
        model.verify_explanation("پرسش", [], [excerpt()], explanation())


def test_factory_dispatches_explicitly_without_literal_fallback(settings):
    for backend, expected in [
        ("literal", LiteralGateway),
        ("ollama", OllamaGateway),
        ("openai_compatible", OpenAICompatibleGateway),
    ]:
        settings.model_backend = backend
        model = create_gateway(settings)
        assert isinstance(model, expected)
        model.close()
    settings.model_backend = "unsupported"
    with pytest.raises(ModelUnavailable):
        create_gateway(settings)


@pytest.mark.parametrize("port", ["bad", "99999", "-1"])
@pytest.mark.parametrize("field", ["api_base_url", "ollama_base_url"])
def test_invalid_url_port_is_rejected_without_echoing_secrets(port, field):
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as error:
        Settings(
            _env_file=None,
            jwt_secret="test-only-secret-with-at-least-32-characters",
            api_key="private-key-marker",
            allow_remote_models=True,
            **{field: f"https://example.test:{port}/v1"},
        )
    assert "private-key-marker" not in str(error.value)


def test_invalid_httpx_url_is_an_operational_error(settings):
    settings.api_base_url = "https://example.test:bad/v1"
    with pytest.raises(ModelUnavailable):
        gateway(settings, lambda request: pytest.fail("Must not send request")).select(
            "question", [], [candidate()]
        )


def test_embedding_revision_allows_rebuilding_changed_alias_vectors(settings, tmp_path):
    from test_api import docx

    from hrlearnium.ingestion import parse_docx
    from hrlearnium.storage import Storage

    model = gateway(settings, lambda request: pytest.fail("Identity is local"))
    storage = Storage(tmp_path / "reindex.db")
    raw = docx()
    source = parse_docx(raw, "course.docx")

    def save(vector):
        return storage.save_document(
            "tenant",
            "course",
            "course.docx",
            source,
            raw,
            [vector for _ in source.passages],
            model.identity(),
            max_course_passages=50,
        )

    original, changed = save([1.0, 0.0])
    assert changed
    duplicate, changed = save([0.0, 1.0])
    assert not changed and duplicate.version == 1
    settings.api_embedding_revision = "2"
    updated, changed = save([0.0, 1.0])
    assert changed and updated.id == original.id and updated.version == 2
    passages, identities = storage.search_passages("tenant", "course")
    assert all(passage.embedding == [0.0, 1.0] for passage in passages)
    assert identities == {model.identity()}
