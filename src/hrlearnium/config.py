from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    jwt_secret: SecretStr = SecretStr("")
    jwt_issuer: str = "django-lms"
    jwt_audience: str = "hrlearnium-api"
    max_token_lifetime_seconds: int = Field(default=300, ge=30, le=3600)
    database_path: Path = Path("data/hrlearnium.sqlite3")
    model_backend: Literal["ollama", "literal", "openai_compatible"] = "ollama"
    api_base_url: str = ""
    api_key: SecretStr = SecretStr("")
    api_model: str = ""
    api_embedding_model: str = ""
    api_embedding_revision: str = Field(default="1", min_length=1, max_length=100)
    api_structured_output: Literal["json_schema", "json_object"] = "json_schema"
    api_max_completion_tokens: int = Field(default=4096, ge=1000, le=16000)
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    selector_model: str = "qwen3:8b"
    model_timeout_seconds: float = Field(default=120, ge=1, le=300)
    allow_remote_models: bool = False
    retrieval_mode: Literal["hybrid", "full_context"] = "hybrid"
    enable_docs: bool = True
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, le=10 * 1024 * 1024)
    max_course_passages: int = Field(default=2000, ge=10, le=10000)
    max_document_characters: int = Field(default=500_000, ge=1000, le=2_000_000)
    max_context_characters: int = Field(default=48_000, ge=2000, le=100_000)
    candidate_limit: int = Field(default=8, ge=2, le=20)
    max_excerpts: int = Field(default=4, ge=1, le=8)
    conversation_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    requests_per_minute: int = Field(default=30, ge=1, le=600)
    tenant_requests_per_minute: int = Field(default=300, ge=1, le=3000)

    @model_validator(mode="after")
    def validate_runtime(self) -> "Settings":
        secret = self.jwt_secret.get_secret_value()
        if len(secret) < 32 or secret.startswith("REPLACE_"):
            raise ValueError(
                "Set a random HR_JWT_SECRET of at least 32 characters; run hrlearnium init"
            )
        url = urlparse(self.ollama_base_url)
        _ = url.port  # Reject malformed/out-of-range ports before creating HTTP clients.
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("HR_OLLAMA_BASE_URL must be an HTTP(S) URL without credentials")
        # 'ollama' is the service name in the optional Docker Compose deployment.
        local_hosts = {"127.0.0.1", "localhost", "::1", "ollama", "host.docker.internal"}
        if not self.allow_remote_models and url.hostname not in local_hosts:
            raise ValueError("A remote model host requires HR_ALLOW_REMOTE_MODELS=true")
        if self.allow_remote_models and url.hostname not in local_hosts and url.scheme != "https":
            raise ValueError("Remote model connections require HTTPS")
        if self.api_base_url:
            api_url = urlparse(self.api_base_url)
            _ = api_url.port
            if (
                api_url.scheme not in {"http", "https"}
                or not api_url.hostname
                or api_url.username
                or api_url.password
                or api_url.query
                or api_url.fragment
            ):
                raise ValueError(
                    "HR_API_BASE_URL must be an HTTP(S) URL without credentials or query"
                )
            if api_url.hostname not in local_hosts:
                if not self.allow_remote_models:
                    raise ValueError("A remote API host requires HR_ALLOW_REMOTE_MODELS=true")
                if api_url.scheme != "https":
                    raise ValueError("Remote API connections require HTTPS")
        return self
