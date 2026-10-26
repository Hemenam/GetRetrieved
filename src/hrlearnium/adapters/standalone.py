"""Configuration for the retained standalone/development JWT adapter only."""

from pydantic import Field, SecretStr, model_validator

from hrlearnium.core_config import CoreSettings


class StandaloneSettings(CoreSettings):
    jwt_secret: SecretStr = SecretStr("")
    jwt_issuer: str = "django-lms"
    jwt_audience: str = "hrlearnium-api"
    max_token_lifetime_seconds: int = Field(default=300, ge=30, le=3600)
    enable_chat_ui: bool = True

    @model_validator(mode="after")
    def validate_standalone_auth(self):
        secret = self.jwt_secret.get_secret_value()
        if len(secret) < 32 or secret.startswith("REPLACE_"):
            raise ValueError(
                "Set a random HR_JWT_SECRET of at least 32 characters; run hrlearnium init"
            )
        return self
