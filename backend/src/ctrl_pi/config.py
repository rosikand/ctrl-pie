from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Process configuration sourced from the environment.

    Secret values deliberately remain ``SecretStr`` objects so accidental
    serialization or logging redacts them.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: SecretStr | None = None
    hf_token: SecretStr | None = None
    hf_namespace: str | None = None
    modal_token_id: SecretStr | None = None
    modal_token_secret: SecretStr | None = None
    ctrl_pi_mock_mode: bool = True
    recording_staging_dir: Path = Path(".ctrl-pi/recordings")
    recording_fps: int = Field(default=20, ge=1, le=60)

    @property
    def mock_mode(self) -> bool:
        return self.ctrl_pi_mock_mode


@lru_cache
def get_config() -> AppConfig:
    return AppConfig()
