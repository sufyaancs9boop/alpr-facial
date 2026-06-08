from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List


class Settings(BaseSettings):
    PORT: int = 3000
    API_PREFIX: str = "api"
    MODELS_DIR: str = "data/models"
    MAX_FILE_SIZE_MB: int = 20
    API_KEYS: str = ""
    RETENTION_DAYS: int = 90
    ENABLE_VEHICLE_DETECTION: bool = True
    ENABLE_FACE_DETECTION: bool = True
    PERSIST_FACE_EVENTS: bool = True

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def api_key_list(self) -> List[str]:
        return [k.strip() for k in self.API_KEYS.split(",") if k.strip()]

    @property
    def auth_enabled(self) -> bool:
        return len(self.api_key_list) > 0

    @property
    def max_file_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


settings = Settings()
