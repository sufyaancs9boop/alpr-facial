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
    DEFAULT_PLATE_REGION: str = "PAKISTAN"
    ALPR_DEDUP_IOU_THRESHOLD: float = 0.5
    ALPR_DEDUP_CENTER_DISTANCE_RATIO: float = 0.25
    ALPR_FLAG_LOW_CONFIDENCE_CHARS: bool = False
    ALPR_LOW_CHAR_CONFIDENCE_THRESHOLD: float = 0.70
    ALPR_ENABLE_OCR_CORRECTION: bool = False
    ALPR_OCR_CORRECTION_MAX_DISTANCE: int = 1
    ALPR_OCR_CORRECTION_MAX_QUALITY: float = 0.85
    ALPR_ENABLE_WATCHLIST_MULTI_READ: bool = False
    ALPR_WATCHLIST_MULTI_READ_PASSES: int = 3
    ALPR_WATCHLIST_NEAR_MATCH_DISTANCE: int = 1
    
    # Database Configuration (PostgreSQL)
    # TESTING: defaults to localhost PostgreSQL (docker-compose)
    DATABASE_URL: str = "postgresql+asyncpg://alpr_user:alpr_password@localhost:5432/alpr"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "alpr"
    DB_USER: str = "alpr_user"
    DB_PASSWORD: str = "alpr_password"  # TODO: Production — use secrets management (AWS Secrets Manager, etc.)

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
