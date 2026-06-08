from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from config import settings

_api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)):
    if not settings.auth_enabled:
        return
    if not api_key or api_key not in settings.api_key_list:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
