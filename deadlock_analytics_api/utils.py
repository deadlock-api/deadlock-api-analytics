import logging
import uuid

from cachetools.func import ttl_cache
from fastapi import HTTPException, Security
from fastapi.openapi.models import APIKey, APIKeyIn
from fastapi.security.api_key import APIKeyBase
from starlette.requests import Request
from starlette.status import HTTP_403_FORBIDDEN

from deadlock_analytics_api.globs import postgres_conn

LOGGER = logging.getLogger(__name__)
STEAM_ID_64_IDENT = 76561197960265728


class APIKeyHeaderOrQuery(APIKeyBase):
    def __init__(
        self,
        *,
        query_name: str,
        header_name: str,
        scheme_name: str | None = None,
        description: str | None = None,
        auto_error: bool = True,
    ):
        self.model: APIKey = APIKey(
            **{"in": APIKeyIn.query},  # type: ignore[arg-type]
            name=query_name,
            description=description,
        )
        self.header_model: APIKey = APIKey(
            **{"in": APIKeyIn.header},  # type: ignore[arg-type]
            name=header_name,
            description=description,
        )
        self.scheme_name = scheme_name or self.__class__.__name__
        self.auto_error = auto_error

    async def __call__(self, request: Request) -> str | None:
        query_api_key = request.query_params.get(self.model.name)
        header_api_key = request.headers.get(self.header_model.name)
        if not query_api_key and not header_api_key:
            if self.auto_error:
                raise HTTPException(
                    status_code=HTTP_403_FORBIDDEN, detail="Not authenticated"
                )
            else:
                return None
        return query_api_key or header_api_key


api_key_param = APIKeyHeaderOrQuery(query_name="api_key", header_name="X-API-Key")


@ttl_cache(maxsize=1024, ttl=60)
def is_valid_api_key(api_key: str, data_access: bool = False) -> bool:
    api_key = api_key.lstrip("HEXE-")
    with postgres_conn().cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM api_keys WHERE key = %s AND disabled IS FALSE AND (data_access OR NOT %s)",
            (str(api_key), data_access),
        )
        return cursor.fetchone() is not None


async def get_api_key(api_key: str = Security(api_key_param)):
    if not is_valid_api_key(api_key):
        raise HTTPException(status_code=HTTP_403_FORBIDDEN)
    return api_key


async def get_data_api_key(api_key: str = Security(api_key_param)):
    try:
        if not is_valid_api_key(api_key, True):
            raise HTTPException(status_code=HTTP_403_FORBIDDEN)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=HTTP_403_FORBIDDEN)
    return api_key


async def get_internal_api_key(api_key: str = Security(api_key_param)):
    with open("internal_api_keys.txt") as f:
        available_api_keys = f.read().splitlines()
        available_api_keys = [a.split("#")[0].strip() for a in available_api_keys]
    if api_key in available_api_keys:
        return api_key
    raise HTTPException(status_code=HTTP_403_FORBIDDEN)


def is_valid_uuid(value: str) -> bool:
    if value is None:
        return False
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        LOGGER.warning(f"Invalid UUID: {value}")
        return False
    except TypeError:
        LOGGER.warning(f"Invalid UUID: {value}")
        return False


def validate_steam_id(steam_id: int | str) -> int:
    try:
        steam_id = int(steam_id)
        if steam_id >= STEAM_ID_64_IDENT:
            return steam_id - STEAM_ID_64_IDENT
        return steam_id
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
