from cachetools.func import ttl_cache
from deadlock_analytics_api.globs import postgres_conn
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyQuery
from starlette.status import HTTP_403_FORBIDDEN

api_key_query = APIKeyQuery(name="api_key", auto_error=True)


@ttl_cache(maxsize=1024, ttl=60)
def is_valid_api_key(api_key: str, data_access: bool = False) -> bool:
    api_key = api_key.lstrip("HEXE-")
    with postgres_conn().cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM api_keys WHERE key = %s AND data_access = %s",
            (str(api_key), data_access),
        )
        return cursor.fetchone() is not None


async def get_api_key(api_key: str = Security(api_key_query)):
    if not is_valid_api_key(api_key):
        raise HTTPException(status_code=HTTP_403_FORBIDDEN)
    return api_key


async def get_data_api_key(api_key: str = Security(api_key_query)):
    if not is_valid_api_key(api_key, True):
        raise HTTPException(status_code=HTTP_403_FORBIDDEN)
    return api_key


async def get_internal_api_key(api_key: str = Security(api_key_query)):
    with open("internal_api_keys.txt") as f:
        available_api_keys = f.read().splitlines()
        available_api_keys = [a.split("#")[0].strip() for a in available_api_keys]
    if api_key in available_api_keys:
        return api_key
    raise HTTPException(status_code=HTTP_403_FORBIDDEN)
