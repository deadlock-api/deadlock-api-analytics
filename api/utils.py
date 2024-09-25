from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
from starlette.status import HTTP_403_FORBIDDEN

api_key_query = APIKeyQuery(name="api_key", auto_error=False)
api_key_header = APIKeyHeader(name="authorization", auto_error=False)


async def get_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query),
):
    with open("api_keys.txt") as f:
        available_api_keys = f.read().splitlines()
    if api_key_header in available_api_keys:
        return api_key_header
    if api_key_query in available_api_keys:
        return api_key_header
    raise HTTPException(status_code=HTTP_403_FORBIDDEN)
