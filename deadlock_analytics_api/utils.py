from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyQuery
from starlette.status import HTTP_403_FORBIDDEN

api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def get_api_key(api_key_query: str = Security(api_key_query)):
    with open("api_keys.txt") as f:
        available_api_keys = f.read().splitlines()
    if api_key_query in available_api_keys:
        return api_key_query
    raise HTTPException(status_code=HTTP_403_FORBIDDEN)
