import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.openapi.models import APIKey
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, RedirectResponse
from starlette.status import HTTP_301_MOVED_PERMANENTLY

from deadlock_analytics_api import utils
from deadlock_analytics_api.globs import CH_POOL
from deadlock_analytics_api.rate_limiter import limiter
from deadlock_analytics_api.rate_limiter.models import RateLimit

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Internal API-Key required"])
no_key_router = APIRouter(prefix="/v1", tags=["Internal"])


@no_key_router.get(
    "/missing-matches",
    summary="Get missing matches",
    description="""
    Get a list of missing matches that we don't have data for yet.

    You can use this endpoint to help us collect data by submitting match salts via the POST `/match-salts` endpoint.
    """,
    deprecated=True,
)
def get_missing_matches(
    req: Request, res: Response, limit: Annotated[int, Query(ge=1, le=100)] = 100
) -> JSONResponse:
    limiter.apply_limits(req, res, "/v1/missing-matches", [RateLimit(limit=100, period=1)])
    query = """
    WITH matches AS (
        SELECT DISTINCT match_id, toUnixTimestamp(start_time) AS start_time FROM finished_matches
        UNION DISTINCT
        SELECT DISTINCT match_id, start_time FROM player_match_history FINAL
        WHERE match_mode IN ('Ranked', 'Unranked')
    )
    SELECT DISTINCT match_id
    FROM matches
    WHERE start_time < now() - INTERVAL '3 hours' AND start_time > toDateTime('2024-11-01')
    AND match_id NOT IN (SELECT match_id FROM match_salts UNION DISTINCT SELECT match_id FROM match_info)
    ORDER BY rand()
    LIMIT %(limit)s
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"limit": limit})
    return JSONResponse(content=[{"match_id": r[0]} for r in result])


# class MatchSalts(BaseModel):
#     match_id: int
#     cluster_id: int | None = Field(None)
#     metadata_salt: int | None = Field(None)
#     replay_salt: int | None = Field(None)
#     failed: bool | None = Field(None)
#     username: str | None = Field(None)


@no_key_router.post(
    "/match-salts",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/matches/active
    """,
    deprecated=True,
)
def post_match_salts() -> RedirectResponse:
    return RedirectResponse(
        "https://api.deadlock-api.com/v1/matches/salts", HTTP_301_MOVED_PERMANENTLY
    )


@router.get("/recent-matches", deprecated=True)
def get_recent_matches(
    api_key: APIKey = Depends(utils.get_internal_api_key),
) -> JSONResponse:
    LOGGER.debug(f"Authenticated with API key: {api_key}")
    query = """
    SELECT DISTINCT match_id
    FROM finished_matches
    WHERE start_time < now() - INTERVAL '3 hours'
        AND start_time > now() - INTERVAL '7 days'
        AND match_id NOT IN (SELECT match_id FROM match_salts)
    ORDER BY start_time DESC
    LIMIT %(limit)s
    """
    query2 = """
    WITH (SELECT MIN(match_id) as min_match_id, MAX(match_id) as max_match_id FROM finished_matches WHERE start_time < now() - INTERVAL 2 HOUR AND start_time > now() - INTERVAL 7 DAY) AS match_range
    SELECT number + match_range.min_match_id as match_id
    FROM numbers(match_range.max_match_id - match_range.min_match_id + 1)
    WHERE (number + match_range.min_match_id) NOT IN (SELECT match_id FROM match_salts)
    ORDER BY match_id
    LIMIT %(limit)s
    """
    batch_size = 10000
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"limit": batch_size})
        if len(result) < batch_size:
            result += client.execute(query2, {"limit": batch_size - len(result)})
    return JSONResponse(content=[{"match_id": r[0]} for r in result])


# class Bundle(BaseModel):
#     path: str
#     manifest_path: str
#     match_ids: list[int]
#     created_at: datetime | None = Field(None)
#
#     @field_validator("created_at", mode="before")
#     @classmethod
#     def utc_created_at(cls, v: datetime | None = None) -> datetime | None:
#         if v is None:
#             return None
#         return v.astimezone(timezone.utc)
#
#
# @router.post("/bundles")
# def post_bundle(
#     bundle: Bundle, api_key: APIKey = Depends(utils.get_internal_api_key)
# ) -> JSONResponse:
#     LOGGER.debug(f"Authenticated with API key: {api_key}")
#     with postgres_conn().cursor() as cursor:
#         cursor.execute(
#             """
#             INSERT INTO match_bundle (match_ids, path, manifest_path, created_at)
#             VALUES (%s, %s, %s, %s)
#             """,
#             (bundle.match_ids, bundle.path, bundle.manifest_path, bundle.created_at),
#         )
#         cursor.execute("COMMIT")
#     return JSONResponse(content={"success": True})
#
#
# @router.get("/matches-to-bundle")
# def get_matches_to_bundle(
#     min_unix_timestamp: Annotated[int, Query(ge=0)],
#     max_unix_timestamp: int,
#     api_key: APIKey = Depends(utils.get_internal_api_key),
#     limit: int | None = None,
# ) -> list[int]:
#     LOGGER.debug(f"Authenticated with API key: {api_key}")
#     if max_unix_timestamp < min_unix_timestamp:
#         raise HTTPException(
#             status_code=400,
#             detail="max_unix_timestamp must be greater than or equal to min_unix_timestamp",
#         )
#     if max_unix_timestamp > (datetime.now() - timedelta(days=1)).timestamp():
#         raise HTTPException(
#             status_code=400, detail="max_unix_timestamp must be at least 24 hours ago"
#         )
#     with CH_POOL.get_client() as client:
#         all_match_ids = client.execute(
#             "SELECT DISTINCT match_id FROM match_info WHERE start_time < now() - INTERVAL 1 DAY AND start_time BETWEEN toDateTime(%(min_unix_timestamp)s) AND toDateTime(%(max_unix_timestamp)s) AND match_mode IN ('Ranked', 'Unranked') AND match_outcome = 'TeamWin'",
#             {
#                 "min_unix_timestamp": min_unix_timestamp,
#                 "max_unix_timestamp": max_unix_timestamp,
#             },
#         )
#     all_match_ids = {r[0] for r in all_match_ids}
#     with postgres_conn().cursor() as cursor:
#         cursor.execute("SELECT DISTINCT unnest(match_ids) FROM match_bundle")
#         bundled_match_ids = {r[0] for r in cursor.fetchall()}
#     match_ids = sorted(list(all_match_ids - bundled_match_ids))
#     if limit:
#         match_ids = match_ids[:limit]
#     return match_ids
