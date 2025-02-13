import logging
from typing import Annotated

import requests
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.openapi.models import APIKey
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from deadlock_analytics_api import utils
from deadlock_analytics_api.globs import CH_POOL
from deadlock_analytics_api.rate_limiter import limiter
from deadlock_analytics_api.rate_limiter.models import RateLimit
from deadlock_analytics_api.utils import is_internal_api_key

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
)
def get_missing_matches(
    req: Request, res: Response, limit: Annotated[int, Query(ge=1, le=100)] = 100
) -> JSONResponse:
    limiter.apply_limits(req, res, "/v1/missing-matches", [RateLimit(limit=100, period=1)])
    query = """
    WITH matches AS (
        SELECT DISTINCT match_id, toUnixTimestamp(start_time) AS start_time FROM finished_matches LIMIT 1 BY match_id
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


class MatchSalts(BaseModel):
    match_id: int
    cluster_id: int | None = Field(None)
    metadata_salt: int | None = Field(None)
    replay_salt: int | None = Field(None)
    failed: bool | None = Field(None)
    username: str | None = Field(None)


@no_key_router.post(
    "/match-salts",
    summary="Ingest match salts into the database",
    description="""
    You can use this endpoint to help us collecting data.

    The endpoint accepts a list of MatchSalts objects, which contain the following fields:

    - `match_id`: The match ID
    - `cluster_id`: The cluster ID
    - `metadata_salt`: The metadata salt
    - `replay_salt`: The replay salt
    - `username`: The username of the person who submitted the match
    """,
)
def post_match_salts(
    req: Request,
    match_salts: list[MatchSalts] | MatchSalts,
    api_key: str | None = None,
) -> JSONResponse:
    api_key = api_key or req.headers.get("X-API-Key")
    bypass_check = is_internal_api_key(api_key)

    LOGGER.debug(f"Authenticated with API key: {api_key}")
    LOGGER.info(f"Received match_salts: {match_salts}")
    match_salts = [match_salts] if isinstance(match_salts, MatchSalts) else match_salts
    errors = []
    for match_salt in match_salts:
        if not bypass_check:
            url = f"http://replay{match_salt.cluster_id}.valve.net/1422450/{match_salt.match_id}_{match_salt.metadata_salt}.meta.bz2"
            try:
                response = requests.head(url)

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Metadata not found for match {match_salt.match_id}",
                    )
            except requests.RequestException as e:
                errors.append(e)
                continue
            except HTTPException as e:
                errors.append(e)
                continue
        try:
            query = "SELECT * FROM match_salts WHERE match_id = %(match_id)s"
            with CH_POOL.get_client() as client:
                result = client.execute(query, {"match_id": match_salt.match_id})
            if len(result) > 0:
                LOGGER.warning(f"Match {match_salt.match_id} already in match_salts")
                continue
            if match_salt.failed:
                query = "INSERT INTO match_salts (match_id, failed) VALUES (%(match_id)s, TRUE)"
            else:
                query = "INSERT INTO match_salts (match_id, cluster_id, metadata_salt, replay_salt, username) VALUES (%(match_id)s, %(cluster_id)s, %(metadata_salt)s, %(replay_salt)s, %(username)s)"
            with CH_POOL.get_client() as client:
                client.execute(query, match_salt.model_dump())
        except Exception as e:
            LOGGER.error(f"Failed to insert match_salt: {e}")
    if errors:
        raise (
            errors[0]
            if isinstance(errors[0], HTTPException)
            else HTTPException(status_code=500, detail="Failed to fetch metadata")
        )
    return JSONResponse(content={"success": True})


@router.get("/recent-matches")
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
