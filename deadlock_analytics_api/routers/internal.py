from fastapi import APIRouter, Depends
from fastapi.openapi.models import APIKey
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, Response

from deadlock_analytics_api import utils
from deadlock_analytics_api.globs import CH_POOL

router = APIRouter(prefix="/v1", tags=["Internal API-Key required"])


@router.get("/recent-matches")
def get_recent_matches(
    api_key: APIKey = Depends(utils.get_internal_api_key),
) -> JSONResponse:
    print(f"Authenticated with API key: {api_key}")
    query = """
    SELECT DISTINCT match_id
    FROM finished_matches
    WHERE start_time < now() - INTERVAL '4 hours'
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


class MatchSalts(BaseModel):
    match_id: int
    cluster_id: int | None = Field(None)
    metadata_salt: int | None = Field(None)
    replay_salt: int | None = Field(None)
    failed: bool | None = Field(None)


@router.post("/match-salts")
def post_match_salts(
    response: Response,
    match_salts: list[MatchSalts] | MatchSalts,
    api_key: APIKey = Depends(utils.get_internal_api_key),
) -> JSONResponse:
    response.headers["Cache-Control"] = "private, max-age=60"
    print(f"Authenticated with API key: {api_key}")
    print(f"Received match_salts: {match_salts}")
    match_salts = [match_salts] if isinstance(match_salts, MatchSalts) else match_salts
    for match_salt in match_salts:
        try:
            query = "SELECT * FROM match_salts WHERE match_id = %(match_id)s"
            with CH_POOL.get_client() as client:
                result = client.execute(query, {"match_id": match_salt.match_id})
            if len(result) > 0:
                for i in range(10):
                    print(f"Match {match_salt.match_id} already in match_salts")
            if match_salt.failed:
                query = "INSERT INTO match_salts (match_id, failed) VALUES (%(match_id)s, TRUE)"
            else:
                query = "INSERT INTO match_salts (match_id, cluster_id, metadata_salt, replay_salt) VALUES (%(match_id)s, %(cluster_id)s, %(metadata_salt)s, %(replay_salt)s)"
            with CH_POOL.get_client() as client:
                client.execute(query, match_salt.model_dump())
        except Exception as e:
            print(f"Failed to insert match_salt: {e}")
    return JSONResponse(content={"success": True})
