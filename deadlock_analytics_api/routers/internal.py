from deadlock_analytics_api import utils
from deadlock_analytics_api.globs import CH_POOL
from fastapi import APIRouter, Depends
from fastapi.openapi.models import APIKey
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, Response

router = APIRouter(prefix="/v1", tags=["Internal API-Key required"])


@router.get("/recent-matches")
def get_recent_matches(
    response: Response,
    api_key: APIKey = Depends(utils.get_internal_api_key),
) -> JSONResponse:
    response.headers["Cache-Control"] = "private, max-age=60"
    print(f"Authenticated with API key: {api_key}")
    query = """
    SELECT DISTINCT match_id
    FROM finished_matches
    WHERE start_time < now() - INTERVAL '2 hours'
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


@router.get("/match-salts")
def get_match_salts(
    response: Response, api_key: APIKey = Depends(utils.get_internal_api_key)
) -> list[MatchSalts]:
    response.headers["Cache-Control"] = "private, max-age=1200"
    print(f"Authenticated with API key: {api_key}")
    query = """
    SELECT match_id, cluster_id, metadata_salt, replay_salt, failed
    FROM match_salts
    """
    with CH_POOL.get_client() as client:
        results = client.execute(query)
    if len(results) == 0:
        raise HTTPException(status_code=404, detail="Match not found")
    return [
        MatchSalts(
            match_id=row[0],
            cluster_id=row[1],
            metadata_salt=row[2],
            replay_salt=row[3],
            failed=row[4],
        )
        for row in results
    ]


@router.post("/match-salts")
def post_match_salts(
    response: Response,
    match_salts: MatchSalts,
    api_key: APIKey = Depends(utils.get_internal_api_key),
) -> JSONResponse:
    response.headers["Cache-Control"] = "private, max-age=60"
    print(f"Authenticated with API key: {api_key}")
    query = "SELECT * FROM match_salts WHERE match_id = %(match_id)s"
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"match_id": match_salts.match_id})
    if len(result) > 0:
        for i in range(10):
            print(f"Match {match_salts.match_id} already in match_salts")
    if match_salts.failed:
        query = "INSERT INTO match_salts (match_id, failed) VALUES (%(match_id)s, TRUE)"
    else:
        query = "INSERT INTO match_salts (match_id, cluster_id, metadata_salt, replay_salt) VALUES (%(match_id)s, %(cluster_id)s, %(metadata_salt)s, %(replay_salt)s)"
    with CH_POOL.get_client() as client:
        client.execute(query, match_salts.model_dump())
    return JSONResponse(content={"success": True})
