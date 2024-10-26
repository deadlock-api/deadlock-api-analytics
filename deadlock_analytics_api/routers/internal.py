from deadlock_analytics_api import utils
from deadlock_analytics_api.globs import CH_POOL
from fastapi import APIRouter, Depends
from fastapi.openapi.models import APIKey
from pydantic import BaseModel
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
    WHERE start_time < now() - INTERVAL '1 hour'
        AND start_time > now() - INTERVAL '7 days'
        AND match_id NOT IN (SELECT match_id FROM match_salts)
    ORDER BY start_time
    LIMIT 10000
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query)
    return JSONResponse(content=[{"match_id": r[0]} for r in result])


class MatchSalts(BaseModel):
    match_id: int
    cluster_id: int
    metadata_salt: int
    replay_salt: int


@router.get("/match-salts")
def get_match_salts(
    response: Response, api_key: APIKey = Depends(utils.get_internal_api_key)
) -> list[MatchSalts]:
    response.headers["Cache-Control"] = "private, max-age=1200"
    print(f"Authenticated with API key: {api_key}")
    query = """
    SELECT match_id, cluster_id, metadata_salt, replay_salt
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
    query = """
    INSERT INTO match_salts (match_id, cluster_id, metadata_salt, replay_salt)
    VALUES (%(match_id)s, %(cluster_id)s, %(metadata_salt)s, %(replay_salt)s)
    """
    with CH_POOL.get_client() as client:
        client.execute(query, match_salts.model_dump())
    return JSONResponse(content={"success": True})
