import datetime

from deadlock_analytics_api import utils
from deadlock_analytics_api.globs import CH_POOL
from fastapi import APIRouter, Depends, Query
from fastapi.openapi.models import APIKey
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.responses import Response

router = APIRouter(prefix="/v1")


class MatchScoreDistribution(BaseModel):
    match_score: int
    count: int


@router.get("/match-score-distribution")
def get_match_score_distribution(
    response: Response,
    hero_ids: list[int] | None = Query(None),
) -> list[MatchScoreDistribution]:
    response.headers["Cache-Control"] = "public, max-age=1200"
    if hero_ids is None:
        query = """
        SELECT match_score, COUNT(DISTINCT match_id) as match_score_count
        FROM active_matches
        GROUP BY match_score
        ORDER BY match_score;
        """
        with CH_POOL.get_client() as client:
            result = client.execute(query)
    else:
        query = """
        SELECT match_score, COUNT(DISTINCT match_id) as count
        FROM active_matches
        ARRAY JOIN players
        WHERE `players.hero_id` IN %(hero_ids)s
        GROUP BY match_score
        ORDER BY match_score;
        """
        with CH_POOL.get_client() as client:
            result = client.execute(query, {"hero_ids": hero_ids})
    return [MatchScoreDistribution(match_score=row[0], count=row[1]) for row in result]


class RegionDistribution(BaseModel):
    region: str
    count: int


@router.get("/match-region-distribution")
def get_match_region_distribution(response: Response) -> list[RegionDistribution]:
    response.headers["Cache-Control"] = "public, max-age=1200"
    query = """
    SELECT region_mode, COUNT(DISTINCT match_id) as count
    FROM active_matches
    GROUP BY region_mode
    ORDER BY region_mode;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query)
    return [RegionDistribution(region=row[0], count=row[1]) for row in result]


class HeroWinLossStat(BaseModel):
    hero_id: int
    wins: int
    losses: int


@router.get("/hero-win-loss-stats")
def get_hero_win_loss_stats(response: Response) -> list[HeroWinLossStat]:
    response.headers["Cache-Control"] = "public, max-age=1200"
    query = """
    SELECT `players.hero_id`                  as hero_id,
            countIf(`players.team` == winner) AS wins,
            countIf(`players.team` != winner) AS losses
    FROM finished_matches
            ARRAY JOIN players
    GROUP BY `players.hero_id`
    ORDER BY wins + losses DESC;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query)
    return [HeroWinLossStat(hero_id=r[0], wins=r[1], losses=r[2]) for r in result]


class MatchScore(BaseModel):
    start_time: datetime.datetime
    match_id: int
    match_score: int


@router.get("/matches/{match_id}/score", tags=["Private (API-Key only)"])
def get_match_scores(
    response: Response, match_id: int, api_key: APIKey = Depends(utils.get_api_key)
) -> MatchScore:
    response.headers["Cache-Control"] = "public, max-age=1200"
    print(f"Authenticated with API key: {api_key}")
    query = """
    SELECT start_time, match_id, match_score
    FROM active_matches
    WHERE match_id = %(match_id)s
    LIMIT 1
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"match_id": match_id})
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Match not found")
    result = result[0]
    return MatchScore(start_time=result[0], match_id=result[1], match_score=result[2])
