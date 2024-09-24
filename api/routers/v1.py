from fastapi import APIRouter, Query
from pydantic import BaseModel

from api.globs import CH_CLIENT

router = APIRouter(prefix="/v1")


class MatchScoreDistribution(BaseModel):
    match_score: int
    count: int


@router.get("/match-score-distribution")
def get_match_score_distribution(
    hero_ids: list[int] | None = Query(None),
) -> list[MatchScoreDistribution]:
    if hero_ids is None:
        query = """
        SELECT match_score, COUNT(DISTINCT match_id) as match_score_count
        FROM active_matches
        GROUP BY match_score
        ORDER BY match_score;
        """
        result = CH_CLIENT.execute(query)
    else:
        query = """
        SELECT match_score, COUNT(DISTINCT match_id) as count
        FROM active_matches
        ARRAY JOIN players
        WHERE `players.hero_id` IN %(hero_ids)s
        GROUP BY match_score
        ORDER BY match_score;
        """
        result = CH_CLIENT.execute(query, {"hero_ids": hero_ids})
    return [MatchScoreDistribution(match_score=row[0], count=row[1]) for row in result]


class RegionDistribution(BaseModel):
    region: int
    count: int


@router.get("/match-region-distribution")
def get_match_region_distribution() -> list[RegionDistribution]:
    query = """
    SELECT region_mode, COUNT(DISTINCT match_id) as count
    FROM active_matches
    GROUP BY region_mode
    ORDER BY region_mode;
    """
    result = CH_CLIENT.execute(query)
    return [RegionDistribution(region=row[0], count=row[1]) for row in result]
