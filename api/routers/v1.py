from fastapi import APIRouter

from api.globs import CH_CLIENT

router = APIRouter(prefix="/v1")


@router.get("/match-score-distribution")
def get_match_score_distribution():
    query = """
    SELECT match_score, COUNT(match_score) as match_score_count
    FROM active_matches
    GROUP BY match_score
    ORDER BY match_score;
    """
    return CH_CLIENT.execute(query)
