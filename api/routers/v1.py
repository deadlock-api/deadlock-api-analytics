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
    result = CH_CLIENT.execute(query)
    return [{"match_score": row[0], "count": row[1]} for row in result]
