import datetime
from typing import Annotated

from deadlock_analytics_api import utils
from deadlock_analytics_api.globs import CH_POOL
from fastapi import APIRouter, Depends, Query
from fastapi.openapi.models import APIKey
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, Response

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


class PlayerLeaderboard(BaseModel):
    account_id: int
    player_score: int
    leaderboard_rank: int


@router.get(
    "/players/{account_id}/rank",
    description="""
Get the rank of a player by their account ID.

As there is no way to get the real rank of a player in the game, this endpoint uses the match scores of all matches ever played.
It runs a regression algorithm to calculate the MMR of each player and then ranks them by their MMR.
""",
    tags=["Private (API-Key only)"],
)
def get_player_rank(
    response: Response, account_id: int, api_key: APIKey = Depends(utils.get_api_key)
) -> PlayerLeaderboard:
    response.headers["Cache-Control"] = "private, max-age=1200"
    print(f"Authenticated with API key: {api_key}")
    query = """
    SELECT leaderboard.account_id, ROUND(leaderboard.player_score), leaderboard.rank
    FROM (SELECT account_id, player_score, row_number() OVER (ORDER BY player_score DESC) as rank FROM player_mmr) leaderboard
    WHERE account_id = %(account_id)s;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"account_id": account_id})
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    result = result[0]
    return PlayerLeaderboard(
        account_id=result[0], player_score=int(result[1]), leaderboard_rank=result[2]
    )


@router.get(
    "/leaderboard",
    description="""
Get the leaderboard of all players.

As there is no way to get the real rank of a player in the game, this endpoint uses the match scores of all matches ever played.
It runs a regression algorithm to calculate the MMR of each player and then ranks them by their MMR.
""",
    tags=["Private (API-Key only)"],
)
def get_leaderboard(
    response: Response,
    start: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(le=10000)] = 1000,
    api_key: APIKey = Depends(utils.get_api_key),
) -> list[PlayerLeaderboard]:
    response.headers["Cache-Control"] = "private, max-age=1200"
    print(f"Authenticated with API key: {api_key}")
    query = """
    SELECT leaderboard.account_id, ROUND(leaderboard.player_score), leaderboard.rank
    FROM (SELECT account_id, player_score, row_number() OVER (ORDER BY player_score DESC) as rank FROM player_mmr) leaderboard
    LIMIT %(limit)s
    OFFSET %(start)s;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"start": start - 1, "limit": limit})
    return [
        PlayerLeaderboard(
            account_id=r[0], player_score=int(r[1]), leaderboard_rank=r[2]
        )
        for r in result
    ]


class MatchScore(BaseModel):
    start_time: datetime.datetime
    match_id: int
    match_score: int


@router.get("/matches/{match_id}/score", tags=["Private (API-Key only)"])
def get_match_scores(
    response: Response, match_id: int, api_key: APIKey = Depends(utils.get_api_key)
) -> MatchScore:
    response.headers["Cache-Control"] = "private, max-age=1200"
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


@router.get(
    "/matches/by-account-id/{account_id}", tags=["Internal (Internal API-Key only)"]
)
def get_matches_by_account_id(
    response: Response,
    account_id: int,
    api_key: APIKey = Depends(utils.get_internal_api_key),
) -> JSONResponse:
    response.headers["Cache-Control"] = "private, max-age=86400"
    print(f"Authenticated with API key: {api_key}")
    query = """
    SELECT *
    FROM finished_matches
    ARRAY JOIN players
    WHERE players.account_id = %(account_id)s
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"account_id": account_id})
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Not found")
    keys = [
        "start_time",
        "winning_team",
        "match_id",
        "players.account_id",
        "players.team",
        "players.abandoned",
        "players.hero_id",
        "lobby_id",
        "net_worth_team_0",
        "net_worth_team_1",
        "duration_s",
        "spectators",
        "open_spectator_slots",
        "objectives_mask_team0",
        "objectives_mask_team1",
        "match_mode",
        "game_mode",
        "match_score",
        "region_mode",
        "scraped_at",
        "team0_core",
        "team0_tier1_lane1",
        "team0_tier2_lane1",
        "team0_tier1_lane2",
        "team0_tier2_lane2",
        "team0_tier1_lane3",
        "team0_tier2_lane3",
        "team0_tier1_lane4",
        "team0_tier2_lane4",
        "team0_titan",
        "team0_titan_shield_generator_1",
        "team0_titan_shield_generator_2",
        "team0_barrack_boss_lane1",
        "team0_barrack_boss_lane2",
        "team0_barrack_boss_lane3",
        "team0_barrack_boss_lane4",
        "team1_core",
        "team1_tier1_lane1",
        "team1_tier2_lane1",
        "team1_tier1_lane2",
        "team1_tier2_lane2",
        "team1_tier1_lane3",
        "team1_tier2_lane3",
        "team1_tier1_lane4",
        "team1_tier2_lane4",
        "team1_titan",
        "team1_titan_shield_generator_1",
        "team1_titan_shield_generator_2",
        "team1_barrack_boss_lane1",
        "team1_barrack_boss_lane2",
        "team1_barrack_boss_lane3",
        "team1_barrack_boss_lane4",
        "winner",
    ]

    result = [
        {
            k: col if not isinstance(col, datetime.datetime) else col.isoformat()
            for k, col in zip(keys, row)
        }
        for row in result
    ]
    return JSONResponse(content=result)
