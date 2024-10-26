from typing import Annotated, Literal

from deadlock_analytics_api.globs import CH_POOL
from deadlock_analytics_api.rate_limiter import limiter
from deadlock_analytics_api.rate_limiter.models import RateLimit
from deadlock_analytics_api.routers.v1 import HeroWinLossStat
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field, computed_field
from starlette.requests import Request
from starlette.responses import Response

router = APIRouter(prefix="/v2", tags=["V2"])


class PlayerLeaderboardV2(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    leaderboard_rank: int
    ranked_badge_level: int | None = None

    @computed_field
    @property
    def ranked_rank(self) -> int | None:
        return (
            self.ranked_badge_level // 10
            if self.ranked_badge_level is not None
            else None
        )

    @computed_field
    @property
    def ranked_subrank(self) -> int | None:
        return (
            self.ranked_badge_level % 10
            if self.ranked_badge_level is not None
            else None
        )


@router.get(
    "/leaderboard",
    response_model_exclude_none=True,
    summary="RateLimit: 10req/s",
)
def get_leaderboard(
    req: Request,
    res: Response,
    start: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(le=10000)] = 1000,
    account_id: int | None = None,
) -> list[PlayerLeaderboardV2]:
    limiter.apply_limits(req, res, "/v2/leaderboard", [RateLimit(limit=10, period=1)])
    res.headers["Cache-Control"] = "public, max-age=300"
    if account_id is not None:
        query = """
        SELECT account_id, rank, ranked_badge_level
        FROM leaderboard_account_v2
        WHERE account_id = %(account_id)s
        ORDER BY rank
        LIMIT 1;
        """
    else:
        query = """
        SELECT account_id, rank, ranked_badge_level
        FROM leaderboard_v2
        ORDER BY rank
        LIMIT 1 by account_id
        LIMIT %(limit)s
        OFFSET %(start)s;
        """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query, {"start": start - 1, "limit": limit, "account_id": account_id}
        )
    return [
        PlayerLeaderboardV2(
            account_id=r[0],
            leaderboard_rank=r[1],
            ranked_badge_level=r[2],
        )
        for r in result
    ]


@router.get(
    "/leaderboard/{region}",
    response_model_exclude_none=True,
    summary="RateLimit: 10req/s",
)
def get_leaderboard_by_region(
    req: Request,
    res: Response,
    region: Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"],
    start: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(le=10000)] = 1000,
) -> list[PlayerLeaderboardV2]:
    limiter.apply_limits(
        req, res, "/v2/leaderboard/{region}", [RateLimit(limit=10, period=1)]
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    query = """
    SELECT account_id, rank() OVER (ORDER BY ranked_badge_level DESC) as rank, ranked_badge_level
    FROM leaderboard_v2
    WHERE region_mode = %(region)s
    ORDER BY rank
    LIMIT %(limit)s
    OFFSET %(start)s;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query, {"start": start - 1, "limit": limit, "region": region}
        )
    return [
        PlayerLeaderboardV2(
            account_id=r[0],
            leaderboard_rank=r[1],
            ranked_badge_level=r[2],
        )
        for r in result
    ]


@router.get("/hero-win-loss-stats", summary="RateLimit: 100req/s")
def get_hero_win_loss_stats(req: Request, res: Response) -> list[HeroWinLossStat]:
    limiter.apply_limits(
        req, res, "/v2/hero-win-loss-stats", [RateLimit(limit=100, period=1)]
    )
    res.headers["Cache-Control"] = "public, max-age=1200"
    query = """
    SELECT hero_id, SUM(wins) AS total_wins, SUM(total) - SUM(wins) AS total_losses
    FROM hero_player_winrate
    GROUP BY hero_id
    ORDER BY total_wins + total_losses DESC;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query)
    return [HeroWinLossStat(hero_id=r[0], wins=r[1], losses=r[2]) for r in result]
