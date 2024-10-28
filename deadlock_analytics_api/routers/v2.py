from datetime import datetime
from typing import Annotated, Literal

from deadlock_analytics_api.globs import CH_POOL
from deadlock_analytics_api.rate_limiter import limiter
from deadlock_analytics_api.rate_limiter.models import RateLimit
from deadlock_analytics_api.routers.v1 import HeroWinLossStat
from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field, computed_field
from starlette.requests import Request
from starlette.responses import Response

router = APIRouter(prefix="/v2", tags=["V2"])


class PlayerLeaderboardV2(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    region_mode: Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"]
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
    summary="RateLimit: 100req/s",
)
def get_leaderboard(
    req: Request,
    res: Response,
    start: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(le=10000)] = 1000,
    account_id: int | None = None,
) -> list[PlayerLeaderboardV2]:
    limiter.apply_limits(req, res, "/v2/leaderboard", [RateLimit(limit=100, period=1)])
    res.headers["Cache-Control"] = "public, max-age=300"
    if account_id is not None:
        query = """
        SELECT account_id, region_mode, rank, ranked_badge_level
        FROM leaderboard_account_v2
        WHERE account_id = %(account_id)s
        ORDER BY rank
        LIMIT 1;
        """
    else:
        query = """
        SELECT account_id, region_mode, rank, ranked_badge_level
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
            region_mode=r[1],
            leaderboard_rank=r[2],
            ranked_badge_level=r[3],
        )
        for r in result
    ]


@router.get(
    "/leaderboard/{region}",
    response_model_exclude_none=True,
    summary="RateLimit: 100req/s",
)
def get_leaderboard_by_region(
    req: Request,
    res: Response,
    region: Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"],
    start: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(le=10000)] = 1000,
) -> list[PlayerLeaderboardV2]:
    limiter.apply_limits(
        req, res, "/v2/leaderboard/{region}", [RateLimit(limit=100, period=1)]
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    query = """
    SELECT account_id, region_mode, rank() OVER (ORDER BY ranked_badge_level DESC) as rank, ranked_badge_level
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
            region_mode=r[1],
            leaderboard_rank=r[2],
            ranked_badge_level=r[3],
        )
        for r in result
    ]


@router.get("/hero-win-loss-stats", summary="RateLimit: 100req/s")
def get_hero_win_loss_stats(
    req: Request,
    res: Response,
    min_badge_level: Annotated[int | None, Query(ge=0)] = None,
    max_badge_level: Annotated[int | None, Query(le=116)] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: Annotated[int | None, Query(le=4070908800)] = None,
) -> list[HeroWinLossStat]:
    limiter.apply_limits(
        req, res, "/v2/hero-win-loss-stats", [RateLimit(limit=100, period=1)]
    )
    res.headers["Cache-Control"] = "public, max-age=1200"
    query = """
    SELECT hero_id,
            countIf(team == mi.winning_team) AS wins,
            countIf(team != mi.winning_team) AS losses
    FROM default.match_player
    INNER JOIN match_info mi USING (match_id)
    WHERE ranked_badge_level IS NULL OR (ranked_badge_level >= %(min_badge_level)s AND ranked_badge_level <= %(max_badge_level)s)
    AND mi.start_time >= toDateTime(%(min_unix_timestamp)s) AND mi.start_time <= toDateTime(%(max_unix_timestamp)s)
    GROUP BY hero_id
    ORDER BY wins + losses DESC;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query,
            {
                "min_badge_level": min_badge_level or 0,
                "max_badge_level": max_badge_level or 116,
                "min_unix_timestamp": min_unix_timestamp or 0,
                "max_unix_timestamp": max_unix_timestamp or 4070908800,
            },
        )
    return [HeroWinLossStat(hero_id=r[0], wins=r[1], losses=r[2]) for r in result]


class PlayerCardSlot(BaseModel):
    slots_id: int
    hero_id: int
    hero_kills: int
    hero_wins: int
    # stat_id: int # Always 0
    # stat_score: int # Always 0


class PlayerCardHistoryEntry(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    created_at: datetime
    slots: list[PlayerCardSlot]
    ranked_badge_level: int

    @classmethod
    def from_row(cls, row) -> "PlayerCardHistoryEntry":
        return cls(
            slots=[
                PlayerCardSlot(
                    **{
                        k.replace("slots_", "", 1): v[0] if isinstance(v, list) else v
                        for k, v in row.items()
                        if k.startswith("slots_")
                    }
                )
            ],
            **{k: v for k, v in row.items() if not k.startswith("slots_")},
        )

    @computed_field
    @property
    def match_ranked_rank(self) -> int | None:
        return (
            self.ranked_badge_level // 10
            if self.ranked_badge_level is not None
            else None
        )

    @computed_field
    @property
    def match_ranked_subrank(self) -> int | None:
        return (
            self.ranked_badge_level % 10
            if self.ranked_badge_level is not None
            else None
        )


@router.get(
    "/players/{account_id}/card-history",
    summary="RateLimit: 10req/s & 1000req/10min, API-Key RateLimit: 10req/s",
)
def get_player_card_history(
    req: Request,
    res: Response,
    account_id: Annotated[
        int, Path(description="The account id of the player, it's a SteamID3")
    ],
) -> list[PlayerCardHistoryEntry]:
    limiter.apply_limits(
        req,
        res,
        "/v2/players/{account_id}/card-history",
        [RateLimit(limit=10, period=1), RateLimit(limit=1000, period=600)],
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    query = """
    SELECT *
    FROM player_card
    WHERE account_id = %(account_id)s
    ORDER BY created_at DESC;
    """
    with CH_POOL.get_client() as client:
        result, keys = client.execute(
            query, {"account_id": account_id}, with_column_types=True
        )
    result = [{k: v for (k, _), v in zip(keys, r)} for r in result]
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    return [PlayerCardHistoryEntry.from_row(r) for r in result]


class PlayerMMRHistoryEntryV2(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    match_id: int
    has_metadata: bool = Field(False)
    match_ranked_badge_level: int | None = Field(None)

    @computed_field
    @property
    def match_ranked_rank(self) -> int | None:
        return (
            self.match_ranked_badge_level // 10
            if self.match_ranked_badge_level is not None
            else None
        )

    @computed_field
    @property
    def match_ranked_subrank(self) -> int | None:
        return (
            self.match_ranked_badge_level % 10
            if self.match_ranked_badge_level is not None
            else None
        )


@router.get(
    "/players/{account_id}/mmr-history",
    summary="RateLimit: 10req/s & 1000req/10min, API-Key RateLimit: 100req/s & 10000req/10min",
)
def get_player_mmr_history(
    req: Request,
    res: Response,
    account_id: Annotated[
        int, Path(description="The account id of the player, it's a SteamID3")
    ],
) -> list[PlayerMMRHistoryEntryV2]:
    limiter.apply_limits(
        req,
        res,
        "/v2/players/{account_id}/mmr-history",
        [RateLimit(limit=10, period=1), RateLimit(limit=1000, period=600)],
        [RateLimit(limit=100, period=1), RateLimit(limit=10000, period=600)],
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    query = """
    WITH full_mmr_history AS (
        SELECT match_id, ranked_badge_level, true as has_metadata
        FROM match_player
        WHERE account_id = 111200932

        UNION DISTINCT

        SELECT match_id, ranked_badge_level, false as has_metadata
        FROM mmr_history
        WHERE account_id = %(account_id)s AND match_id NOT IN (
            SELECT match_id
            FROM match_player
            WHERE account_id = %(account_id)s
        )
    )
    SELECT DISTINCT ON(match_id) match_id, ranked_badge_level, has_metadata
    FROM full_mmr_history
    ORDER BY match_id DESC;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"account_id": account_id})
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    return [
        PlayerMMRHistoryEntryV2(
            account_id=account_id,
            match_id=r[0],
            match_ranked_badge_level=r[1],
            has_metadata=r[2],
        )
        for r in result
    ]
