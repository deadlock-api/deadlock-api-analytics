import os
from datetime import datetime
from typing import Annotated, Literal

from clickhouse_driver import Client
from fastapi import APIRouter, Depends, Path, Query
from fastapi.openapi.models import APIKey
from pydantic import BaseModel, Field, computed_field
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response, StreamingResponse
from starlette.status import HTTP_301_MOVED_PERMANENTLY

from deadlock_analytics_api import utils
from deadlock_analytics_api.globs import CH_POOL
from deadlock_analytics_api.models.active_match import (
    ACTIVE_MATCHES_KEYS,
    ACTIVE_MATCHES_REDUCED_KEYS,
    ActiveMatch,
)
from deadlock_analytics_api.rate_limiter import limiter
from deadlock_analytics_api.rate_limiter.models import RateLimit

router = APIRouter(prefix="/v1", tags=["V1"])
no_tagged_router = APIRouter(prefix="/v1")


class TableSize(BaseModel):
    rows: int
    is_view: bool
    data_compressed_bytes: int
    data_uncompressed_bytes: int


class APIInfo(BaseModel):
    table_sizes: dict[str, TableSize]
    fetched_matches_per_day: int


@router.get("/info", summary="RateLimit: 100req/s")
def get_api_info(req: Request, res: Response) -> APIInfo:
    limiter.apply_limits(
        req,
        res,
        "/v1/info",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    query = """
    SELECT
        name                     AS table,
        toBool(parts IS NULL)    AS is_view,
        total_rows               AS rows,
        total_bytes              AS data_compressed_bytes,
        total_bytes_uncompressed AS data_uncompressed_bytes
    FROM system.tables
    WHERE database = 'default'
        AND name NOT LIKE 'system.%'
        AND name NOT LIKE '%inner%'
    ORDER BY table;
    """
    query2 = """
    SELECT COUNT() as fetched_matches_per_day
    FROM match_salts
    WHERE created_at > now() - INTERVAL 1 DAY;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query)
        result2 = client.execute(query2)
    return APIInfo(
        table_sizes={
            r[0]: TableSize(
                is_view=r[1],
                rows=r[2],
                data_compressed_bytes=r[3],
                data_uncompressed_bytes=r[4],
            )
            for r in result
        },
        fetched_matches_per_day=result2[0][0],
    )


class MatchScoreDistribution(BaseModel):
    match_score: int
    count: int


@router.get("/match-score-distribution", summary="RateLimit: 100req/s")
def get_match_score_distribution(
    req: Request, res: Response
) -> list[MatchScoreDistribution]:
    limiter.apply_limits(
        req, res, "/v1/match-score-distribution", [RateLimit(limit=100, period=1)]
    )
    res.headers["Cache-Control"] = "public, max-age=3600"
    query = """
    SELECT match_score as score, COUNT(DISTINCT match_id) as match_score_count
    FROM finished_matches
    WHERE start_time > '2024-10-11 06:00:00'
    GROUP BY score
    ORDER BY score;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query)
    return [MatchScoreDistribution(match_score=row[0], count=row[1]) for row in result]


class MatchBadgeLevelDistribution(BaseModel):
    match_badge_level: int
    count: int

    @computed_field
    @property
    def match_ranked_rank(self) -> int | None:
        return (
            self.match_badge_level // 10 if self.match_badge_level is not None else None
        )

    @computed_field
    @property
    def match_ranked_subrank(self) -> int | None:
        return (
            self.match_badge_level % 10 if self.match_badge_level is not None else None
        )


@router.get("/match-badge-level-distribution", summary="RateLimit: 100req/s")
def get_match_badge_level_distribution(
    req: Request, res: Response
) -> list[MatchBadgeLevelDistribution]:
    limiter.apply_limits(
        req, res, "/v1/match-score-distribution", [RateLimit(limit=100, period=1)]
    )
    res.headers["Cache-Control"] = "public, max-age=3600"
    query = """
    SELECT ranked_badge_level, COUNT(DISTINCT match_id) as match_score_count
    FROM finished_matches
    WHERE ranked_badge_level IS NOT NULL
    GROUP BY ranked_badge_level
    ORDER BY ranked_badge_level;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query)
    return [
        MatchBadgeLevelDistribution(match_badge_level=row[0], count=row[1])
        for row in result
    ]


class PlayerBadgeLevelDistribution(BaseModel):
    player_badge_level: int
    count: int


@router.get("/player-badge-level-distribution", summary="RateLimit: 100req/s")
def get_player_badge_level_distribution(
    req: Request,
    res: Response,
    min_unix_timestamp: int | None = None,
    max_unix_timestamp: int | None = None,
    region: (
        Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None
    ) = None,
) -> list[PlayerBadgeLevelDistribution]:
    limiter.apply_limits(
        req,
        res,
        "/v1/player-badge-level-distribution",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=3600"
    query = """
    WITH ranked_badge AS (
        SELECT ranked_badge_level
        FROM player_card
        INNER JOIN player USING account_id
        WHERE ranked_badge_level > 0
        AND (%(region)s IS NULL OR region_mode = %(region)s)
        AND (%(min_unix_timestamp)s IS NULL OR created_at >= toDateTime(%(min_unix_timestamp)s))
        AND (%(max_unix_timestamp)s IS NULL OR created_at <= toDateTime(%(max_unix_timestamp)s))
        ORDER BY created_at DESC
        LIMIT 1 BY account_id
    )
    SELECT ranked_badge_level, COUNT(*) AS count
    FROM ranked_badge
    WHERE ranked_badge_level > 0
    GROUP BY ranked_badge_level
    ORDER BY ranked_badge_level;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query,
            {
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "region": region,
            },
        )
    return [
        PlayerBadgeLevelDistribution(player_badge_level=row[0], count=row[1])
        for row in result
    ]


class RegionDistribution(BaseModel):
    region: str
    count: int


@router.get("/match-region-distribution", summary="RateLimit: 100req/s")
def get_match_region_distribution(
    req: Request, res: Response
) -> list[RegionDistribution]:
    limiter.apply_limits(
        req, res, "/v1/match-region-distribution", [RateLimit(limit=100, period=1)]
    )
    res.headers["Cache-Control"] = "public, max-age=1200"
    query = """
    SELECT region_mode, COUNT(DISTINCT match_id) as count
    FROM active_matches
    GROUP BY region_mode
    ORDER BY region_mode;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query)
    return [RegionDistribution(region=row[0], count=row[1]) for row in result]


class HeroLeaderboard(BaseModel):
    hero_id: int
    account_id: int
    wins: int
    total: int


@router.get(
    "/hero-leaderboard/{hero_id}",
    summary="RateLimit: 100req/s",
)
def get_hero_leaderboard(
    req: Request,
    res: Response,
    hero_id: int,
    min_total_games: Annotated[int, Query(ge=10)] = 10,
    start: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[HeroLeaderboard]:
    limiter.apply_limits(
        req, res, "/v1/hero-leaderboard/{hero_id}", [RateLimit(limit=100, period=1)]
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    query = """
    SELECT hero_id, account_id, wins, matches as total
    FROM player_hero_stats
    WHERE matches >= %(min_total_games)s AND hero_id = %(hero_id)s
    ORDER BY wins / matches DESC
    LIMIT %(limit)s
    OFFSET %(start)s;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query,
            {
                "start": start - 1,
                "limit": limit,
                "hero_id": hero_id,
                "min_total_games": min_total_games,
            },
        )
    return [
        HeroLeaderboard(hero_id=r[0], account_id=r[1], wins=r[2], total=r[3])
        for r in result
    ]


class MatchSearchResult(BaseModel):
    match_id: int
    start_time: datetime
    duration_s: int
    match_mode: str
    game_mode: str


@router.get(
    "/matches/search-ids",
    summary="RateLimit: 100req/s",
    deprecated=True,
    include_in_schema=False,
)
def match_search_ids() -> RedirectResponse:
    return RedirectResponse(
        url="/v1/matches/search", status_code=HTTP_301_MOVED_PERMANENTLY
    )


@router.get(
    "/matches/search",
    summary="RateLimit: 100req/s",
)
def match_search(
    req: Request,
    res: Response,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    min_match_id: Annotated[int | None, Query(ge=0)] = None,
    max_match_id: int | None = None,
    min_duration_s: Annotated[int | None, Query(ge=0)] = None,
    max_duration_s: Annotated[int | None, Query(le=7000)] = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
    is_high_skill_range_parties: bool | None = None,
    is_low_pri_pool: bool | None = None,
    is_new_player_pool: bool | None = None,
    limit: Annotated[int, Query(ge=1, le=100000)] = 1000,
) -> list[MatchSearchResult]:
    limiter.apply_limits(
        req,
        res,
        "/v1/matches/search",
        [RateLimit(limit=100, period=1)],
    )
    query = """
    SELECT DISTINCT ON(match_id) match_id, start_time, duration_s, match_mode, game_mode
    FROM match_info
    WHERE TRUE
    AND match_outcome = 'TeamWin'
    AND (%(min_unix_timestamp)s IS NULL OR start_time >= toDateTime(%(min_unix_timestamp)s))
    AND (%(max_unix_timestamp)s IS NULL OR start_time <= toDateTime(%(max_unix_timestamp)s))
    AND (%(min_match_id)s IS NULL OR match_id >= %(min_match_id)s)
    AND (%(max_match_id)s IS NULL OR match_id <= %(max_match_id)s)
    AND (%(min_duration_s)s IS NULL OR duration_s >= %(min_duration_s)s)
    AND (%(max_duration_s)s IS NULL OR duration_s <= %(max_duration_s)s)
    AND (%(match_mode)s IS NULL OR match_mode = %(match_mode)s)
    AND (%(is_high_skill_range_party)s IS NULL OR is_high_skill_range_parties = %(is_high_skill_range_party)s)
    AND (%(is_low_pri_pool)s IS NULL OR low_pri_pool = %(is_low_pri_pool)s)
    AND (%(is_new_player_pool)s IS NULL OR new_player_pool = %(is_new_player_pool)s)
    ORDER BY match_id
    LIMIT %(limit)s
    """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query,
            {
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "min_match_id": min_match_id,
                "max_match_id": max_match_id,
                "min_duration_s": min_duration_s,
                "max_duration_s": max_duration_s,
                "match_mode": match_mode,
                "is_high_skill_range_party": is_high_skill_range_parties,
                "is_low_pri_pool": is_low_pri_pool,
                "is_new_player_pool": is_new_player_pool,
                "limit": limit,
            },
        )
    return [
        MatchSearchResult(
            match_id=r[0],
            start_time=r[1],
            duration_s=r[2],
            match_mode=r[3],
            game_mode=r[4],
        )
        for r in result
    ]


@router.get(
    "/matches/{match_id}/short",
    summary="RateLimit: 100req/s",
)
def match_short(req: Request, res: Response, match_id: int) -> ActiveMatch:
    limiter.apply_limits(
        req,
        res,
        "/v1/matches/{match_id}/short",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=1200"
    query = f"""
    SELECT {", ".join(ACTIVE_MATCHES_KEYS)}
    FROM finished_matches
    WHERE match_id = %(match_id)s
    LIMIT 1
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"match_id": match_id})
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Match not found")
    row = result[0]
    return ActiveMatch.from_row(row)


@router.get(
    "/matches/{match_id}/timestamps",
    summary="RateLimit: 100req/s",
)
def match_timestamps(req: Request, res: Response, match_id: int) -> list[ActiveMatch]:
    limiter.apply_limits(
        req,
        res,
        "/v1/matches/{match_id}/timestamps",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=1200"
    query = f"""
    SELECT DISTINCT ON(scraped_at) {", ".join(ACTIVE_MATCHES_KEYS)}
    FROM active_matches
    WHERE match_id = %(match_id)s
    ORDER BY scraped_at
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"match_id": match_id})
    return [ActiveMatch.from_row(row) for row in result]


@router.get(
    "/matches/{match_id}/metadata",
    description="# Moved to Data API",
    deprecated=True,
)
def get_match_metadata(match_id: int) -> RedirectResponse:
    return RedirectResponse(
        url=f"https://data.deadlock-api.com/v1/matches/{match_id}/metadata",
        status_code=HTTP_301_MOVED_PERMANENTLY,
    )


@router.get(
    "/matches/{match_id}/raw_metadata",
    description="# Moved to Data API",
    deprecated=True,
)
def get_raw_metadata_file(match_id: int) -> RedirectResponse:
    return RedirectResponse(
        url=f"https://data.deadlock-api.com/v1/matches/{match_id}/raw-metadata",
        status_code=HTTP_301_MOVED_PERMANENTLY,
    )


@no_tagged_router.get(
    "/matches/short",
    summary="RateLimit: 10req/min 100req/hour, Apply for an API-Key with data access",
    tags=["Data API-Key required"],
)
def get_all_finished_matches(
    req: Request,
    res: Response,
    api_key: APIKey = Depends(utils.get_data_api_key),
    limit: int | None = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=1728626400)] = None,
    max_unix_timestamp: int | None = None,
    min_match_id: Annotated[int | None, Query(ge=0)] = None,
    max_match_id: int | None = None,
    min_match_score: Annotated[int | None, Query(ge=0)] = None,
    max_match_score: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
    region: (
        Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None
    ) = None,
    hero_id: int | None = None,
) -> StreamingResponse:
    limiter.apply_limits(
        req,
        res,
        "/v1/matches",
        [RateLimit(limit=10, period=60), RateLimit(limit=100, period=3600)],
    )
    res.headers["Cache-Control"] = "private, max-age=3600"
    print(f"Authenticated with API key: {api_key}")
    if min_unix_timestamp is None:
        min_unix_timestamp = 1728626400
    min_unix_timestamp = max(min_unix_timestamp, 1728626400)
    if max_unix_timestamp is None:
        max_unix_timestamp = 4070908800
    if min_match_id is None:
        min_match_id = 0
    if max_match_id is None:
        max_match_id = 999999999
    if min_match_score is None:
        min_match_score = 0
    if max_match_score is None:
        max_match_score = 5000
    query = f"""
    SELECT {", ".join(ACTIVE_MATCHES_REDUCED_KEYS)}
    FROM finished_matches
    ARRAY JOIN players
    WHERE start_time BETWEEN toDateTime(%(min_unix_timestamp)s) AND toDateTime(%(max_unix_timestamp)s)
    AND match_id >= %(min_match_id)s AND match_id <= %(max_match_id)s
    AND match_score >= %(min_match_score)s AND match_score <= %(max_match_score)s
    AND (%(region)s IS NULL OR region_mode = %(region)s)
    AND (%(hero_id)s IS NULL OR players.hero_id = %(hero_id)s)
    AND (%(match_mode)s IS NULL OR match_mode = %(match_mode)s)
    AND start_time < now() - INTERVAL '1 day'
    LIMIT %(limit)s
    """
    client = Client(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", 9000)),
        user=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        database=os.getenv("CLICKHOUSE_DB", "default"),
    )

    async def stream():
        is_first = True
        for row in client.execute_iter(
            query,
            {
                "limit": limit if limit is not None and limit > 0 else 100_000_000,
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "min_match_id": min_match_id,
                "max_match_id": max_match_id,
                "min_match_score": min_match_score,
                "max_match_score": max_match_score,
                "region": region,
                "hero_id": hero_id,
                "match_mode": match_mode,
            },
            settings={
                "max_block_size": 1000000,
            },
            with_column_types=True,
        ):
            if is_first:
                is_first = False
                yield (",".join(c for (c, _) in row) + "\n").encode()
                continue
            yield (
                ",".join(
                    (str(c) if not isinstance(c, datetime) else c.isoformat())
                    for c in row
                )
                + "\n"
            ).encode()

    return StreamingResponse(stream())


@router.get(
    "/matches/by-account-id/{account_id}",
    deprecated=True,
    summary="RateLimit: 100req/s",
)
def get_matches_by_account_id(
    req: Request, res: Response, account_id: int
) -> list[dict]:
    limiter.apply_limits(
        req,
        res,
        "/v1/matches/by-account-id/{account_id}",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    account_id = utils.validate_steam_id(account_id)
    query = """
    SELECT match_id, start_time, ranked_badge_level
    FROM finished_matches
    ARRAY JOIN players
    WHERE players.account_id = %(account_id)s
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"account_id": account_id})
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return [
        {"match_id": r[0], "start_time": r[1].isoformat(), "ranked_badge_level": r[2]}
        for r in result
    ]


class HeroWinLossStat(BaseModel):
    hero_id: int
    wins: int
    losses: int


@router.get("/hero-win-loss-stats", summary="RateLimit: 100req/s", deprecated=True)
def get_hero_win_loss_stats(
    req: Request,
    res: Response,
    min_match_score: Annotated[int | None, Query(ge=0)] = None,
    max_match_score: Annotated[int | None, Query(le=3000)] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: Annotated[int | None, Query(le=4070908800)] = None,
) -> list[HeroWinLossStat]:
    limiter.apply_limits(
        req, res, "/v1/hero-win-loss-stats", [RateLimit(limit=100, period=1)]
    )
    if min_match_score is None:
        min_match_score = 0
    if max_match_score is None:
        max_match_score = 3000
    if min_unix_timestamp is None:
        min_unix_timestamp = 0
    if max_unix_timestamp is None:
        max_unix_timestamp = 4070908800
    res.headers["Cache-Control"] = "public, max-age=1200"
    query = """
    SELECT `players.hero_id`                  as hero_id,
            countIf(`players.team` == winner) AS wins,
            countIf(`players.team` != winner) AS losses
    FROM finished_matches
        ARRAY JOIN players
    WHERE match_score >= %(min_match_score)s AND match_score <= %(max_match_score)s
    AND start_time >= toDateTime(%(min_unix_timestamp)s) AND start_time <= toDateTime(%(max_unix_timestamp)s)
    GROUP BY `players.hero_id`
    HAVING wins + losses > 100
    ORDER BY wins + losses DESC;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query,
            {
                "min_match_score": min_match_score,
                "max_match_score": max_match_score,
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
            },
        )
    return [HeroWinLossStat(hero_id=r[0], wins=r[1], losses=r[2]) for r in result]


class PlayerLeaderboardV1(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    player_score: int
    leaderboard_rank: int
    matches_played: int | None = None
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
    deprecated=True,
    summary="RateLimit: 100req/s",
)
def get_leaderboard(
    req: Request,
    res: Response,
    start: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(le=10000)] = 1000,
    account_id: int | None = None,
) -> list[PlayerLeaderboardV1]:
    limiter.apply_limits(req, res, "/v1/leaderboard", [RateLimit(limit=100, period=1)])
    res.headers["Cache-Control"] = "public, max-age=300"
    account_id = utils.validate_steam_id(account_id)
    if account_id is not None:
        start = 1
        limit = 1
    query = """
    SELECT account_id, player_score, rank, matches_played, ranked_badge_level
    FROM leaderboard
    WHERE (%(account_id)s IS NULL OR account_id = %(account_id)s)
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
        PlayerLeaderboardV1(
            account_id=r[0],
            player_score=int(r[1]),
            leaderboard_rank=r[2],
            matches_played=r[3],
            ranked_badge_level=r[4],
        )
        for r in result
    ]


@router.get(
    "/leaderboard/{region}",
    response_model_exclude_none=True,
    deprecated=True,
    summary="RateLimit: 100req/s",
)
def get_leaderboard_by_region(
    req: Request,
    res: Response,
    region: Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"],
    start: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(le=10000)] = 1000,
) -> list[PlayerLeaderboardV1]:
    limiter.apply_limits(
        req, res, "/v1/leaderboard/{region}", [RateLimit(limit=100, period=1)]
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    query = """
    SELECT account_id, player_score, row_number() OVER (ORDER BY player_score DESC) as rank, matches_played, ranked_badge_level
    FROM leaderboard
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
        PlayerLeaderboardV1(
            account_id=r[0],
            player_score=int(r[1]),
            leaderboard_rank=r[2],
            matches_played=r[3],
            ranked_badge_level=r[4],
        )
        for r in result
    ]


class PlayerMMRHistoryEntry(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    match_id: int
    match_start_time: str
    region_mode: str | None = Field(None)
    player_score: int
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
    deprecated=True,
    summary="RateLimit: 100req/s",
)
def get_player_mmr_history(
    req: Request,
    res: Response,
    account_id: Annotated[
        int, Path(description="The account id of the player, it's a SteamID3")
    ],
) -> list[PlayerMMRHistoryEntry]:
    limiter.apply_limits(
        req,
        res,
        "/v1/players/{account_id}/mmr-history",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    account_id = utils.validate_steam_id(account_id)
    query = """
    SELECT account_id, match_id, ROUND(player_score), ranked_badge_level
    FROM mmr_history
    WHERE account_id = %(account_id)s
    ORDER BY match_id DESC;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"account_id": account_id})
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    match_ids = [r[1] for r in result]
    query = """
    SELECT match_id, start_time, region_mode
    FROM active_matches
    WHERE match_id IN %(match_ids)s
    LIMIT 1 BY match_id;
    """
    with CH_POOL.get_client() as client:
        result2 = client.execute(query, {"match_ids": match_ids})
    match_id_start_time = {r[0]: (r[1], r[2]) for r in result2}
    return [
        PlayerMMRHistoryEntry(
            account_id=r[0],
            match_id=r[1],
            match_start_time=match_id_start_time[r[1]][0].isoformat(),
            region_mode=match_id_start_time[r[1]][1],
            player_score=r[2],
            match_ranked_badge_level=r[3],
        )
        for r in result
    ]
