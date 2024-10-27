import datetime
import os
from typing import Annotated, Literal

from clickhouse_driver import Client
from deadlock_analytics_api import utils
from deadlock_analytics_api.globs import CH_POOL
from deadlock_analytics_api.models.active_match import (
    ACTIVE_MATCHES_KEYS,
    ACTIVE_MATCHES_REDUCED_KEYS,
    ActiveMatch,
)
from deadlock_analytics_api.models.match_metadata import MatchMetadata
from deadlock_analytics_api.rate_limiter import limiter
from deadlock_analytics_api.rate_limiter.models import RateLimit
from fastapi import APIRouter, Depends, Path, Query
from fastapi.openapi.models import APIKey
from pydantic import BaseModel, Field, computed_field
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

router = APIRouter(prefix="/v1", tags=["V1"])


class MatchScoreDistribution(BaseModel):
    match_score: int
    count: int


@router.get("/match-score-distribution", summary="RateLimit: 10req/s")
def get_match_score_distribution(
    req: Request, res: Response
) -> list[MatchScoreDistribution]:
    limiter.apply_limits(
        req, res, "/v1/match-score-distribution", [RateLimit(limit=10, period=1)]
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


@router.get("/match-badge-level-distribution", summary="RateLimit: 10req/s")
def get_match_badge_level_distribution(
    req: Request, res: Response
) -> list[MatchBadgeLevelDistribution]:
    limiter.apply_limits(
        req, res, "/v1/match-score-distribution", [RateLimit(limit=10, period=1)]
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
    req: Request, res: Response, unix_timestamp: int | None = None
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
        WHERE ranked_badge_level > 0 AND (%(unix_timestamp)s IS NULL OR created_at <= toDateTime(%(unix_timestamp)s))
        ORDER BY created_at
        LIMIT 1 BY account_id
    )
    SELECT ranked_badge_level, COUNT(*) AS count
    FROM ranked_badge
    WHERE ranked_badge_level > 0
    GROUP BY ranked_badge_level
    ORDER BY ranked_badge_level;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"unix_timestamp": unix_timestamp})
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


class PlayerMMRHistoryEntry(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    match_id: int
    match_start_time: str
    region_mode: str
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
    summary="RateLimit: 10req/s & 1000req/10min, API-Key RateLimit: 100req/s & 10000req/10min",
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
        [RateLimit(limit=10, period=1), RateLimit(limit=1000, period=600)],
        [RateLimit(limit=100, period=1), RateLimit(limit=10000, period=600)],
    )
    res.headers["Cache-Control"] = "public, max-age=300"
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
    if account_id is not None:
        query = """
        SELECT account_id, player_score, rank, matches_played, ranked_badge_level
        FROM leaderboard_account
        WHERE account_id = %(account_id)s
        ORDER BY rank
        LIMIT 1;
        """
    else:
        query = """
        SELECT account_id, player_score, rank, matches_played, ranked_badge_level
        FROM leaderboard
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
    SELECT *
    FROM hero_player_winrate
    WHERE total >= %(min_total_games)s AND hero_id = %(hero_id)s
    ORDER BY wins / total DESC
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


@router.get("/matches/by-account-id/{account_id}", summary="RateLimit: 100req/s")
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


@router.get(
    "/matches/search",
    summary="RateLimit: 100req/min 1000req/hour, Apply for an API-Key to get higher limits",
)
def match_search(
    req: Request,
    res: Response,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    min_match_id: Annotated[int | None, Query(ge=0)] = None,
    max_match_id: int | None = None,
    min_match_score: Annotated[int | None, Query(ge=0)] = None,
    max_match_score: int | None = None,
    limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
    region: (
        Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None
    ) = None,
    hero_id: int | None = None,
    only_with_metadata: bool = False,
) -> list[ActiveMatch]:
    limiter.apply_limits(
        req,
        res,
        "/v1/matches/search",
        [RateLimit(limit=100, period=60), RateLimit(limit=1000, period=3600)],
    )
    res.headers["Cache-Control"] = "public, max-age=1200"
    if min_unix_timestamp is None:
        min_unix_timestamp = 0
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
    SELECT DISTINCT ON(match_id) {", ".join(ACTIVE_MATCHES_KEYS)}
    FROM finished_matches
    LEFT OUTER JOIN match_info mi ON mi.match_id = match_id
    WHERE start_time BETWEEN toDateTime(%(min_unix_timestamp)s) AND toDateTime(%(max_unix_timestamp)s)
    AND match_id >= %(min_match_id)s AND match_id <= %(max_match_id)s
    AND match_score >= %(min_match_score)s AND match_score <= %(max_match_score)s
    AND (%(region)s IS NULL OR region_mode = %(region)s)
    AND (%(hero_id)s IS NULL OR has(`players.hero_id`, %(hero_id)s))
    AND (%(match_mode)s IS NULL OR match_mode = %(match_mode)s)
    AND (%(only_with_metadata)s = FALSE OR mi.match_id > 0)
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
                "min_match_score": min_match_score,
                "max_match_score": max_match_score,
                "limit": limit,
                "region": region,
                "hero_id": hero_id,
                "match_mode": match_mode,
                "only_with_metadata": only_with_metadata,
            },
        )
    return [ActiveMatch.from_row(row) for row in result]


@router.get(
    "/matches/{match_id}/short",
    summary="RateLimit: 1000req/min 10000req/hour, API-Key RateLimit: 1000req/min",
)
def match_short(req: Request, res: Response, match_id: int) -> ActiveMatch:
    limiter.apply_limits(
        req,
        res,
        "/v1/matches/{match_id}/short",
        [RateLimit(limit=1000, period=60), RateLimit(limit=10000, period=3600)],
        [RateLimit(limit=1000, period=60)],
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
    summary="RateLimit: 1000req/min 10000req/hour, API-Key RateLimit: 1000req/min",
)
def match_timestamps(req: Request, res: Response, match_id: int) -> list[ActiveMatch]:
    limiter.apply_limits(
        req,
        res,
        "/v1/matches/{match_id}/timestamps",
        [RateLimit(limit=100, period=60), RateLimit(limit=1000, period=3600)],
        [RateLimit(limit=1000, period=60)],
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
    summary="RateLimit: 100req/min 1000req/hour, API-Key RateLimit: 100req/min",
)
def get_match_metadata(
    req: Request,
    res: Response,
    match_id: int,
) -> MatchMetadata:
    limiter.apply_limits(
        req,
        res,
        "/v1/matches/{match_id}/metadata",
        [RateLimit(limit=100, period=60), RateLimit(limit=1000, period=3600)],
        [RateLimit(limit=100, period=60)],
    )
    res.headers["Cache-Control"] = "public, max-age=3600"
    query = "SELECT * FROM match_info WHERE match_id = %(match_id)s LIMIT 1"
    with CH_POOL.get_client() as client:
        match_info, keys = client.execute(
            query, {"match_id": match_id}, with_column_types=True
        )
    if len(match_info) == 0:
        raise HTTPException(status_code=404, detail="Match not found")
    match_info = {k: v for (k, _), v in zip(keys, match_info[0])}

    query = "SELECT * FROM match_player WHERE match_id = %(match_id)s LIMIT 12"
    with CH_POOL.get_client() as client:
        match_players, keys = client.execute(
            query, {"match_id": match_id}, with_column_types=True
        )
    if len(match_players) == 0:
        raise HTTPException(status_code=404, detail="Match Players not found")
    match_players = [{k: v for (k, _), v in zip(keys, row)} for row in match_players]

    return MatchMetadata.from_rows(match_info, match_players)


@router.get(
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
                    (str(c) if not isinstance(c, datetime.datetime) else c.isoformat())
                    for c in row
                )
                + "\n"
            ).encode()

    return StreamingResponse(stream())
