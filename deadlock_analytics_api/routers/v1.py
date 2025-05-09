import logging
import os
from datetime import datetime, timezone
from typing import Annotated, Literal

from clickhouse_driver import Client
from fastapi import APIRouter, Depends, Path, Query
from fastapi.openapi.models import APIKey
from pydantic import BaseModel, Field, computed_field, field_validator
from starlette.datastructures import URL
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

LOGGER = logging.getLogger("app-v1")

LOGGER.setLevel(logging.INFO)


@router.get(
    "/info",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/info
    """,
    deprecated=True,
)
def get_api_info() -> RedirectResponse:
    return RedirectResponse("https://api.deadlock-api.com/v1/info", HTTP_301_MOVED_PERMANENTLY)


class MatchScoreDistribution(BaseModel):
    match_score: int
    count: int


@router.get(
    "/match-score-distribution",
    summary="RateLimit: 100req/s",
    deprecated=True,
)
def get_match_score_distribution(req: Request, res: Response) -> list[MatchScoreDistribution]:
    limiter.apply_limits(req, res, "/v1/match-score-distribution", [RateLimit(limit=100, period=1)])
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


@router.get(
    "/match-badge-level-distribution",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/matches/badge-distribution
    """,
    deprecated=True,
)
def get_match_badge_level_distribution(req: Request) -> RedirectResponse:
    url = URL("https://api.deadlock-api.com/v1/matches/badge-distribution")
    url = url.include_query_params(**{k: v for k, v in req.query_params.items() if v is not None})
    return RedirectResponse(url, HTTP_301_MOVED_PERMANENTLY)


class HeroLeaderboard(BaseModel):
    hero_id: int
    account_id: int
    wins: int
    total: int


@router.get(
    "/hero-leaderboard/{hero_id}",
    summary="RateLimit: 100req/s",
    deprecated=True,
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
    SELECT hero_id, account_id, wins, matches_played as total
    FROM player_hero_stats
    WHERE matches >= %(min_total_games)s AND hero_id = %(hero_id)s
    ORDER BY wins / matches_played DESC
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
    return [HeroLeaderboard(hero_id=r[0], account_id=r[1], wins=r[2], total=r[3]) for r in result]


class MatchSearchResult(BaseModel):
    match_id: int
    start_time: datetime
    duration_s: int
    match_mode: str
    game_mode: str

    @field_validator("start_time", mode="before")
    @classmethod
    def utc_start_time(cls, v: datetime) -> datetime:
        return v.astimezone(timezone.utc)


@router.get(
    "/matches/search-ids",
    summary="RateLimit: 100req/s",
    deprecated=True,
    include_in_schema=False,
)
def match_search_ids(
    req: Request,
    account_id: Annotated[
        int | None, Path(description="The account id of a player to search for")
    ] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    min_match_id: Annotated[int | None, Query(ge=0)] = None,
    max_match_id: int | None = None,
    min_duration_s: Annotated[int | None, Query(ge=0)] = None,
    max_duration_s: Annotated[int | None, Query(le=7000)] = None,
    min_average_badge: Annotated[int | None, Query(ge=0)] = None,
    max_average_badge: Annotated[int | None, Query(le=116)] = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
    is_high_skill_range_parties: bool | None = None,
    is_low_pri_pool: bool | None = None,
    is_new_player_pool: bool | None = None,
    limit: Annotated[int, Query(ge=1, le=100000)] = 1000,
) -> RedirectResponse:
    url = req.url_for("match_search").include_query_params(**req.query_params)
    return RedirectResponse(url=url, status_code=HTTP_301_MOVED_PERMANENTLY)


MATCH_INFO_FIELDS = [
    "match_id",
    "start_time",
    "winning_team",
    "duration_s",
    "match_outcome",
    "match_mode",
    "game_mode",
    "sample_time_s",
    "stat_type",
    "source_name",
    "objectives_mask_team0",
    "objectives_mask_team1",
    "objectives.destroyed_time_s",
    "objectives.creep_damage",
    "objectives.creep_damage_mitigated",
    "objectives.player_damage",
    "objectives.player_damage_mitigated",
    "objectives.first_damage_time_s",
    "objectives.team_objective",
    "objectives.team",
    "mid_boss.team_killed",
    "mid_boss.team_claimed",
    "mid_boss.destroyed_time_s",
    "is_high_skill_range_parties",
    "low_pri_pool",
    "new_player_pool",
    "average_badge_team0",
    "average_badge_team1",
    "created_at",
]
match_fields_markdown_list = "\n".join(f"- {f}" for f in MATCH_INFO_FIELDS)

MATCH_PLAYER_FIELDS = [
    "account_id",
    "player_slot",
    "team",
    "kills",
    "deaths",
    "assists",
    "net_worth",
    "hero_id",
    "last_hits",
    "denies",
    "ability_points",
    "party",
    "assigned_lane",
    "player_level",
    "abandon_match_time_s",
    "ability_stats",
    "stats_type_stat",
    "book_reward.book_id",
    "book_reward.xp_amount",
    "book_reward.starting_xp",
    "death_details.game_time_s",
    "death_details.killer_player_slot",
    "death_details.death_pos",
    "death_details.killer_pos",
    "death_details.death_duration_s",
    "items.game_time_s",
    "items.item_id",
    "items.upgrade_id",
    "items.sold_time_s",
    "items.flags",
    "items.imbued_ability_id",
    "stats.time_stamp_s",
    "stats.net_worth",
    "stats.gold_player",
    "stats.gold_player_orbs",
    "stats.gold_lane_creep_orbs",
    "stats.gold_neutral_creep_orbs",
    "stats.gold_boss",
    "stats.gold_boss_orb",
    "stats.gold_treasure",
    "stats.gold_denied",
    "stats.gold_death_loss",
    "stats.gold_lane_creep",
    "stats.gold_neutral_creep",
    "stats.kills",
    "stats.deaths",
    "stats.assists",
    "stats.creep_kills",
    "stats.neutral_kills",
    "stats.possible_creeps",
    "stats.creep_damage",
    "stats.player_damage",
    "stats.neutral_damage",
    "stats.boss_damage",
    "stats.denies",
    "stats.player_healing",
    "stats.ability_points",
    "stats.self_healing",
    "stats.player_damage_taken",
    "stats.max_health",
    "stats.weapon_power",
    "stats.tech_power",
    "stats.shots_hit",
    "stats.shots_missed",
    "stats.damage_absorbed",
    "stats.absorption_provided",
    "stats.hero_bullets_hit",
    "stats.hero_bullets_hit_crit",
    "stats.heal_prevented",
    "stats.heal_lost",
    "stats.damage_mitigated",
    "stats.level",
    "won",
]
match_player_fields_markdown_list = "\n".join(f"- {f}" for f in MATCH_PLAYER_FIELDS)


@router.get(
    "/matches/search",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/matches/metadata
    """,
    deprecated=True,
)
def match_search(
    req: Request,
    res: Response,
    match_info_return_fields: Annotated[
        str | None,
        Query(description=f"Possible fields:\n{match_fields_markdown_list}"),
    ] = "match_id,start_time,duration_s,match_mode,game_mode",
    match_player_return_fields: Annotated[
        str | None,
        Query(
            description=f"Possible fields:\n{match_player_fields_markdown_list}",
        ),
    ] = None,
    account_id: Annotated[
        int | None, Query(description="The account id of a player to search for")
    ] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    min_match_id: Annotated[int | None, Query(ge=0)] = None,
    max_match_id: int | None = None,
    min_duration_s: Annotated[int | None, Query(ge=0)] = None,
    max_duration_s: Annotated[int | None, Query(le=7000)] = None,
    min_average_badge: Annotated[int | None, Query(ge=0)] = None,
    max_average_badge: Annotated[int | None, Query(le=116)] = None,
    match_mode: Annotated[
        str | None,
        Query(
            description="Comma seperated List; Possible values: Unranked, PrivateLobby, CoopBot, Ranked, ServerTest, Tutorial, HeroLabs; Default: Unranked,Ranked",
        ),
    ] = None,
    is_high_skill_range_parties: bool | None = None,
    is_low_pri_pool: bool | None = None,
    is_new_player_pool: bool | None = None,
    order_by: Literal["match_id", "start_time"] | None = None,
    order: Literal["ASC", "DESC"] | None = None,
    limit: Annotated[int, Query(ge=1, le=100000)] = 1000,
) -> list[dict]:
    limiter.apply_limits(
        req,
        res,
        "/v1/matches/search",
        [RateLimit(limit=100, period=1)],
    )
    print(account_id)
    select_list = ", ".join(
        m
        for m in [
            f"any(mi.{f}) as {f}"
            for f in (match_info_return_fields or "").split(",")
            if f != "match_id" and len(f) > 0 and f in MATCH_INFO_FIELDS
        ]
        + [
            f"groupArray(mp.{f}) as players_{f.replace('.', '_')}"
            for f in (match_player_return_fields or "").split(",")
            if f != "match_id" and len(f) > 0 and f in MATCH_PLAYER_FIELDS
        ]
    )
    if match_mode is None:
        match_mode = "Unranked,Ranked"
    if order_by is None:
        order_by = "match_id"
    if order is None:
        order = "ASC"
    query = f"""
    SELECT match_id, {select_list}
    FROM match_player mp FINAL
    INNER JOIN match_info mi USING match_id
    WHERE TRUE
    AND mi.match_outcome = 'TeamWin'
    AND (%(account_id)s IS NULL OR mp.account_id = %(account_id)s)
    AND (%(min_unix_timestamp)s IS NULL OR mi.start_time >= toDateTime(%(min_unix_timestamp)s))
    AND (%(max_unix_timestamp)s IS NULL OR mi.start_time <= toDateTime(%(max_unix_timestamp)s))
    AND (%(min_match_id)s IS NULL OR mi.match_id >= %(min_match_id)s)
    AND (%(max_match_id)s IS NULL OR mi.match_id <= %(max_match_id)s)
    AND (%(min_duration_s)s IS NULL OR mi.duration_s >= %(min_duration_s)s)
    AND (%(max_duration_s)s IS NULL OR mi.duration_s <= %(max_duration_s)s)
    AND (%(min_average_badge)s IS NULL OR least(mi.average_badge_team0, mi.average_badge_team1) >= %(min_average_badge)s)
    AND (%(max_average_badge)s IS NULL OR greatest(mi.average_badge_team0, mi.average_badge_team1) <= %(max_average_badge)s)
    AND (%(match_mode)s IS NULL OR mi.match_mode IN %(match_mode)s)
    AND (%(is_high_skill_range_party)s IS NULL OR mi.is_high_skill_range_parties = %(is_high_skill_range_party)s)
    AND (%(is_low_pri_pool)s IS NULL OR mi.low_pri_pool = %(is_low_pri_pool)s)
    AND (%(is_new_player_pool)s IS NULL OR mi.new_player_pool = %(is_new_player_pool)s)
    GROUP BY match_id
    ORDER BY {order_by} {order}
    LIMIT %(limit)s
    """
    with CH_POOL.get_client() as client:
        results, keys = client.execute(
            query,
            {
                "account_id": account_id,
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "min_match_id": min_match_id,
                "max_match_id": max_match_id,
                "min_duration_s": min_duration_s,
                "max_duration_s": max_duration_s,
                "min_average_badge": min_average_badge,
                "max_average_badge": max_average_badge,
                "match_mode": match_mode.split(",") if match_mode is not None else None,
                "is_high_skill_range_party": is_high_skill_range_parties,
                "is_low_pri_pool": is_low_pri_pool,
                "is_new_player_pool": is_new_player_pool,
                "limit": limit,
            },
            with_column_types=True,
        )
    return [
        {
            k: r if not isinstance(r, datetime) else r.astimezone(timezone.utc)
            for (k, _), r in zip(keys, result)
        }
        for result in results
    ]


@router.get(
    "/matches/{match_id}/short",
    summary="RateLimit: 100req/s",
    deprecated=True,
    include_in_schema=False,
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
    deprecated=True,
    include_in_schema=False,
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
    description="# Moved to new API",
    deprecated=True,
)
def get_match_metadata(match_id: int) -> RedirectResponse:
    return RedirectResponse(
        url=f"https://data.deadlock-api.com/v1/matches/{match_id}/metadata",
        status_code=HTTP_301_MOVED_PERMANENTLY,
    )


@router.get(
    "/matches/{match_id}/raw_metadata",
    description="# Moved to new API",
    deprecated=True,
    include_in_schema=False,
)
def get_raw_metadata_file(match_id: int) -> RedirectResponse:
    return RedirectResponse(
        url=f"https://api.deadlock-api.com/v1/matches/{match_id}/metadata",
        status_code=HTTP_301_MOVED_PERMANENTLY,
    )


@no_tagged_router.get(
    "/matches/short",
    summary="RateLimit: 10req/min 100req/hour, Apply for an API-Key with data access",
    tags=["Data API-Key required"],
    deprecated=True,
    include_in_schema=False,
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
    region: (Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None) = None,
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
        port=int(os.getenv("CLICKHOUSE_NATIVE_PORT", 9000)),
        user=os.getenv("CLICKHOUSE_USERNAME", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        database=os.getenv("CLICKHOUSE_DBNAME", "default"),
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
                    str(c)
                    if not isinstance(c, datetime)
                    else c.astimezone(timezone.utc).isoformat()
                    for c in row
                )
                + "\n"
            ).encode()

    return StreamingResponse(stream())


@router.get(
    "/matches/by-account-id/{account_id}",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/players/{account_id}/match-history
    """,
    deprecated=True,
    include_in_schema=False,
)
def get_matches_by_account_id(req: Request, res: Response, account_id: int) -> list[dict]:
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
        {
            "match_id": r[0],
            "start_time": r[1].astimezone(timezone.utc),
            "ranked_badge_level": r[2],
        }
        for r in result
    ]


class HeroWinLossStat(BaseModel):
    hero_id: int
    wins: int
    losses: int
    matches: int


@router.get("/hero-win-loss-stats", summary="RateLimit: 100req/s", deprecated=True)
def get_hero_win_loss_stats(
    req: Request,
    res: Response,
    min_match_score: Annotated[int | None, Query(ge=0)] = None,
    max_match_score: Annotated[int | None, Query(le=3000)] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: Annotated[int | None, Query(le=4070908800)] = None,
) -> list[HeroWinLossStat]:
    limiter.apply_limits(req, res, "/v1/hero-win-loss-stats", [RateLimit(limit=100, period=1)])
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
    return [
        HeroWinLossStat(hero_id=r[0], wins=r[1], losses=r[2], matches=r[1] + r[2]) for r in result
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
        return self.ranked_badge_level // 10 if self.ranked_badge_level is not None else None

    @computed_field
    @property
    def ranked_subrank(self) -> int | None:
        return self.ranked_badge_level % 10 if self.ranked_badge_level is not None else None


@router.get(
    "/leaderboard",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/players/scoreboard
    """,
    deprecated=True,
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
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/players/scoreboard
    """,
    deprecated=True,
)
def get_leaderboard_by_region(
    req: Request,
    res: Response,
    region: Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"],
    start: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(le=10000)] = 1000,
) -> list[PlayerLeaderboardV1]:
    limiter.apply_limits(req, res, "/v1/leaderboard/{region}", [RateLimit(limit=100, period=1)])
    res.headers["Cache-Control"] = "public, max-age=300"
    query = """
    SELECT account_id, player_score, row_number() OVER (ORDER BY player_score DESC) as rank, matches_played, ranked_badge_level
    FROM leaderboard
    ORDER BY rank
    LIMIT %(limit)s
    OFFSET %(start)s;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"start": start - 1, "limit": limit})
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
    "/players/{account_id}/mmr-history",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/players/{account_id}/mmr-history
    """,
    deprecated=True,
)
def get_player_mmr_history(
    req: Request,
    account_id: Annotated[int, Path(description="The account id of the player, it's a SteamID3")],
) -> RedirectResponse:
    url = URL(f"https://api.deadlock-api.com/v1/players/{account_id}/mmr-history")
    url = url.include_query_params(**{k: v for k, v in req.query_params.items() if v is not None})
    return RedirectResponse(url, HTTP_301_MOVED_PERMANENTLY)


class ItemWinRateEntry(BaseModel):
    item_id: int
    total: int
    wins: int
    unique_users: int

    @computed_field
    @property
    def win_rate(self) -> float:
        return round(self.wins / self.total, 2)


@router.post(
    "/dev/win-rate-analysis", summary="Rate Limit 10req/min | API-Key Rate Limit 10req/min"
)
def post_win_rate_analysis(
    req: Request,
    res: Response,
    hero_id: int,
    excluded_item_ids: list[int] | None = None,
    required_item_ids: list[int] | None = None,
    min_badge_level: int = 80,
    min_unix_timestamp: int = 0,
) -> list[ItemWinRateEntry]:
    limiter.apply_limits(
        req,
        res,
        "/v1/dev/win-rate-analysis",
        [RateLimit(limit=10, period=60)],
        [RateLimit(limit=10, period=60)],
    )

    excluded_item_ids = excluded_item_ids or []
    required_item_ids = required_item_ids or []

    try:
        with CH_POOL.get_client() as client:

            def clean(query: str, params: dict):
                return client.substitute_params(query, params, client.connection.context)

            # Build the exclusion/requirement conditions
            exclude_conditions = [
                clean(
                    "NOT arrayExists(i -> i = %(item_id)s, items)",
                    {"item_id": item_id},
                )
                for item_id in excluded_item_ids[:30]
            ]
            require_conditions = [
                clean(
                    "arrayExists(i -> i = %(item_id)s, items)",
                    {"item_id": item_id},
                )
                for item_id in required_item_ids[:30]
            ]

            additional_conditions = " AND ".join(exclude_conditions + require_conditions)
            having_conditions = f"HAVING {additional_conditions}" if additional_conditions else ""

            query = f"""
            WITH valid_matches AS (
                SELECT
                    DISTINCT ON(match_id)
                    match_id,
                    hero_id,
                    groupArray(item_id) AS items
                FROM match_player_item_v2 mpi
                WHERE
                    mpi.start_time > %(start_time)s  AND
                    mpi.hero_id = %(hero_id)s AND
                    mpi.average_match_badge > %(min_badge_level)s
                GROUP BY match_id, hero_id
                {having_conditions}
            )
            SELECT
                hero_id,
                mpi.item_id AS item_id,
                COUNT() AS total,
                countIf(won = true) AS wins,
                count(DISTINCT mpi.account_id) AS unique_users
            FROM match_player_item_v2 mpi
            JOIN valid_matches USING (match_id)
            WHERE
                mpi.hero_id = %(hero_id)s AND
                won IS NOT NULL AND
                start_time > toDateTime(%(start_time)s)
            GROUP BY hero_id, mpi.item_id
            ORDER BY hero_id, item_id
            SETTINGS max_execution_time = 360, join_algorithm = 'auto', max_threads = 8, count_distinct_implementation = 'uniq';
            """

            result = client.execute(
                query,
                {
                    "hero_id": hero_id,
                    "min_badge_level": min_badge_level,
                    "start_time": min_unix_timestamp,
                },
            )

            entries = []
            for _hero_id, item_id, total, wins, unique_users in result:
                if total > 5:
                    entries.append(
                        ItemWinRateEntry(
                            item_id=item_id, total=total, wins=wins, unique_users=unique_users
                        )
                    )

            # For now only return the items that are the same
            return entries
    except Exception as e:
        LOGGER.error("Error in get_win_rate_analysis %s", e)
        raise HTTPException(status_code=500, detail="Internal server error in clickhouse")


class HeroLanePerformance(BaseModel):
    hero1: int
    hero2: int
    lane: Literal["solo", "duo"]
    avg_net_worth_180s_1: float
    max_net_worth_180s_1: int
    avg_net_worth_180s_2: float
    max_net_worth_180s_2: int
    avg_net_worth_360s_1: float
    max_net_worth_360s_1: int
    avg_net_worth_360s_2: float
    max_net_worth_360s_2: int
    avg_net_worth_540s_1: float
    max_net_worth_540s_1: int
    avg_net_worth_540s_2: float
    max_net_worth_540s_2: int
    avg_net_worth_720s_1: float
    max_net_worth_720s_1: int
    avg_net_worth_720s_2: float
    max_net_worth_720s_2: int
    wins_1: int
    wins_2: int
    matches_played: int


@router.get("/dev/hero/{hero_id}/lane-performance", summary="RateLimit: 100req/s")
def get_hero_lane_performance(
    req: Request,
    res: Response,
    hero_id: int,
    account_id: Annotated[
        int | None, Query(description="The account id of a player to search for")
    ] = None,
    min_match_id: Annotated[int | None, Query(ge=0)] = None,
    max_match_id: int | None = None,
    min_duration_s: Annotated[int | None, Query(ge=0)] = None,
    max_duration_s: Annotated[int | None, Query(le=7000)] = None,
    min_badge_level: Annotated[int | None, Query(ge=0)] = None,
    max_badge_level: Annotated[int | None, Query(le=116)] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Annotated[
        str | None,
        Query(
            description="Comma seperated List; Possible values: Unranked, PrivateLobby, CoopBot, Ranked, ServerTest, Tutorial, HeroLabs; Default: Unranked,Ranked",
        ),
    ] = None,
) -> list[HeroLanePerformance]:
    limiter.apply_limits(
        req,
        res,
        "/v1/dev/hero/{hero_id}/lane-performance",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=3600"
    account_id = utils.validate_steam_id(account_id)
    if match_mode is None:
        match_mode = "Unranked,Ranked"

    try:
        with CH_POOL.get_client() as client:
            if min_unix_timestamp is not None:
                query = "SELECT match_id FROM match_info WHERE start_time >= toDateTime(%(min_unix_timestamp)s) ORDER BY match_id LIMIT 1"
                result = client.execute(query, {"min_unix_timestamp": min_unix_timestamp})
                if len(result) >= 1:
                    min_match_id = (
                        max(min_match_id, result[0][0])
                        if min_match_id is not None
                        else result[0][0]
                    )

            if max_unix_timestamp is not None:
                query = "SELECT match_id FROM match_info WHERE start_time <= toDateTime(%(max_unix_timestamp)s) ORDER BY match_id DESC LIMIT 1"
                result = client.execute(query, {"max_unix_timestamp": max_unix_timestamp})
                if len(result) >= 1:
                    max_match_id = (
                        min(max_match_id, result[0][0])
                        if max_match_id is not None
                        else result[0][0]
                    )

            query = """
                SELECT
                    mp1.hero_id                 as hero1,
                    mp2.hero_id                 as hero2,
                    if(mp1.assigned_lane IN (1,6), 'solo', 'duo') as lane,
                    avg(mp1.stats.net_worth[1])  as avg_net_worth_180s_1,
                    max(mp1.stats.net_worth[1])  as max_net_worth_180s_1,
                    avg(mp2.stats.net_worth[1])  as avg_net_worth_180s_2,
                    max(mp2.stats.net_worth[1])  as max_net_worth_180s_2,
                    avg(mp1.stats.net_worth[2])  as avg_net_worth_360s_1,
                    max(mp1.stats.net_worth[2])  as max_net_worth_360s_1,
                    avg(mp2.stats.net_worth[2])  as avg_net_worth_360s_2,
                    max(mp2.stats.net_worth[2])  as max_net_worth_360s_2,
                    avg(mp1.stats.net_worth[3])  as avg_net_worth_540s_1,
                    max(mp1.stats.net_worth[3])  as max_net_worth_540s_1,
                    avg(mp2.stats.net_worth[3])  as avg_net_worth_540s_2,
                    max(mp2.stats.net_worth[3])  as max_net_worth_540s_2,
                    avg(mp1.stats.net_worth[4])  as avg_net_worth_720s_1,
                    max(mp1.stats.net_worth[4])  as max_net_worth_720s_1,
                    avg(mp2.stats.net_worth[4])  as avg_net_worth_720s_2,
                    max(mp2.stats.net_worth[4])  as max_net_worth_720s_2,
                    sum(mp1.won)                 as wins_1,
                    sum(mp2.won)                 as wins_2,
                    count()                      as matches_played
                FROM match_player mp1 FINAL
                    INNER JOIN match_info mi FINAL USING (match_id)
                    INNER JOIN match_player mp2 FINAL USING (match_id)
                PREWHERE hero1 = %(hero_id)s
                WHERE true
                    AND mp1.team != mp2.team
                    AND mp1.assigned_lane = mp2.assigned_lane
                    AND length(mp1.stats.net_worth) >= 4
                    AND length(mp2.stats.net_worth) >= 4
                    AND mi.match_outcome = 'TeamWin'
                    AND (%(account_id)s IS NULL OR mp1.account_id = %(account_id)s)
                    AND (%(min_match_id)s IS NULL OR mi.match_id >= %(min_match_id)s)
                    AND (%(max_match_id)s IS NULL OR mi.match_id <= %(max_match_id)s)
                    AND (%(min_duration_s)s IS NULL OR mi.duration_s >= %(min_duration_s)s)
                    AND (%(max_duration_s)s IS NULL OR mi.duration_s <= %(max_duration_s)s)
                    AND (%(min_badge_level)s IS NULL OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 >= %(min_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 >= %(min_badge_level)s))
                    AND (%(max_badge_level)s IS NULL OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 <= %(max_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 <= %(max_badge_level)s))
                    AND (%(match_mode)s IS NULL OR mi.match_mode IN %(match_mode)s)
                GROUP BY hero1, hero2, lane
                ORDER BY hero2, lane DESC;
            """
            result, keys = client.execute(
                query,
                {
                    "hero_id": hero_id,
                    "account_id": account_id,
                    "min_badge_level": min_badge_level,
                    "max_badge_level": max_badge_level,
                    "min_match_id": min_match_id,
                    "max_match_id": max_match_id,
                    "min_duration_s": min_duration_s,
                    "max_duration_s": max_duration_s,
                    "match_mode": match_mode.split(",") if match_mode is not None else None,
                },
                with_column_types=True,
            )
        return [HeroLanePerformance(**{k: v for (k, _), v in zip(keys, r)}) for r in result]
    except Exception as e:
        LOGGER.error("Error in duo lane %s", e)
        raise HTTPException(status_code=500, detail="Something went wrong, this has been logged.")


class HeroDuoLanePerformance(BaseModel):
    hero1: int
    hero2: int
    hero3: int
    hero4: int
    avg_net_worth_180s_1: float
    max_net_worth_180s_1: int
    avg_net_worth_180s_2: float
    max_net_worth_180s_2: int
    avg_net_worth_180s_3: float
    max_net_worth_180s_3: int
    avg_net_worth_180s_4: float
    max_net_worth_180s_4: int
    avg_net_worth_360s_1: float
    max_net_worth_360s_1: int
    avg_net_worth_360s_2: float
    max_net_worth_360s_2: int
    avg_net_worth_360s_3: float
    max_net_worth_360s_3: int
    avg_net_worth_360s_4: float
    max_net_worth_360s_4: int
    avg_net_worth_540s_1: float
    max_net_worth_540s_1: int
    avg_net_worth_540s_2: float
    max_net_worth_540s_2: int
    avg_net_worth_540s_3: float
    max_net_worth_540s_3: int
    avg_net_worth_540s_4: float
    max_net_worth_540s_4: int
    avg_net_worth_720s_1: float
    max_net_worth_720s_1: int
    avg_net_worth_720s_2: float
    max_net_worth_720s_2: int
    avg_net_worth_720s_3: float
    max_net_worth_720s_3: int
    avg_net_worth_720s_4: float
    max_net_worth_720s_4: int
    wins_1: int
    wins_3: int
    matches_played: int


@router.get("/dev/hero/{hero1_id}/{hero2_id}/duo-lane-performance", summary="RateLimit: 100req/s")
def get_hero_duo_lane_performance(
    req: Request,
    res: Response,
    hero1_id: int,
    hero2_id: int,
    min_match_id: Annotated[int | None, Query(ge=0)] = None,
    max_match_id: int | None = None,
    min_duration_s: Annotated[int | None, Query(ge=0)] = None,
    max_duration_s: Annotated[int | None, Query(le=7000)] = None,
    min_badge_level: Annotated[int | None, Query(ge=0)] = None,
    max_badge_level: Annotated[int | None, Query(le=116)] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Annotated[
        str | None,
        Query(
            description="Comma seperated List; Possible values: Unranked, PrivateLobby, CoopBot, Ranked, ServerTest, Tutorial, HeroLabs; Default: Unranked,Ranked",
        ),
    ] = None,
) -> list[HeroDuoLanePerformance]:
    limiter.apply_limits(
        req,
        res,
        "/v1/dev/hero/{hero1_id}/{hero2_id}/duo-lane-performance",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=3600"
    if match_mode is None:
        match_mode = "Unranked,Ranked"
    with CH_POOL.get_client() as client:
        if min_unix_timestamp is not None:
            query = "SELECT match_id FROM match_info WHERE start_time >= toDateTime(%(min_unix_timestamp)s) ORDER BY match_id LIMIT 1"
            result = client.execute(query, {"min_unix_timestamp": min_unix_timestamp})
            if len(result) >= 1:
                min_match_id = (
                    max(min_match_id, result[0][0]) if min_match_id is not None else result[0][0]
                )

        if max_unix_timestamp is not None:
            query = "SELECT match_id FROM match_info WHERE start_time <= toDateTime(%(max_unix_timestamp)s) ORDER BY match_id DESC LIMIT 1"
            result = client.execute(query, {"max_unix_timestamp": max_unix_timestamp})
            if len(result) >= 1:
                max_match_id = (
                    min(max_match_id, result[0][0]) if max_match_id is not None else result[0][0]
                )

        query = """
            SELECT
                mp1.hero_id as hero1,
                mp2.hero_id as hero2,
                mp3.hero_id as hero3,
                mp4.hero_id as hero4,
                avg(mp1.stats.net_worth[1])  as avg_net_worth_180s_1,
                max(mp1.stats.net_worth[1])  as max_net_worth_180s_1,
                avg(mp2.stats.net_worth[1])  as avg_net_worth_180s_2,
                max(mp2.stats.net_worth[1])  as max_net_worth_180s_2,
                avg(mp3.stats.net_worth[1])  as avg_net_worth_180s_3,
                max(mp3.stats.net_worth[1])  as max_net_worth_180s_3,
                avg(mp4.stats.net_worth[1])  as avg_net_worth_180s_4,
                max(mp4.stats.net_worth[1])  as max_net_worth_180s_4,
                avg(mp1.stats.net_worth[2])  as avg_net_worth_360s_1,
                max(mp1.stats.net_worth[2])  as max_net_worth_360s_1,
                avg(mp2.stats.net_worth[2])  as avg_net_worth_360s_2,
                max(mp2.stats.net_worth[2])  as max_net_worth_360s_2,
                avg(mp3.stats.net_worth[2])  as avg_net_worth_360s_3,
                max(mp3.stats.net_worth[2])  as max_net_worth_360s_3,
                avg(mp4.stats.net_worth[2])  as avg_net_worth_360s_4,
                max(mp4.stats.net_worth[2])  as max_net_worth_360s_4,
                avg(mp1.stats.net_worth[3])  as avg_net_worth_540s_1,
                max(mp1.stats.net_worth[3])  as max_net_worth_540s_1,
                avg(mp2.stats.net_worth[3])  as avg_net_worth_540s_2,
                max(mp2.stats.net_worth[3])  as max_net_worth_540s_2,
                avg(mp3.stats.net_worth[3])  as avg_net_worth_540s_3,
                max(mp3.stats.net_worth[3])  as max_net_worth_540s_3,
                avg(mp4.stats.net_worth[3])  as avg_net_worth_540s_4,
                max(mp4.stats.net_worth[3])  as max_net_worth_540s_4,
                avg(mp1.stats.net_worth[4])  as avg_net_worth_720s_1,
                max(mp1.stats.net_worth[4])  as max_net_worth_720s_1,
                avg(mp2.stats.net_worth[4])  as avg_net_worth_720s_2,
                max(mp2.stats.net_worth[4])  as max_net_worth_720s_2,
                avg(mp3.stats.net_worth[4])  as avg_net_worth_720s_3,
                max(mp3.stats.net_worth[4])  as max_net_worth_720s_3,
                avg(mp4.stats.net_worth[4])  as avg_net_worth_720s_4,
                max(mp4.stats.net_worth[4])  as max_net_worth_720s_4,
                sum(mp1.won)                 as wins_1,
                sum(mp3.won)                 as wins_3,
                count()                      as matches_played
            FROM match_player mp1 FINAL
                INNER JOIN match_info mi FINAL USING (match_id)
                INNER JOIN match_player mp2 FINAL USING (match_id)
                INNER JOIN match_player mp3 FINAL USING (match_id)
                INNER JOIN match_player mp4 FINAL USING (match_id)
            WHERE true
                AND mp1.hero_id = %(hero1_id)s
                AND mp2.hero_id = %(hero2_id)s
                AND mp1.team = mp2.team
                AND mp1.account_id != mp2.account_id
                AND mp3.team = mp4.team
                AND mp3.account_id != mp4.account_id
                AND mp1.team != mp3.team
                AND mp1.assigned_lane = mp2.assigned_lane
                AND mp1.assigned_lane = mp3.assigned_lane
                AND mp1.assigned_lane = mp4.assigned_lane
                AND length(mp1.stats.net_worth) >= 4
                AND length(mp2.stats.net_worth) >= 4
                AND length(mp3.stats.net_worth) >= 4
                AND length(mp4.stats.net_worth) >= 4
                AND mi.match_outcome = 'TeamWin'
                AND (%(min_match_id)s IS NULL OR mi.match_id >= %(min_match_id)s)
                AND (%(max_match_id)s IS NULL OR mi.match_id <= %(max_match_id)s)
                AND (%(min_duration_s)s IS NULL OR mi.duration_s >= %(min_duration_s)s)
                AND (%(max_duration_s)s IS NULL OR mi.duration_s <= %(max_duration_s)s)
                AND (%(min_badge_level)s IS NULL OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 >= %(min_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 >= %(min_badge_level)s))
                AND (%(max_badge_level)s IS NULL OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 <= %(max_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 <= %(max_badge_level)s))
                AND (%(match_mode)s IS NULL OR mi.match_mode IN %(match_mode)s)
            GROUP BY hero1, hero2, hero3, hero4
            ORDER BY hero2, hero3, hero4;
        """
        result, keys = client.execute(
            query,
            {
                "hero1_id": hero1_id,
                "hero2_id": hero2_id,
                "min_badge_level": min_badge_level,
                "max_badge_level": max_badge_level,
                "min_match_id": min_match_id,
                "max_match_id": max_match_id,
                "min_duration_s": min_duration_s,
                "max_duration_s": max_duration_s,
                "match_mode": match_mode.split(",") if match_mode is not None else None,
            },
            with_column_types=True,
        )
    return [HeroDuoLanePerformance(**{k: v for (k, _), v in zip(keys, r)}) for r in result]


@router.get("/dev/net-worth-win-rate-analysis/{game_time}", summary="RateLimit: 100req/s")
def get_net_worth_win_rate_analysis(
    req: Request,
    res: Response,
    game_time: int,
    min_match_id: Annotated[int | None, Query(ge=0)] = None,
    max_match_id: int | None = None,
    min_networth_advantage: Annotated[int | None, Query(ge=0)] = None,
    max_networth_advantage: int | None = None,
    min_duration_s: Annotated[int | None, Query(ge=0)] = None,
    max_duration_s: Annotated[int | None, Query(le=7000)] = None,
    min_badge_level: Annotated[int | None, Query(ge=0)] = None,
    max_badge_level: Annotated[int | None, Query(le=116)] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
) -> dict[str, int | float]:
    limiter.apply_limits(
        req,
        res,
        "/v1/dev/net-worth-win-rate-analysis",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=3600"
    if (
        max_duration_s is not None
        and min_duration_s is not None
        and max_duration_s < min_duration_s
    ):
        raise HTTPException(
            status_code=400, detail="max_duration_s must be greater than min_duration_s"
        )
    if (
        max_badge_level is not None
        and min_badge_level is not None
        and max_badge_level < min_badge_level
    ):
        raise HTTPException(
            status_code=400, detail="max_badge_level must be greater than min_badge_level"
        )
    if (
        max_networth_advantage is not None
        and min_networth_advantage is not None
        and max_networth_advantage < min_networth_advantage
    ):
        raise HTTPException(
            status_code=400,
            detail="max_networth_advantage must be greater than min_networth_advantage",
        )
    if max_match_id is not None and min_match_id is not None and max_match_id < min_match_id:
        raise HTTPException(
            status_code=400, detail="max_match_id must be greater than min_match_id"
        )
    if (
        max_unix_timestamp is not None
        and min_unix_timestamp is not None
        and max_unix_timestamp < min_unix_timestamp
    ):
        raise HTTPException(
            status_code=400, detail="max_unix_timestamp must be greater than min_unix_timestamp"
        )
    time_stamps = [
        180,
        360,
        540,
        720,
        900,
        1200,
        1500,
        1800,
        2100,
        2400,
        2700,
        3000,
        3300,
        3600,
        3900,
        4200,
        4500,
        4800,
        5100,
        5400,
        5700,
        6000,
        6300,
        6600,
        6900,
        7200,
        7500,
        7800,
        8100,
        8400,
        8700,
        9000,
        9300,
        9600,
        9900,
        10200,
        10460,
    ]
    try:
        game_time_index = time_stamps.index(game_time)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid game_time, choose from {time_stamps}")
    min_duration_s = max(min_duration_s, game_time) if min_duration_s is not None else game_time
    with CH_POOL.get_client() as client:
        if min_unix_timestamp is not None:
            query = "SELECT match_id FROM match_info WHERE start_time >= toDateTime(%(min_unix_timestamp)s) ORDER BY match_id LIMIT 1"
            result = client.execute(query, {"min_unix_timestamp": min_unix_timestamp})
            if len(result) >= 1:
                min_match_id = (
                    max(min_match_id, result[0][0]) if min_match_id is not None else result[0][0]
                )

        if max_unix_timestamp is not None:
            query = "SELECT match_id FROM match_info WHERE start_time <= toDateTime(%(max_unix_timestamp)s) ORDER BY match_id DESC LIMIT 1"
            result = client.execute(query, {"max_unix_timestamp": max_unix_timestamp})
            if len(result) >= 1:
                max_match_id = (
                    min(max_match_id, result[0][0]) if max_match_id is not None else result[0][0]
                )

        query = """
WITH net_worth_diff AS (
    SELECT mi.match_id,
        sum(if(mp.team = 'Team0', 1, 0) * mp.stats.net_worth[%(game_time_index)s]) as team0_net_worth,
        sum(if(mp.team = 'Team1', 1, 0) * mp.stats.net_worth[%(game_time_index)s]) as team1_net_worth,
        any(mi.winning_team)                                                       as winning_team
    FROM match_info mi FINAL
        INNER JOIN match_player mp FINAL USING match_id
    WHERE TRUE
    AND mi.game_mode = 'Normal'
    AND mi.match_outcome = 'TeamWin'
    AND (%(min_match_id)s IS NULL OR mi.match_id >= %(min_match_id)s)
    AND (%(max_match_id)s IS NULL OR mi.match_id <= %(max_match_id)s)
    AND (%(min_duration_s)s IS NULL OR mi.duration_s >= %(min_duration_s)s)
    AND (%(max_duration_s)s IS NULL OR mi.duration_s <= %(max_duration_s)s)
    AND (%(min_badge_level)s IS NULL OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 >= %(min_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 >= %(min_badge_level)s))
    AND (%(max_badge_level)s IS NULL OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 <= %(max_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 <= %(max_badge_level)s))
    GROUP BY mi.match_id
)
SELECT
    avg(team0_net_worth >= team1_net_worth AND winning_team = 'Team0' OR
        team1_net_worth >= team0_net_worth AND winning_team = 'Team1') as richer_wins,
    count(*)                                                           as matches
FROM net_worth_diff
WHERE TRUE
    AND (%(min_networth_advantage)s IS NULL OR abs(team0_net_worth - team1_net_worth) >= %(min_networth_advantage)s)
    AND (%(max_networth_advantage)s IS NULL OR abs(team0_net_worth - team1_net_worth) <= %(max_networth_advantage)s)
        """
        result, keys = client.execute(
            query,
            {
                "game_time_index": game_time_index + 1,
                "min_networth_advantage": min_networth_advantage,
                "max_networth_advantage": max_networth_advantage,
                "min_badge_level": min_badge_level,
                "max_badge_level": max_badge_level,
                "min_match_id": min_match_id,
                "max_match_id": max_match_id,
                "min_duration_s": min_duration_s,
                "max_duration_s": max_duration_s,
            },
            with_column_types=True,
        )
    if len(result) == 0:
        return {"richer_wins": 0, "matches": 0}
    return {k: v for (k, _), v in zip(keys, result[0])}
