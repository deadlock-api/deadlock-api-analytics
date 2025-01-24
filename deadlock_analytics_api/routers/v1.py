import os
from datetime import datetime, timezone
from typing import Annotated, Literal

import requests
from clickhouse_driver import Client
from fastapi import APIRouter, Depends, Path, Query
from fastapi.openapi.models import APIKey
from pydantic import BaseModel, Field, computed_field, field_validator
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
        AND total_rows IS NOT NULL
        AND total_bytes IS NOT NULL
        AND total_bytes_uncompressed IS NOT NULL
    ORDER BY table;
    """
    query2 = """
    WITH fetched_matches AS (
        SELECT match_id
        FROM match_info
        WHERE created_at > now() - INTERVAL 1 DAY
        UNION
        DISTINCT
        SELECT match_id
        FROM match_salts
        WHERE created_at > now() - INTERVAL 1 DAY
    )
    SELECT COUNT() as fetched_matches_per_day
    FROM fetched_matches;
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


class MatchBadgeLevelDistribution(BaseModel):
    match_badge_level: int
    count: int

    @computed_field
    @property
    def match_ranked_rank(self) -> int | None:
        return self.match_badge_level // 10 if self.match_badge_level is not None else None

    @computed_field
    @property
    def match_ranked_subrank(self) -> int | None:
        return self.match_badge_level % 10 if self.match_badge_level is not None else None


@router.get("/match-badge-level-distribution", summary="RateLimit: 100req/s")
def get_match_badge_level_distribution(
    req: Request,
    res: Response,
    min_unix_timestamp: int | None = None,
    max_unix_timestamp: int | None = None,
) -> list[MatchBadgeLevelDistribution]:
    limiter.apply_limits(req, res, "/v1/match-score-distribution", [RateLimit(limit=100, period=1)])
    res.headers["Cache-Control"] = "public, max-age=3600"
    query = """
    WITH ranked_badge AS (
        SELECT ranked_badge_level
        FROM finished_matches
        WHERE (%(min_unix_timestamp)s IS NULL OR start_time >= toDateTime(%(min_unix_timestamp)s))
        AND (%(max_unix_timestamp)s IS NULL OR start_time <= toDateTime(%(max_unix_timestamp)s))

        UNION ALL

        SELECT ranked_badge_level
        FROM match_info
        ARRAY JOIN [average_badge_team0, average_badge_team1] AS ranked_badge_level
        WHERE match_id NOT IN (SELECT match_id FROM finished_matches)
        AND (%(min_unix_timestamp)s IS NULL OR start_time >= toDateTime(%(min_unix_timestamp)s))
        AND (%(max_unix_timestamp)s IS NULL OR start_time <= toDateTime(%(max_unix_timestamp)s))
    )
    SELECT ranked_badge_level, COUNT() as match_score_count
    FROM ranked_badge
    WHERE ranked_badge_level > 0
    GROUP BY ranked_badge_level
    ORDER BY ranked_badge_level;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query,
            {"min_unix_timestamp": min_unix_timestamp, "max_unix_timestamp": max_unix_timestamp},
        )
    return [MatchBadgeLevelDistribution(match_badge_level=row[0], count=row[1]) for row in result]


class PlayerBadgeLevelDistribution(BaseModel):
    player_badge_level: int
    count: int

    @computed_field
    @property
    def player_ranked_rank(self) -> int | None:
        return self.player_badge_level // 10 if self.player_badge_level is not None else None

    @computed_field
    @property
    def player_ranked_subrank(self) -> int | None:
        return self.player_badge_level % 10 if self.player_badge_level is not None else None


@router.get("/player-badge-level-distribution", summary="RateLimit: 100req/s")
def get_player_badge_level_distribution(
    req: Request,
    res: Response,
    min_unix_timestamp: int | None = None,
    max_unix_timestamp: int | None = None,
    region: (Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None) = None,
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
        WHERE created_at <= toDateTime('2024-11-22 01:08:32')
        AND (%(region)s IS NULL OR region_mode = %(region)s)
        AND (%(min_unix_timestamp)s IS NULL OR created_at >= toDateTime(%(min_unix_timestamp)s))
        AND (%(max_unix_timestamp)s IS NULL OR created_at <= toDateTime(%(max_unix_timestamp)s))
        ORDER BY created_at DESC
        LIMIT 1 BY account_id

        UNION ALL

        SELECT multiIf(team = 'Team0', mi.average_badge_team0, team = 'Team1', mi.average_badge_team1, 0) AS ranked_badge_level
        FROM match_player
        INNER JOIN match_info mi USING (match_id)
        INNER JOIN player USING account_id
        WHERE account_id > 0
        AND created_at > toDateTime('2024-11-22 01:08:32')
        AND (%(region)s IS NULL OR region_mode = %(region)s)
        AND (%(min_unix_timestamp)s IS NULL OR start_time >= toDateTime(%(min_unix_timestamp)s))
        AND (%(max_unix_timestamp)s IS NULL OR start_time <= toDateTime(%(max_unix_timestamp)s))
        ORDER BY start_time DESC
        LIMIT 1 by account_id
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
    return [PlayerBadgeLevelDistribution(player_badge_level=row[0], count=row[1]) for row in result]


class RegionDistribution(BaseModel):
    region: str
    count: int


@router.get("/match-region-distribution", summary="RateLimit: 100req/s")
def get_match_region_distribution(req: Request, res: Response) -> list[RegionDistribution]:
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
    account_id: Annotated[int, Path(description="The account id of a player to search for")] = None,
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
    "ranked_badge_level",
    "ranked_badge_detail",
    "won",
]
match_player_fields_markdown_list = "\n".join(f"- {f}" for f in MATCH_PLAYER_FIELDS)


@router.get(
    "/matches/search",
    summary="RateLimit: 100req/s",
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
    account_id: Annotated[int, Path(description="The account id of a player to search for")] = None,
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
    limit: Annotated[int, Query(ge=1, le=100000)] = 1000,
) -> list[dict]:
    limiter.apply_limits(
        req,
        res,
        "/v1/matches/search",
        [RateLimit(limit=100, period=1)],
    )
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
    query = f"""
    SELECT match_id, {select_list}
    FROM match_player mp
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
    ORDER BY match_id
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
                    (str(c) if not isinstance(c, datetime) else c.astimezone(timezone.utc))
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
    limiter.apply_limits(req, res, "/v1/leaderboard/{region}", [RateLimit(limit=100, period=1)])
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
        result = client.execute(query, {"start": start - 1, "limit": limit, "region": region})
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
    account_id: Annotated[int, Path(description="The account id of the player, it's a SteamID3")],
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


class ItemCombWinRateEntry(BaseModel):
    item_ids: list[int]
    wins: int
    total: int
    unique_users: int
    min_distance: int | None
    max_distance: int | None

    @computed_field
    @property
    def win_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return round(self.wins / self.total, 2)


@router.get("/dev/item-win-rate-analysis/by-similarity", summary="RateLimit: 100req/s")
def get_item_comb_win_rate_by_similarity(
    req: Request,
    res: Response,
    item_ids: str | None = None,
    build_id: int | None = None,
    hero_id: int | None = None,
    min_badge_level: int | None = None,
    max_badge_level: int | None = None,
    min_match_id: int | None = None,
    max_match_id: int | None = None,
    min_used_items: int | None = None,
    max_distance: Annotated[int | None, Query(ge=0, le=1000)] = None,
    distance_function: Literal[
        "L1", "cosine", "non_matching_build_items", "non_matching_items"
    ] = None,
    k_most_similar_builds: Annotated[int, Query(ge=1, le=100000)] = 10000,
) -> ItemCombWinRateEntry:
    limiter.apply_limits(
        req,
        res,
        "/v1/dev/item-win-rate-analysis/by-similarity",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=1800"
    if item_ids is None and build_id is None:
        raise HTTPException(status_code=400, detail="Either item_ids or build_id must be provided")
    if item_ids is not None and build_id is not None:
        raise HTTPException(
            status_code=400, detail="Only one of item_ids or build_id can be provided"
        )
    if item_ids is not None:
        try:
            item_ids = [int(i) for i in item_ids.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid item_ids, must be comma separated")
    if item_ids is None and build_id is not None:
        build = requests.get(f"https://data.deadlock-api.com/v1/builds/{build_id}")
        if build.status_code != 200:
            raise HTTPException(status_code=404, detail="Build not found")
        build = build.json()
        mod_categories = build["hero_build"]["details"]["mod_categories"]
        item_ids = list({i["ability_id"] for c in mod_categories for i in c.get("mods", [])})

    if distance_function is None or distance_function == "L1":
        distance_function = "L1Distance(encoded_build_items, encoded_items)"
    elif distance_function == "cosine":
        distance_function = "cosineDistance(encoded_build_items, encoded_items)"
    elif distance_function == "non_matching_build_items":
        distance_function = (
            "arraySum(encoded_build_items) - arrayDotProduct(encoded_build_items, encoded_items)"
        )
    elif distance_function == "non_matching_items":
        distance_function = (
            "arraySum(encoded_items) - arrayDotProduct(encoded_build_items, encoded_items)"
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid distance_function")

    query = f"""
    WITH
        all_items as (SELECT groupUniqArray(item_id) as items_arr FROM items),
        build_items as (SELECT arrayMap(x -> toBool(has(%(item_ids)s, x)), arraySort(items_arr)) as encoded_build_items FROM all_items),
        relevant_matches as (
            SELECT account_id, won, {distance_function} as distance
            FROM match_player_encoded_items
                CROSS JOIN build_items
            WHERE 1=1
                AND (%(hero_id)s IS NULL OR hero_id = %(hero_id)s)
                AND (%(min_badge_level)s IS NULL OR average_badge >= %(min_badge_level)s)
                AND (%(max_badge_level)s IS NULL OR average_badge <= %(max_badge_level)s)
                AND (%(min_match_id)s IS NULL OR match_id >= %(min_match_id)s)
                AND (%(max_match_id)s IS NULL OR match_id <= %(max_match_id)s)
                AND (%(max_distance)s IS NULL OR distance <= %(max_distance)s)
                AND (%(min_used_items)s IS NULL OR arraySum(encoded_items) >= %(min_used_items)s)
            ORDER BY distance
            LIMIT %(limit)s
        )
    SELECT sum(won) as wins, count() as total, COUNT(DISTINCT account_id) as unique_accounts, min(distance) as min_distance, max(distance) as max_distance
    FROM relevant_matches
    """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query,
            {
                "item_ids": item_ids,
                "hero_id": hero_id,
                "min_badge_level": min_badge_level,
                "max_badge_level": max_badge_level,
                "min_match_id": min_match_id,
                "max_match_id": max_match_id,
                "max_distance": max_distance,
                "min_used_items": min_used_items,
                "limit": k_most_similar_builds,
            },
        )
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="No matches found")
    result = result[0]
    return ItemCombWinRateEntry(
        item_ids=item_ids,
        wins=result[0] or 0,
        total=result[1] or 0,
        unique_users=result[2] or 0,
        min_distance=result[3],
        max_distance=result[4],
    )


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
    excluded_item_ids: list[int] = [],
    required_item_ids: list[int] = [],
    min_badge_level: int = 80,
) -> list[ItemWinRateEntry]:
    limiter.apply_limits(
        req,
        res,
        "/v1/dev/win-rate-analysis",
        [RateLimit(limit=10, period=60)],
        [RateLimit(limit=10, period=60)],
    )

    START_TIME = "2024-12-06"
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

            query = f"""
            WITH valid_matches AS (
                SELECT
                    DISTINCT ON(match_id)
                    match_id,
                    hero_id,
                    groupArray(item_id) AS items
                FROM match_player_item_v2 mpi
                WHERE
                    mpi.start_time > toDateTime(%(start_time)s)  AND
                    mpi.hero_id = %(hero_id)s AND
                    mpi.average_match_badge > %(min_badge_level)s
                GROUP BY match_id, hero_id
                HAVING
                    {additional_conditions}
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
                {"hero_id": hero_id, "min_badge_level": min_badge_level, "start_time": START_TIME},
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
        print("Error in get_win_rate_analysis", e)
        raise HTTPException(status_code=500, detail="Internal server error")
