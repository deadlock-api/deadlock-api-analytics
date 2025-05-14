import logging
from typing import Annotated, Literal

from deadlock_analytics_api import utils
from deadlock_analytics_api.globs import CH_POOL
from deadlock_analytics_api.rate_limiter import limiter
from deadlock_analytics_api.rate_limiter.models import RateLimit
from fastapi import APIRouter, Query
from pydantic import BaseModel, computed_field
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

router = APIRouter(prefix="/v1", tags=["V1"])

LOGGER = logging.getLogger("app-v1")


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
