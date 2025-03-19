import itertools
from collections import defaultdict
from datetime import datetime, timezone
from typing import Annotated, Literal

from cachetools.func import ttl_cache
from fastapi import APIRouter, HTTPException, Path, Query
from starlette.datastructures import URL
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.status import HTTP_301_MOVED_PERMANENTLY

from deadlock_analytics_api import utils
from deadlock_analytics_api.globs import CH_POOL
from deadlock_analytics_api.rate_limiter import limiter
from deadlock_analytics_api.rate_limiter.models import RateLimit
from deadlock_analytics_api.routers.v2_models import (
    HeroItemWinLossStat,
    PlayerCardHistoryEntry,
    PlayerHeroStat,
    PlayerItemStat,
    PlayerLeaderboardV2,
    PlayerMMRHistoryEntryV2,
    HeroCombsWinLossStat,
    HeroWinLossStatV2,
    HeroMatchUpWinLossStat,
    HeroMatchUpWinLossStatMatchUp,
    ItemWinLossStat,
)

router = APIRouter(prefix="/v2", tags=["V2"])


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
    sort_by: Literal["winrate", "wins", "matches", "matches_played"] | None = None,
    account_id: int | None = None,
) -> list[PlayerLeaderboardV2]:
    limiter.apply_limits(req, res, "/v2/leaderboard", [RateLimit(limit=100, period=1)])
    res.headers["Cache-Control"] = "public, max-age=300"
    account_id = utils.validate_steam_id(account_id)
    if account_id is not None:
        start = 1
        limit = 1
    order_by_clause = {
        "winrate": "ORDER BY rank, wins / greatest(1, matches_played) DESC, account_id",
        "wins": "ORDER BY rank, wins DESC, account_id",
        "matches": "ORDER BY rank, matches_played DESC, account_id",
        "matches_played": "ORDER BY matches_played DESC, rank",
    }[sort_by or "winrate"]
    query = f"""
    SELECT account_id, region_mode, rank, ranked_badge_level, wins, matches_played, kills, deaths, assists
    FROM leaderboard_v2
    WHERE (%(account_id)s IS NULL OR account_id = %(account_id)s)
    {order_by_clause}
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
            wins=r[4],
            matches_played=r[5],
            kills=r[6],
            deaths=r[7],
            assists=r[8],
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
    sort_by: Literal["winrate", "wins", "matches", "matches_played"] | None = None,
) -> list[PlayerLeaderboardV2]:
    limiter.apply_limits(req, res, "/v2/leaderboard/{region}", [RateLimit(limit=100, period=1)])
    res.headers["Cache-Control"] = "public, max-age=300"
    order_by_clause = {
        "winrate": "ORDER BY rank, wins / greatest(1, matches_played) DESC, account_id",
        "wins": "ORDER BY rank, wins DESC, account_id",
        "matches": "ORDER BY rank, matches_played DESC, account_id",
        "matches_played": "ORDER BY matches_played DESC, rank",
    }[sort_by or "winrate"]
    query = f"""
    SELECT account_id, region_mode, rank() OVER (ORDER BY ranked_badge_level DESC) as rank, ranked_badge_level, wins, matches_played, kills, deaths, assists
    FROM leaderboard_v2
    WHERE region_mode = %(region)s
    {order_by_clause}
    LIMIT %(limit)s
    OFFSET %(start)s;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"start": start - 1, "limit": limit, "region": region})
    return [
        PlayerLeaderboardV2(
            account_id=r[0],
            region_mode=r[1],
            leaderboard_rank=r[2],
            ranked_badge_level=r[3],
            wins=r[4],
            matches_played=r[5],
            kills=r[6],
            deaths=r[7],
            assists=r[8],
        )
        for r in result
    ]


@router.get(
    "/hero-win-loss-stats",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/analytics/hero-win-loss-stats
    """,
    deprecated=True,
)
def get_hero_win_loss_stats(
    req: Request,
    res: Response,
    min_badge_level: Annotated[int | None, Query(ge=0)] = None,
    max_badge_level: Annotated[int | None, Query(le=116)] = None,
    min_hero_matches_per_player: Annotated[int | None, Query(ge=0)] = None,
    max_hero_matches_per_player: int = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
    region: (Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None) = None,
) -> list[HeroWinLossStatV2]:
    limiter.apply_limits(req, res, "/v2/hero-win-loss-stats", [RateLimit(limit=100, period=1)])
    res.headers["Cache-Control"] = "public, max-age=1200"
    query = """
    WITH filtered_players AS (
        SELECT
            hero_id,
            account_id,
            countIf(won) AS player_wins,
            countIf(not won) AS player_losses,
            sum(kills) AS kills,
            sum(deaths) AS deaths,
            sum(assists) AS assists
        FROM match_player FINAL
        INNER JOIN match_info mi USING (match_id)
        INNER JOIN player p USING (account_id)
        WHERE 1=1
        AND mi.match_outcome = 'TeamWin'
        AND mi.match_mode IN ('Ranked', 'Unranked')
        AND (%(min_badge_level)s IS NULL OR (ranked_badge_level IS NOT NULL AND ranked_badge_level >= %(min_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 >= %(min_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 >= %(min_badge_level)s))
        AND (%(max_badge_level)s IS NULL OR (ranked_badge_level IS NOT NULL AND ranked_badge_level <= %(max_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 <= %(max_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 <= %(max_badge_level)s))
        AND (%(min_unix_timestamp)s IS NULL OR mi.start_time >= toDateTime(%(min_unix_timestamp)s))
        AND (%(max_unix_timestamp)s IS NULL OR mi.start_time <= toDateTime(%(max_unix_timestamp)s))
        AND (%(match_mode)s IS NULL OR mi.match_mode = %(match_mode)s)
        AND (%(region)s IS NULL OR p.region_mode = %(region)s)
        GROUP BY hero_id, account_id
    )
    SELECT
        hero_id,
        sum(player_wins) AS wins,
        sum(player_losses) AS losses,
        sum(kills) AS total_kills,
        sum(deaths) AS total_deaths,
        sum(assists) AS total_assists
    FROM filtered_players
    WHERE (%(min_hero_matches)s IS NULL OR player_wins + player_losses <= %(min_hero_matches)s)
    AND (%(max_hero_matches)s IS NULL OR player_wins + player_losses <= %(max_hero_matches)s)
    GROUP BY hero_id
    ORDER BY wins + losses DESC;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query,
            {
                "min_badge_level": min_badge_level,
                "max_badge_level": max_badge_level,
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "match_mode": match_mode,
                "region": region,
                "min_hero_matches": min_hero_matches_per_player,
                "max_hero_matches": max_hero_matches_per_player,
            },
        )
    return [
        HeroWinLossStatV2(
            hero_id=r[0],
            wins=r[1],
            losses=r[2],
            matches=r[1] + r[2],
            total_kills=r[3],
            total_deaths=r[4],
            total_assists=r[5],
        )
        for r in result
    ]


@router.get("/hero-combs-win-loss-stats", summary="RateLimit: 100req/s")
def get_hero_combs_win_loss_stats(
    req: Request,
    res: Response,
    comb_size: Annotated[int, Query(ge=2, le=6, description="Size of the hero combination")] = 6,
    include_hero_ids: Annotated[
        str, Query(description="Comma separated hero ids that must be included")
    ] = None,
    exclude_hero_ids: Annotated[
        str, Query(description="Comma separated hero ids that must be excluded")
    ] = None,
    min_total_matches: Annotated[
        int | None, Query(ge=0, description="Minimum total matches")
    ] = None,
    sorted_by: Annotated[
        Literal["winrate", "wins", "matches"] | None,
        Query(description="Sort output descending by this field"),
    ] = None,
    limit: Annotated[int | None, Query(ge=0)] = None,
    min_badge_level: Annotated[int | None, Query(ge=0)] = None,
    max_badge_level: Annotated[int | None, Query(le=116)] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
    region: (Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None) = None,
) -> list[HeroCombsWinLossStat]:
    limiter.apply_limits(
        req, res, "/v2/hero-combs-win-loss-stats", [RateLimit(limit=100, period=1)]
    )
    res.headers["Cache-Control"] = "public, max-age=1200"

    # validate include_hero_ids and exclude_hero_ids
    include_hero_ids = [
        int(h.strip()) for h in (include_hero_ids.split(",") if include_hero_ids else [])
    ]
    if len(include_hero_ids) > comb_size:
        raise HTTPException(
            status_code=400,
            detail="include_hero_ids can't have more elements than comb_size",
        )
    exclude_hero_ids = [
        int(h.strip()) for h in (exclude_hero_ids.split(",") if exclude_hero_ids else [])
    ]
    if (
        include_hero_ids
        and exclude_hero_ids
        and any(h in include_hero_ids for h in exclude_hero_ids)
    ):
        raise HTTPException(
            status_code=400,
            detail="include_hero_ids and exclude_hero_ids can't have common elements",
        )

    query = """
    WITH hero_combinations AS (
        SELECT
            arraySort(groupUniqArray(6)(hero_id)) AS hero_ids,
            countIf(won) AS team_wins,
            countIf(not won) AS team_losses,
            sum(kills) AS kills,
            sum(deaths) AS deaths,
            sum(assists) AS assists
        FROM match_player FINAL
        INNER JOIN match_info mi USING (match_id)
        INNER JOIN player p USING (account_id)
        WHERE 1=1
        AND mi.match_outcome = 'TeamWin'
        AND mi.match_mode IN ('Ranked', 'Unranked')
        AND (%(min_badge_level)s IS NULL OR (ranked_badge_level IS NOT NULL AND ranked_badge_level >= %(min_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 >= %(min_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 >= %(min_badge_level)s))
        AND (%(max_badge_level)s IS NULL OR (ranked_badge_level IS NOT NULL AND ranked_badge_level <= %(max_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 <= %(max_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 <= %(max_badge_level)s))
        AND (%(min_unix_timestamp)s IS NULL OR mi.start_time >= toDateTime(%(min_unix_timestamp)s))
        AND (%(max_unix_timestamp)s IS NULL OR mi.start_time <= toDateTime(%(max_unix_timestamp)s))
        AND (%(match_mode)s IS NULL OR mi.match_mode = %(match_mode)s)
        AND (%(region)s IS NULL OR p.region_mode = %(region)s)
        GROUP BY match_id, team
    )
    SELECT
        hero_ids,
        sum(team_wins) / 6 AS wins,
        sum(team_losses) / 6 AS losses,
        sum(kills) AS total_kills,
        sum(deaths) AS total_deaths,
        sum(assists) AS total_assists
    FROM hero_combinations
    WHERE 1=1
    AND length(hero_ids) = 6
    AND arrayAll(x -> has(hero_ids, x), %(include_hero_ids)s)
    AND NOT arrayExists(x -> has(hero_ids, x), %(exclude_hero_ids)s)
    GROUP BY hero_ids
    """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query,
            {
                "min_badge_level": min_badge_level,
                "max_badge_level": max_badge_level,
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "match_mode": match_mode,
                "region": region,
                "include_hero_ids": include_hero_ids,
                "exclude_hero_ids": exclude_hero_ids,
            },
        )
    if comb_size == 6:
        comb_stats = [
            HeroCombsWinLossStat(
                hero_ids=heroes,
                wins=wins,
                losses=losses,
                matches=wins + losses,
                total_kills=kills,
                total_deaths=deaths,
                total_assists=assists,
            )
            for heroes, wins, losses, kills, deaths, assists in result
            if min_total_matches is None or wins + losses >= min_total_matches
        ]
    else:
        comb_stats = defaultdict(lambda: [0, 0, 0, 0, 0])
        for hero_ids, wins, losses, kills, deaths, assists in result:
            for hero_comb in itertools.combinations(hero_ids, comb_size):
                if include_hero_ids and not all(h in hero_comb for h in include_hero_ids):
                    continue
                comb_stats[hero_comb][0] += wins
                comb_stats[hero_comb][1] += losses
                comb_stats[hero_comb][2] += kills
                comb_stats[hero_comb][3] += deaths
                comb_stats[hero_comb][4] += assists

        comb_stats = [
            HeroCombsWinLossStat(
                hero_ids=list(heroes),
                wins=wins,
                losses=losses,
                matches=wins + losses,
                total_kills=kills,
                total_deaths=deaths,
                total_assists=assists,
            )
            for heroes, (wins, losses, kills, deaths, assists) in comb_stats.items()
            if min_total_matches is None or wins + losses >= min_total_matches
        ]
    match sorted_by:
        case "winrate":
            comb_stats.sort(key=lambda x: x.wins / max(1, x.wins + x.losses), reverse=True)
        case "wins":
            comb_stats.sort(key=lambda x: x.wins, reverse=True)
        case "matches":
            comb_stats.sort(key=lambda x: x.wins + x.losses, reverse=True)
    if limit is not None:
        comb_stats = comb_stats[:limit]
    return comb_stats


@router.get("/hero-matchups-win-loss-stats", summary="RateLimit: 100req/s")
def get_hero_matchups_win_loss_stats(
    req: Request,
    res: Response,
    min_badge_level: Annotated[int | None, Query(ge=0)] = None,
    max_badge_level: Annotated[int | None, Query(le=116)] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
    region: (Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None) = None,
) -> list[HeroMatchUpWinLossStat]:
    limiter.apply_limits(
        req, res, "/v2/hero-matchup-win-loss-stats", [RateLimit(limit=100, period=1)]
    )
    res.headers["Cache-Control"] = "public, max-age=1200"

    query = """
    WITH hero_combinations AS (
        SELECT
            groupArrayIf(hero_id, team = 'Team0') AS team0_hero_id,
            groupArrayIf(hero_id, team = 'Team1') AS team1_hero_id,
            sumIf(won, team = 'Team0')            AS team0_wins,
            sumIf(won, team = 'Team1')            AS team1_wins,
            sumIf(kills, team = 'Team0')          AS team0_kills,
            sumIf(kills, team = 'Team1')          AS team1_kills,
            sumIf(deaths, team = 'Team0')         AS team0_deaths,
            sumIf(deaths, team = 'Team1')         AS team1_deaths,
            sumIf(assists, team = 'Team0')        AS team0_assists,
            sumIf(assists, team = 'Team1')        AS team1_assists
        FROM match_player FINAL
        INNER JOIN match_info mi USING (match_id)
        INNER JOIN player p USING (account_id)
        WHERE 1=1
        AND mi.match_outcome = 'TeamWin'
        AND mi.match_mode IN ('Ranked', 'Unranked')
        AND (%(min_badge_level)s IS NULL OR (ranked_badge_level IS NOT NULL AND ranked_badge_level >= %(min_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 >= %(min_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 >= %(min_badge_level)s))
        AND (%(max_badge_level)s IS NULL OR (ranked_badge_level IS NOT NULL AND ranked_badge_level <= %(max_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 <= %(max_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 <= %(max_badge_level)s))
        AND (%(min_unix_timestamp)s IS NULL OR mi.start_time >= toDateTime(%(min_unix_timestamp)s))
        AND (%(max_unix_timestamp)s IS NULL OR mi.start_time <= toDateTime(%(max_unix_timestamp)s))
        AND (%(match_mode)s IS NULL OR mi.match_mode = %(match_mode)s)
        AND (%(region)s IS NULL OR p.region_mode = %(region)s)
        GROUP BY match_id, assigned_lane
        HAVING length(team0_hero_id) = 1
        AND length(team1_hero_id) = 1
        AND team0_hero_id != team1_hero_id)
    SELECT
        least(team0_hero_id[1], team1_hero_id[1])                                  AS hero0,
        greatest(team0_hero_id[1], team1_hero_id[1])                               AS hero1,
        sum(if(team0_hero_id[1] < team1_hero_id[1], team0_wins, team1_wins))       AS hero0_wins,
        sum(if(team0_hero_id[1] < team1_hero_id[1], team0_kills, team1_kills))     AS hero0_kills,
        sum(if(team0_hero_id[1] < team1_hero_id[1], team0_deaths, team1_deaths))   AS hero0_deaths,
        sum(if(team0_hero_id[1] < team1_hero_id[1], team0_assists, team1_assists)) AS hero0_assists,
        sum(if(team0_hero_id[1] < team1_hero_id[1], team1_wins, team0_wins))       AS hero1_wins,
        sum(if(team0_hero_id[1] < team1_hero_id[1], team1_kills, team0_kills))     AS hero1_kills,
        sum(if(team0_hero_id[1] < team1_hero_id[1], team1_deaths, team0_deaths))   AS hero1_deaths,
        sum(if(team0_hero_id[1] < team1_hero_id[1], team1_assists, team0_assists)) AS hero1_assists
    FROM hero_combinations
    GROUP BY hero0, hero1
    ORDER BY hero0, hero1;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query,
            {
                "min_badge_level": min_badge_level,
                "max_badge_level": max_badge_level,
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "match_mode": match_mode,
                "region": region,
            },
        )
    matchups = defaultdict(list)
    for (
        hero0,
        hero1,
        hero0_wins,
        hero0_kills,
        hero0_deaths,
        hero0_assists,
        hero1_wins,
        hero1_kills,
        hero1_deaths,
        hero1_assists,
    ) in result:
        matchups[hero0].append(
            HeroMatchUpWinLossStatMatchUp(
                hero_id=hero1,
                wins=hero0_wins,
                losses=hero1_wins,
                matches=hero0_wins + hero1_wins,
                total_kills=hero0_kills,
                total_kills_received=hero1_deaths,
                total_deaths=hero0_deaths,
                total_deaths_received=hero1_deaths,
                total_assists=hero0_assists,
                total_assists_received=hero1_assists,
            )
        )
        matchups[hero1].append(
            HeroMatchUpWinLossStatMatchUp(
                hero_id=hero0,
                wins=hero1_wins,
                losses=hero0_wins,
                matches=hero0_wins + hero1_wins,
                total_kills=hero1_kills,
                total_kills_received=hero0_deaths,
                total_deaths=hero1_deaths,
                total_deaths_received=hero0_deaths,
                total_assists=hero1_assists,
                total_assists_received=hero0_assists,
            )
        )
    return [HeroMatchUpWinLossStat(hero_id=h, matchups=matchups[h]) for h in matchups]


@router.get("/item-win-loss-stats", summary="RateLimit: 100req/s")
def get_item_win_loss_stats(
    req: Request,
    res: Response,
    item_id: int | None = None,
    min_badge_level: Annotated[int | None, Query(ge=0)] = None,
    max_badge_level: Annotated[int | None, Query(le=116)] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
    region: (Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None) = None,
) -> list[ItemWinLossStat]:
    limiter.apply_limits(
        req,
        res,
        "/v2/item-win-loss-stats",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=3600"
    return get_item_win_loss_stats_cached(
        item_id,
        min_badge_level,
        max_badge_level,
        min_unix_timestamp,
        max_unix_timestamp,
        match_mode,
        region,
    )


@router.get("/hero/{hero_id}/item-win-loss-stats", summary="RateLimit: 100req/s")
def get_hero_item_win_loss_stats(
    req: Request,
    res: Response,
    hero_id: int,
    item_id: int | None = None,
    min_badge_level: Annotated[int | None, Query(ge=0)] = None,
    max_badge_level: Annotated[int | None, Query(le=116)] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
    region: (Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None) = None,
) -> list[HeroItemWinLossStat]:
    limiter.apply_limits(
        req,
        res,
        "/v2/hero/{hero_id}/item-win-loss-stats",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=3600"
    return get_hero_item_win_loss_stats_cached(
        hero_id,
        item_id,
        min_badge_level,
        max_badge_level,
        min_unix_timestamp,
        max_unix_timestamp,
        match_mode,
        region,
    )


@ttl_cache(ttl=3600)
def get_hero_item_win_loss_stats_cached(
    hero_id: int,
    item_id: int | None = None,
    min_badge_level: Annotated[int | None, Query(ge=0)] = None,
    max_badge_level: Annotated[int | None, Query(le=116)] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
    region: (Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None) = None,
) -> list[HeroItemWinLossStat]:
    query = """
    SELECT
        hero_id,
        item_id,
        countIf(won) AS wins,
        countIf(NOT won) AS losses
    FROM match_player_item_v2
    INNER JOIN match_info mi USING (match_id)
    INNER JOIN player p USING (account_id)
    WHERE TRUE
    AND mi.match_outcome = 'TeamWin'
    AND mi.match_mode IN ('Ranked', 'Unranked')
    AND %(hero_id)s = hero_id
    AND (%(item_id)s IS NULL OR item_id = %(item_id)s)
    AND (%(min_badge_level)s IS NULL OR (average_match_badge IS NOT NULL AND average_match_badge >= %(min_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 >= %(min_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 >= %(min_badge_level)s))
    AND (%(max_badge_level)s IS NULL OR (average_match_badge IS NOT NULL AND average_match_badge <= %(max_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 <= %(max_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 <= %(max_badge_level)s))
    AND (%(min_unix_timestamp)s IS NULL OR mi.start_time >= toDateTime(%(min_unix_timestamp)s))
    AND (%(max_unix_timestamp)s IS NULL OR mi.start_time <= toDateTime(%(max_unix_timestamp)s))
    AND (%(match_mode)s IS NULL OR mi.match_mode = %(match_mode)s)
    AND (%(region)s IS NULL OR p.region_mode = %(region)s)
    GROUP BY hero_id, item_id
    ORDER BY wins + losses DESC;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query,
            {
                "hero_id": hero_id,
                "item_id": item_id,
                "min_badge_level": min_badge_level,
                "max_badge_level": max_badge_level,
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "match_mode": match_mode,
                "region": region,
            },
        )
    return [
        HeroItemWinLossStat(hero_id=r[0], item_id=r[1], wins=r[2], losses=r[3], matches=r[2] + r[3])
        for r in result
    ]


@ttl_cache(ttl=3600)
def get_item_win_loss_stats_cached(
    item_id: int | None = None,
    min_badge_level: Annotated[int | None, Query(ge=0)] = None,
    max_badge_level: Annotated[int | None, Query(le=116)] = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
    region: (Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None) = None,
) -> list[ItemWinLossStat]:
    query = """
    SELECT
        item_id,
        countIf(won) AS wins,
        countIf(NOT won) AS losses
    FROM match_player_item_v2
    INNER JOIN match_info mi USING (match_id)
    INNER JOIN player p USING (account_id)
    WHERE TRUE
    AND mi.match_outcome = 'TeamWin'
    AND mi.match_mode IN ('Ranked', 'Unranked')
    AND (%(item_id)s IS NULL OR item_id = %(item_id)s)
    AND (%(min_badge_level)s IS NULL OR (average_match_badge IS NOT NULL AND average_match_badge >= %(min_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 >= %(min_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 >= %(min_badge_level)s))
    AND (%(max_badge_level)s IS NULL OR (average_match_badge IS NOT NULL AND average_match_badge <= %(max_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 <= %(max_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 <= %(max_badge_level)s))
    AND (%(min_unix_timestamp)s IS NULL OR mi.start_time >= toDateTime(%(min_unix_timestamp)s))
    AND (%(max_unix_timestamp)s IS NULL OR mi.start_time <= toDateTime(%(max_unix_timestamp)s))
    AND (%(match_mode)s IS NULL OR mi.match_mode = %(match_mode)s)
    AND (%(region)s IS NULL OR p.region_mode = %(region)s)
    GROUP BY item_id
    ORDER BY wins + losses DESC;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(
            query,
            {
                "item_id": item_id,
                "min_badge_level": min_badge_level,
                "max_badge_level": max_badge_level,
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "match_mode": match_mode,
                "region": region,
            },
        )
    return [
        ItemWinLossStat(item_id=r[0], wins=r[1], losses=r[2], matches=r[1] + r[2]) for r in result
    ]


@router.get(
    "/players/hero-stats",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/players/{account_id}/hero-stats
    """,
    deprecated=True,
)
def get_player_hero_stats_batch(
    req: Request,
    res: Response,
    account_ids: Annotated[
        str,
        Query(description="Comma separated account ids of the players, at most 100 allowed"),
    ],
    hero_id: int | None = None,
    min_match_id: Annotated[int | None, Query(ge=0)] = None,
    max_match_id: int | None = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
) -> dict[int, list[PlayerHeroStat]]:
    account_ids = [utils.validate_steam_id(int(a)) for a in account_ids.split(",")]
    if len(account_ids) > 100:
        raise HTTPException(status_code=400, detail="Max 100 account_ids allowed")
    limiter.apply_limits(
        req,
        res,
        "/v2/players/hero-stats",
        [RateLimit(limit=100, period=1)],
        count=len(account_ids),
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    query = """
        SELECT
            account_id,
            hero_id,
            count(*)                                                                                   AS matches,
            max(ranked_badge_level)                                                                    AS highest_ranked_badge_level,
            countIf(won)                                                                               AS wins,
            avg(arrayMax(stats.level))                                                                 AS ending_level,
            sum(kills)                                                                                 AS kills,
            sum(deaths)                                                                                AS deaths,
            sum(assists)                                                                               AS assists,
            avg(denies)                                                                                AS denies_per_match,
            60 * avg(mp.kills / duration_s)                                                            AS kills_per_min,
            60 * avg(mp.deaths / duration_s)                                                           AS deaths_per_min,
            60 * avg(mp.assists / duration_s)                                                          AS assists_per_min,
            60 * avg(denies / duration_s)                                                              AS denies_per_min,
            60 * avg(net_worth / duration_s)                                                           AS networth_per_min,
            60 * avg(last_hits / duration_s)                                                           AS last_hits_per_min,
            60 * avg(arrayMax(stats.player_damage) / duration_s)                                       AS damage_mitigated_per_min,
            60 * avg(arrayMax(stats.player_damage_taken) / duration_s)                                 AS damage_taken_per_min,
            60 * avg(arrayMax(stats.creep_kills) / duration_s)                                         AS creeps_per_min,
            60 * avg(arrayMax(stats.neutral_damage) / duration_s)                                      AS obj_damage_per_min,
            avg(arrayMax(stats.shots_hit) / greatest(1, arrayMax(stats.shots_hit) + arrayMax(stats.shots_missed))) AS accuracy,
            avg(arrayMax(stats.hero_bullets_hit_crit) / greatest(1, arrayMax(stats.hero_bullets_hit_crit) + arrayMax(stats.hero_bullets_hit))) AS crit_shot_rate
        FROM match_player mp
        INNER JOIN default.match_info AS mi USING (match_id)
        WHERE account_id IN %(account_ids)s
        AND mi.match_outcome = 'TeamWin'
        AND mi.match_mode IN ('Ranked', 'Unranked')
        AND (%(hero_id)s IS NULL OR hero_id = %(hero_id)s)
        AND (%(min_unix_timestamp)s IS NULL OR mi.start_time >= toDateTime(%(min_unix_timestamp)s))
        AND (%(max_unix_timestamp)s IS NULL OR mi.start_time <= toDateTime(%(max_unix_timestamp)s))
        AND (%(min_match_id)s IS NULL OR mi.match_id >= %(min_match_id)s)
        AND (%(max_match_id)s IS NULL OR mi.match_id <= %(max_match_id)s)
        AND (%(match_mode)s IS NULL OR mi.match_mode = %(match_mode)s)
        GROUP BY account_id, hero_id
        ORDER BY account_id, hero_id;
    """
    with CH_POOL.get_client() as client:
        result, keys = client.execute(
            query,
            {
                "account_ids": account_ids,
                "hero_id": hero_id,
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "min_match_id": min_match_id,
                "max_match_id": max_match_id,
                "match_mode": match_mode,
            },
            with_column_types=True,
        )
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    return {
        k: list(v)
        for k, v in itertools.groupby(
            (PlayerHeroStat(**{k: v for (k, _), v in zip(keys, r)}) for r in result),
            key=lambda x: x.account_id,
        )
    }


@router.get(
    "/players/{account_id}/hero-stats",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/players/{account_id}/hero-stats
    """,
    deprecated=True,
)
def get_player_hero_stats(
    req: Request,
    res: Response,
    account_id: Annotated[int, Path(description="The account id of the player, it's a SteamID3")],
    hero_id: int | None = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
) -> list[PlayerHeroStat]:
    account_id = utils.validate_steam_id(account_id)
    return get_player_hero_stats_batch(
        req,
        res,
        account_ids=str(account_id),
        hero_id=hero_id,
        min_unix_timestamp=min_unix_timestamp,
        max_unix_timestamp=max_unix_timestamp,
        match_mode=match_mode,
    )[account_id]


@router.get(
    "/players/item-stats",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/players/{account_id}/item-stats
    """,
    deprecated=True,
)
def get_player_item_stats_batch(
    req: Request,
    res: Response,
    account_ids: Annotated[
        str,
        Query(description="Comma separated account ids of the players, at most 100 allowed"),
    ],
    hero_id: int | None = None,
    item_id: int | None = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
) -> dict[int, list[PlayerItemStat]]:
    account_ids = [utils.validate_steam_id(int(a)) for a in account_ids.split(",")]
    if len(account_ids) > 100:
        raise HTTPException(status_code=400, detail="Max 100 account_ids allowed")
    limiter.apply_limits(
        req,
        res,
        "/v2/players/{account_id}/item-stats",
        [RateLimit(limit=100, period=1)],
        count=len(account_ids),
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    query = """
        SELECT
            account_id,
            hero_id,
            item_id,
            count(*) AS matches,
            countIf(won) AS wins
        FROM match_player_item_v2
        INNER JOIN default.match_info AS mi USING (match_id)
        WHERE account_id IN %(account_ids)s
        AND mi.match_outcome = 'TeamWin'
        AND mi.match_mode IN ('Ranked', 'Unranked')
        AND (%(hero_id)s IS NULL OR hero_id = %(hero_id)s)
        AND (%(item_id)s IS NULL OR item_id = %(item_id)s)
        AND (%(min_unix_timestamp)s IS NULL OR mi.start_time >= toDateTime(%(min_unix_timestamp)s))
        AND (%(max_unix_timestamp)s IS NULL OR mi.start_time <= toDateTime(%(max_unix_timestamp)s))
        AND (%(match_mode)s IS NULL OR mi.match_mode = %(match_mode)s)
        GROUP BY account_id, hero_id, item_id
        ORDER BY account_id, hero_id, item_id;
    """
    with CH_POOL.get_client() as client:
        result, keys = client.execute(
            query,
            {
                "account_ids": account_ids,
                "hero_id": hero_id,
                "item_id": item_id,
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "match_mode": match_mode,
            },
            with_column_types=True,
        )
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    return {
        k: list(v)
        for k, v in itertools.groupby(
            (PlayerItemStat(**{k: v for (k, _), v in zip(keys, r)}) for r in result),
            key=lambda x: x.account_id,
        )
    }


@router.get(
    "/players/{account_id}/item-stats",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/players/{account_id}/item-stats
    """,
    deprecated=True,
)
def get_player_item_stats(
    req: Request,
    res: Response,
    account_id: Annotated[int, Path(description="The account id of the player, it's a SteamID3")],
    hero_id: int | None = None,
    item_id: int | None = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
) -> list[PlayerItemStat]:
    return get_player_item_stats_batch(
        req,
        res,
        account_ids=str(utils.validate_steam_id(account_id)),
        hero_id=hero_id,
        item_id=item_id,
        min_unix_timestamp=min_unix_timestamp,
        max_unix_timestamp=max_unix_timestamp,
        match_mode=match_mode,
    )[account_id]


@router.get(
    "/players/{account_id}/mates",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/players/{account_id}/mate-stats
    """,
    include_in_schema=False,
    deprecated=True,
)
def get_player_mates(
    account_id: Annotated[int, Path(description="The account id of the player, it's a SteamID3")],
):
    return RedirectResponse(
        url=f"https://api.deadlock-api.com/v1/players/{account_id}/mate-stats",
        status_code=HTTP_301_MOVED_PERMANENTLY,
    )


@router.get(
    "/players/{account_id}/mate-stats",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/players/{account_id}/mate-stats
    """,
    deprecated=True,
)
def get_player_mate_stats(
    req: Request,
    account_id: Annotated[int, Path(description="The account id of the player, it's a SteamID3")],
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
) -> RedirectResponse:
    url = URL(f"https://api.deadlock-api.com/v1/players/{account_id}/mate-stats")
    url = url.include_query_params(**{k: v for k, v in req.query_params.items() if v is not None})
    return RedirectResponse(url, HTTP_301_MOVED_PERMANENTLY)


@router.get(
    "/players/{account_id}/parties",
    summary="RateLimit: 100req/s",
    include_in_schema=False,
    deprecated=True,
)
def get_player_parties(
    account_id: Annotated[int, Path(description="The account id of the player, it's a SteamID3")],
):
    return RedirectResponse(
        url=f"https://api.deadlock-api.com/v1/players/{account_id}/party-stats",
        status_code=HTTP_301_MOVED_PERMANENTLY,
    )


@router.get(
    "/players/{account_id}/party-stats",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/players/{account_id}/party-stats
    """,
    deprecated=True,
)
def get_player_party_stats(
    req: Request,
    account_id: Annotated[int, Path(description="The account id of the player, it's a SteamID3")],
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
) -> RedirectResponse:
    url = URL(f"https://api.deadlock-api.com/v1/players/{account_id}/party-stats")
    url = url.include_query_params(**{k: v for k, v in req.query_params.items() if v is not None})
    return RedirectResponse(url, HTTP_301_MOVED_PERMANENTLY)


@router.get(
    "/players/card-history",
    summary="RateLimit: 100req/s",
)
def get_player_card_history_batch(
    req: Request,
    res: Response,
    account_ids: Annotated[
        str,
        Query(description="Comma separated account ids of the players, at most 100 allowed"),
    ],
) -> list[list[PlayerCardHistoryEntry]]:
    account_ids = [utils.validate_steam_id(int(a)) for a in account_ids.split(",")]
    if len(account_ids) > 100:
        raise HTTPException(status_code=400, detail="Max 100 account_ids allowed")

    limiter.apply_limits(
        req,
        res,
        "/v2/players/card-history",
        [RateLimit(limit=100, period=1)],
        count=len(account_ids),
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    query = """
    SELECT *
    FROM player_card
    WHERE account_id IN %(account_ids)s
    ORDER BY account_id, created_at DESC;
    """
    with CH_POOL.get_client() as client:
        result, keys = client.execute(query, {"account_ids": account_ids}, with_column_types=True)
    result = [{k: v for (k, _), v in zip(keys, r)} for r in result]
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    cards = [PlayerCardHistoryEntry.from_row(r) for r in result]
    cards_grouped = itertools.groupby(cards, key=lambda x: x.account_id)
    return [list(cards) for account_id, cards in cards_grouped]


@router.get(
    "/players/{account_id}/card-history",
    summary="RateLimit: 100req/s",
)
def get_player_card_history(
    req: Request,
    res: Response,
    account_id: Annotated[int, Path(description="The account id of the player, it's a SteamID3")],
) -> list[PlayerCardHistoryEntry]:
    return get_player_card_history_batch(req, res, account_ids=str(account_id))[0]


@router.get(
    "/players/mmr-history",
    summary="RateLimit: 100req/s",
)
def get_player_mmr_history_batch(
    req: Request,
    res: Response,
    account_ids: Annotated[
        str,
        Query(description="Comma separated account ids of the players, at most 100 allowed"),
    ],
) -> list[list[PlayerMMRHistoryEntryV2]]:
    account_ids = [utils.validate_steam_id(int(a)) for a in account_ids.split(",")]
    if len(account_ids) > 100:
        raise HTTPException(status_code=400, detail="Max 100 account_ids allowed")

    limiter.apply_limits(
        req,
        res,
        "/v2/players/mmr-history",
        [RateLimit(limit=100, period=1)],
        count=len(account_ids),
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    query = """
    SELECT match_id, ranked_badge_level, won, 'metadata' as source
    FROM match_player
    WHERE account_id IN %(account_ids)s
    ORDER BY match_id DESC;
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"account_ids": account_ids})
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    return [
        [
            PlayerMMRHistoryEntryV2(
                account_id=account_id,
                match_id=r[0],
                match_ranked_badge_level=r[1],
                won=r[2],
                source=r[3],
            )
            for r in result
        ]
        for account_id in account_ids
    ]


@router.get(
    "/players/{account_id}/mmr-history",
    summary="RateLimit: 100req/s",
)
def get_player_mmr_history(
    req: Request,
    res: Response,
    account_id: Annotated[int, Path(description="The account id of the player, it's a SteamID3")],
) -> list[PlayerMMRHistoryEntryV2]:
    return get_player_mmr_history_batch(req, res, account_ids=str(account_id))[0]


@router.get(
    "/players/{account_id}/match-history",
    summary="Moved to new API: https://api.deadlock-api.com/",
    description="""
# Endpoint moved to new API
- New API Docs: https://api.deadlock-api.com/docs
- New API Endpoint: https://api.deadlock-api.com/v1/players/{account_id}/match-history
    """,
    deprecated=True,
)
def get_matches_by_account_id(
    req: Request,
    res: Response,
    account_id: int,
    has_metadata: bool | None = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    min_match_id: Annotated[int | None, Query(ge=0)] = None,
    max_match_id: int | None = None,
    min_duration_s: Annotated[int | None, Query(ge=0)] = None,
    max_duration_s: Annotated[int | None, Query(le=7000)] = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
    without_avg_badge: bool | None = None,
) -> list[dict]:
    limiter.apply_limits(
        req,
        res,
        "/v2/players/{account_id}/match-history",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    account_id = utils.validate_steam_id(account_id)
    if without_avg_badge:
        select = "pmh.* EXCEPT (created_at)"
    else:
        select = """
        pmh.* EXCEPT (created_at),
        toBool(mi.match_id > 0) AS has_metadata,
        mi.match_id IS NOT null ? intDivOrZero(mi.average_badge_team0 + mi.average_badge_team1, 2): null AS average_match_badge,
        round(avg(average_match_badge) OVER (ORDER BY match_id DESC ROWS BETWEEN CURRENT ROW AND 10 FOLLOWING), 2) AS moving_average_match_badge,
        round(median(average_match_badge) OVER (ORDER BY match_id DESC ROWS BETWEEN CURRENT ROW AND 10 FOLLOWING), 2) AS moving_median_match_badge
        """
    query = f"""
    SELECT {select}
    FROM player_match_history pmh FINAL
    {"" if without_avg_badge else "LEFT JOIN match_info mi FINAL USING (match_id)"}
    WHERE account_id = %(account_id)s
    AND pmh.match_mode IN ('Ranked', 'Unranked')
    {"" if without_avg_badge else "AND (%(has_metadata)s IS NULL OR toBool(mi.match_id > 0) = %(has_metadata)s)"}
    AND (%(min_unix_timestamp)s IS NULL OR pmh.start_time >= toDateTime(%(min_unix_timestamp)s))
    AND (%(max_unix_timestamp)s IS NULL OR pmh.start_time <= toDateTime(%(max_unix_timestamp)s))
    AND (%(min_match_id)s IS NULL OR pmh.match_id >= %(min_match_id)s)
    AND (%(max_match_id)s IS NULL OR pmh.match_id <= %(max_match_id)s)
    AND (%(min_duration_s)s IS NULL OR pmh.match_duration_s >= %(min_duration_s)s)
    AND (%(max_duration_s)s IS NULL OR pmh.match_duration_s <= %(max_duration_s)s)
    AND (%(match_mode)s IS NULL OR pmh.match_mode = %(match_mode)s)
    ORDER BY match_id DESC
    """
    with CH_POOL.get_client() as client:
        entries, keys = client.execute(
            query,
            {
                "account_id": account_id,
                "has_metadata": has_metadata,
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "min_match_id": min_match_id,
                "max_match_id": max_match_id,
                "min_duration_s": min_duration_s,
                "max_duration_s": max_duration_s,
                "match_mode": match_mode,
            },
            with_column_types=True,
        )
    if len(entries) == 0:
        raise HTTPException(status_code=404, detail="Not found")

    def transform(v):
        return v if not isinstance(v, datetime) else v.astimezone(timezone.utc)

    return [{k: transform(v) for (k, _), v in zip(keys, r)} for r in entries]
