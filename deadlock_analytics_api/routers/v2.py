import itertools
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Path, Query
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.status import HTTP_301_MOVED_PERMANENTLY

from deadlock_analytics_api import utils
from deadlock_analytics_api.globs import CH_POOL
from deadlock_analytics_api.rate_limiter import limiter
from deadlock_analytics_api.rate_limiter.models import RateLimit
from deadlock_analytics_api.routers.v1 import HeroWinLossStat
from deadlock_analytics_api.routers.v2_models import (
    ItemWinLossStat,
    PlayerCardHistoryEntry,
    PlayerHeroStat,
    PlayerItemStat,
    PlayerLeaderboardV2,
    PlayerMate,
    PlayerMMRHistoryEntryV2,
    PlayerParty,
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
    sort_by: Literal["winrate", "wins", "matches"] | None = None,
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
    sort_by: Literal["winrate", "wins", "matches"] | None = None,
) -> list[PlayerLeaderboardV2]:
    limiter.apply_limits(
        req, res, "/v2/leaderboard/{region}", [RateLimit(limit=100, period=1)]
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    order_by_clause = {
        "winrate": "ORDER BY rank, wins / greatest(1, matches_played) DESC, account_id",
        "wins": "ORDER BY rank, wins DESC, account_id",
        "matches": "ORDER BY rank, matches_played DESC, account_id",
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
        result = client.execute(
            query, {"start": start - 1, "limit": limit, "region": region}
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


@router.get("/hero-win-loss-stats", summary="RateLimit: 100req/s")
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
    region: (
        Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None
    ) = None,
) -> list[HeroWinLossStat]:
    limiter.apply_limits(
        req, res, "/v2/hero-win-loss-stats", [RateLimit(limit=100, period=1)]
    )
    res.headers["Cache-Control"] = "public, max-age=1200"
    query = """
    WITH filtered_players AS (
        SELECT hero_id, account_id, countIf(team == mi.winning_team) AS player_wins, countIf(team != mi.winning_team) AS player_losses
        FROM match_player
        INNER JOIN match_info mi USING (match_id)
        INNER JOIN player p USING (account_id)
        WHERE 1=1
        AND (%(min_badge_level)s IS NULL OR (ranked_badge_level IS NOT NULL AND ranked_badge_level >= %(min_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 >= %(min_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 >= %(min_badge_level)s))
        AND (%(max_badge_level)s IS NULL OR (ranked_badge_level IS NOT NULL AND ranked_badge_level <= %(max_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 <= %(max_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 <= %(max_badge_level)s))
        AND (%(min_unix_timestamp)s IS NULL OR mi.start_time >= toDateTime(%(min_unix_timestamp)s))
        AND (%(max_unix_timestamp)s IS NULL OR mi.start_time <= toDateTime(%(max_unix_timestamp)s))
        AND (%(match_mode)s IS NULL OR mi.match_mode = %(match_mode)s)
        AND (%(region)s IS NULL OR p.region_mode = %(region)s)
        GROUP BY hero_id, account_id
    )
    SELECT hero_id, sum(player_wins) AS wins, sum(player_losses) AS losses
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
    return [HeroWinLossStat(hero_id=r[0], wins=r[1], losses=r[2]) for r in result]


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
    region: (
        Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None
    ) = None,
) -> list[ItemWinLossStat]:
    limiter.apply_limits(
        req,
        res,
        "/v2/hero/{hero_id}/item-win-loss-stats",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=1200"
    query = """
    SELECT
        hero_id,
        items.item_id AS item_id,
        countIf(won) AS wins,
        countIf(NOT won) AS losses
    FROM match_player
    INNER JOIN match_info mi USING (match_id)
    INNER JOIN player p USING (account_id)
    ARRAY JOIN items
    WHERE TRUE
    AND %(hero_id)s = hero_id
    AND (%(item_id)s IS NULL OR items.item_id = %(item_id)s)
    AND (%(min_badge_level)s IS NULL OR (ranked_badge_level IS NOT NULL AND ranked_badge_level >= %(min_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 >= %(min_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 >= %(min_badge_level)s))
    AND (%(max_badge_level)s IS NULL OR (ranked_badge_level IS NOT NULL AND ranked_badge_level <= %(max_badge_level)s) OR (mi.average_badge_team0 IS NOT NULL AND mi.average_badge_team0 <= %(max_badge_level)s) OR (mi.average_badge_team1 IS NOT NULL AND mi.average_badge_team1 <= %(max_badge_level)s))
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
        ItemWinLossStat(hero_id=r[0], item_id=r[1], wins=r[2], losses=r[3])
        for r in result
    ]


@router.get(
    "/players/hero-stats",
    summary="RateLimit: 100req/s",
)
def get_player_hero_stats_batch(
    req: Request,
    res: Response,
    account_ids: Annotated[
        str,
        Query(
            description="Comma separated account ids of the players, at most 100 allowed"
        ),
    ],
    hero_id: int | None = None,
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
            sum(team = mi.winning_team)                                                                AS wins,
            sum(kills)                                                                                 AS kills,
            sum(deaths)                                                                                AS deaths,
            sum(assists)                                                                               AS assists,
            avg(arrayMax(stats.level))                                                                 AS ending_level,
            avg(denies)                                                                                AS denies_per_match,
            60 * avg(net_worth / duration_s)                                                           AS networth_per_min,
            60 * avg(last_hits / duration_s)                                                           AS last_hits_per_min,
            60 * avg(denies / duration_s)                                                              AS denies_per_min,
            60 * avg(arrayMax(stats.player_damage) / duration_s)                                       AS damage_mitigated_per_min,
            60 * avg(arrayMax(stats.player_damage_taken) / duration_s)                                 AS damage_taken_per_min,
            60 * avg(arrayMax(stats.creep_kills) / duration_s)                                         AS creeps_per_min,
            60 * avg(arrayMax(stats.neutral_damage) / duration_s)                                      AS obj_damage_per_min,
            avg(arrayMax(stats.shots_hit) / greatest(1, arrayMax(stats.shots_hit) + arrayMax(stats.shots_missed))) AS accuracy,
            avg(arrayMax(stats.hero_bullets_hit_crit) / greatest(1, arrayMax(stats.hero_bullets_hit_crit) + arrayMax(stats.hero_bullets_hit))) AS crit_shot_rate
        FROM default.match_player
        INNER JOIN default.match_info AS mi USING (match_id)
        WHERE account_id IN %(account_ids)s
        AND (%(hero_id)s IS NULL OR hero_id = %(hero_id)s)
        AND (%(min_unix_timestamp)s IS NULL OR mi.start_time >= toDateTime(%(min_unix_timestamp)s))
        AND (%(max_unix_timestamp)s IS NULL OR mi.start_time <= toDateTime(%(max_unix_timestamp)s))
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
    summary="RateLimit: 100req/s",
)
def get_player_hero_stats(
    req: Request,
    res: Response,
    account_id: Annotated[
        int, Path(description="The account id of the player, it's a SteamID3")
    ],
    hero_id: int | None = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
) -> list[PlayerHeroStat]:
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
    summary="RateLimit: 100req/s",
)
def get_player_item_stats_batch(
    req: Request,
    res: Response,
    account_ids: Annotated[
        str,
        Query(
            description="Comma separated account ids of the players, at most 100 allowed"
        ),
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
            items.item_id as item_id,
            count(*)                                                                                   AS matches,
            sum(team = mi.winning_team)                                                                AS wins
        FROM default.match_player
        INNER JOIN default.match_info AS mi USING (match_id)
        ARRAY JOIN items
        WHERE account_id IN %(account_ids)s
        AND (%(hero_id)s IS NULL OR hero_id = %(hero_id)s)
        AND (%(item_id)s IS NULL OR items.item_id = %(item_id)s)
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
    summary="RateLimit: 100req/s",
)
def get_player_item_stats(
    req: Request,
    res: Response,
    account_id: Annotated[
        int, Path(description="The account id of the player, it's a SteamID3")
    ],
    hero_id: int | None = None,
    item_id: int | None = None,
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
) -> list[PlayerItemStat]:
    return get_player_item_stats_batch(
        req,
        res,
        account_ids=str(account_id),
        hero_id=hero_id,
        item_id=item_id,
        min_unix_timestamp=min_unix_timestamp,
        max_unix_timestamp=max_unix_timestamp,
        match_mode=match_mode,
    )[account_id]


@router.get(
    "/players/{account_id}/mates",
    summary="RateLimit: 100req/s",
    include_in_schema=False,
    deprecated=True,
)
def get_player_mates(
    account_id: Annotated[
        int, Path(description="The account id of the player, it's a SteamID3")
    ],
):
    return RedirectResponse(
        url=f"/v2/players/{account_id}/mate-stats",
        status_code=HTTP_301_MOVED_PERMANENTLY,
    )


@router.get(
    "/players/{account_id}/mate-stats",
    summary="RateLimit: 100req/s",
)
def get_player_mate_stats(
    req: Request,
    res: Response,
    account_id: Annotated[
        int, Path(description="The account id of the player, it's a SteamID3")
    ],
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
) -> list[PlayerMate]:
    limiter.apply_limits(
        req,
        res,
        "/v2/players/{account_id}/mate-stats",
        [RateLimit(limit=100, period=1)],
    )
    account_id = utils.validate_steam_id(account_id)
    query = """
    SELECT mate_id, countIf(p.team == mi.winning_team) as wins, COUNT() as matches_played, groupArray(p.match_id) as matches
    FROM match_parties p
    ARRAY JOIN p.account_ids as mate_id
    INNER JOIN match_info mi USING (match_id)
    WHERE has(p.account_ids, %(account_id)s)
    AND (%(min_unix_timestamp)s IS NULL OR mi.start_time >= toDateTime(%(min_unix_timestamp)s))
    AND (%(max_unix_timestamp)s IS NULL OR mi.start_time <= toDateTime(%(max_unix_timestamp)s))
    AND (%(match_mode)s IS NULL OR mi.match_mode = %(match_mode)s)
    GROUP BY mate_id
    ORDER BY matches_played DESC;
    """
    with CH_POOL.get_client() as client:
        result, keys = client.execute(
            query,
            {
                "account_id": account_id,
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "match_mode": match_mode,
            },
            with_column_types=True,
        )
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    return [PlayerMate(**{k: v for (k, _), v in zip(keys, r)}) for r in result]


@router.get(
    "/players/{account_id}/parties",
    summary="RateLimit: 100req/s",
    include_in_schema=False,
    deprecated=True,
)
def get_player_parties(
    account_id: Annotated[
        int, Path(description="The account id of the player, it's a SteamID3")
    ],
):
    return RedirectResponse(
        url=f"/v2/players/{account_id}/party-stats",
        status_code=HTTP_301_MOVED_PERMANENTLY,
    )


@router.get(
    "/players/{account_id}/party-stats",
    summary="RateLimit: 100req/s",
)
def get_player_party_stats(
    req: Request,
    res: Response,
    account_id: Annotated[
        int, Path(description="The account id of the player, it's a SteamID3")
    ],
    min_unix_timestamp: Annotated[int | None, Query(ge=0)] = None,
    max_unix_timestamp: int | None = None,
    match_mode: Literal["Ranked", "Unranked"] | None = None,
) -> list[PlayerParty]:
    limiter.apply_limits(
        req,
        res,
        "/v2/players/{account_id}/party-stats",
        [RateLimit(limit=100, period=1)],
    )
    account_id = utils.validate_steam_id(account_id)
    query = """
    SELECT
    length(p.account_ids)              as party_size,
    countIf(p.team == mi.winning_team) as wins,
    COUNT()                            as matches_played,
    groupArray(p.match_id)             as matches
    FROM match_parties p
    INNER JOIN match_info mi USING (match_id)
    WHERE has(p.account_ids, %(account_id)s)
    AND (%(min_unix_timestamp)s IS NULL OR mi.start_time >= toDateTime(%(min_unix_timestamp)s))
    AND (%(max_unix_timestamp)s IS NULL OR mi.start_time <= toDateTime(%(max_unix_timestamp)s))
    AND (%(match_mode)s IS NULL OR mi.match_mode = %(match_mode)s)
    GROUP BY party_size
    ORDER BY matches_played DESC;
    """
    with CH_POOL.get_client() as client:
        result, keys = client.execute(
            query,
            {
                "account_id": account_id,
                "min_unix_timestamp": min_unix_timestamp,
                "max_unix_timestamp": max_unix_timestamp,
                "match_mode": match_mode,
            },
            with_column_types=True,
        )
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    return [PlayerParty(**{k: v for (k, _), v in zip(keys, r)}) for r in result]


@router.get(
    "/players/card-history",
    summary="RateLimit: 100req/s",
)
def get_player_card_history_batch(
    req: Request,
    res: Response,
    account_ids: Annotated[
        str,
        Query(
            description="Comma separated account ids of the players, at most 100 allowed"
        ),
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
        result, keys = client.execute(
            query, {"account_ids": account_ids}, with_column_types=True
        )
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
    account_id: Annotated[
        int, Path(description="The account id of the player, it's a SteamID3")
    ],
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
        Query(
            description="Comma separated account ids of the players, at most 100 allowed"
        ),
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
    account_id: Annotated[
        int, Path(description="The account id of the player, it's a SteamID3")
    ],
) -> list[PlayerMMRHistoryEntryV2]:
    return get_player_mmr_history_batch(req, res, account_ids=str(account_id))[0]


@router.get("/players/{account_id}/match-history", summary="RateLimit: 100req/s")
def get_matches_by_account_id(
    req: Request, res: Response, account_id: int
) -> list[dict]:
    limiter.apply_limits(
        req,
        res,
        "/v2/players/{account_id}/match-history",
        [RateLimit(limit=100, period=1)],
    )
    res.headers["Cache-Control"] = "public, max-age=300"
    account_id = utils.validate_steam_id(account_id)
    query = """
    WITH matches as (
        SELECT match_id, mi.start_time, ranked_badge_level, 'metadata' as source
        FROM match_player
        INNER JOIN match_info mi USING (match_id)
        WHERE account_id = %(account_id)s

        UNION ALL

        SELECT match_id, start_time, ranked_badge_level, 'short' as source
        FROM finished_matches
        ARRAY JOIN players
        WHERE players.account_id = %(account_id)s
        AND match_id NOT IN (SELECT match_id FROM match_info)
    )
    SELECT *
    FROM matches
    ORDER BY match_id DESC
    """
    with CH_POOL.get_client() as client:
        result = client.execute(query, {"account_id": account_id})
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return [
        {
            "match_id": r[0],
            "start_time": r[1].isoformat(),
            "ranked_badge_level": r[2],
            "source": r[3],
        }
        for r in result
    ]
