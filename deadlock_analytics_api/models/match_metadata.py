import datetime

from pydantic import BaseModel, ConfigDict, Field


class MatchMetadataObjectives(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    destroyed_time_s: int
    creep_damage: int
    creep_damage_mitigated: int
    player_damage: int
    player_damage_mitigated: int
    first_damage_time_s: int
    team_objective: str
    team: str

    @classmethod
    def from_dict(cls, row: dict[str, any]):
        return cls(**row)


class MatchMetadataMidBoss(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    team_killed: str
    team_claimed: str
    destroyed_time_s: int

    @classmethod
    def from_dict(cls, row: dict[str, any]):
        return cls(**row)


class MatchMetadataPlayerBookReward(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    book_id: int
    xp_amount: int
    starting_xp: int

    @classmethod
    def from_dict(cls, row: dict[str, any]):
        return cls(**row)


class MatchMetadataPlayerDeathDetails(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    game_time_s: int
    killer_player_slot: int
    death_pos: list[float]
    killer_pos: list[float]
    death_duration_s: int

    @classmethod
    def from_dict(cls, row: dict[str, any]):
        return cls(**row)


class MatchMetadataPlayerItems(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    game_time_s: int
    item_id: int
    upgrade_id: int
    sold_time_s: int
    flags: int
    imbued_ability_id: int

    @classmethod
    def from_dict(cls, row: dict[str, any]):
        return cls(**row)


class MatchMetadataPlayerStats(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    time_stamp_s: int
    net_worth: int
    gold_player: int
    gold_player_orbs: int
    gold_lane_creep_orbs: int
    gold_neutral_creep_orbs: int
    gold_boss: int
    gold_boss_orb: int
    gold_treasure: int
    gold_denied: int
    gold_death_loss: int
    gold_lane_creep: int
    gold_neutral_creep: int
    kills: int
    deaths: int
    assists: int
    creep_kills: int
    neutral_kills: int
    possible_creeps: int
    creep_damage: int
    player_damage: int
    neutral_damage: int
    boss_damage: int
    denies: int
    player_healing: int
    ability_points: int
    self_healing: int
    player_damage_taken: int
    max_health: int
    weapon_power: int
    tech_power: int
    shots_hit: int
    shots_missed: int
    damage_absorbed: int
    absorption_provided: int
    hero_bullets_hit: int
    hero_bullets_hit_crit: int
    heal_prevented: int
    heal_lost: int
    damage_mitigated: int
    level: int

    @classmethod
    def from_dict(cls, row: dict[str, any]):
        return cls(**row)


class MatchMetadataPlayer(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    account_id: int
    player_slot: int
    team: str
    kills: int
    deaths: int
    assists: int
    net_worth: int
    hero_id: int
    last_hits: int
    denies: int
    ability_points: int
    party: int
    assigned_lane: int
    player_level: int
    abandon_match_time_s: int
    ability_stats: dict[int, int]
    stats_type_stat: list[float]
    book_reward: list[MatchMetadataPlayerBookReward]
    death_details: list[MatchMetadataPlayerDeathDetails]
    items: list[MatchMetadataPlayerItems]

    @classmethod
    def from_dict(cls, row: dict[str, any]):
        book_reward = {
            k.replace("book_reward.", ""): v
            for k, v in row.items()
            if k.startswith("book_reward.")
        }
        book_reward = [dict(zip(book_reward, t)) for t in zip(*book_reward.values())]
        items = {
            k.replace("items.", ""): v for k, v in row.items() if k.startswith("items.")
        }
        items = [dict(zip(items, t)) for t in zip(*items.values())]
        death_details = {
            k.replace("death_details.", ""): v
            for k, v in row.items()
            if k.startswith("death_details.")
        }
        death_details = [
            dict(zip(death_details, t)) for t in zip(*death_details.values())
        ]
        return cls(
            account_id=row["account_id"],
            player_slot=row["player_slot"],
            team=row["team"],
            kills=row["kills"],
            deaths=row["deaths"],
            assists=row["assists"],
            net_worth=row["net_worth"],
            hero_id=row["hero_id"],
            last_hits=row["last_hits"],
            denies=row["denies"],
            ability_points=row["ability_points"],
            party=row["party"],
            assigned_lane=row["assigned_lane"],
            player_level=row["player_level"],
            abandon_match_time_s=row["abandon_match_time_s"],
            ability_stats=row["ability_stats"],
            stats_type_stat=row["stats_type_stat"],
            book_reward=[
                MatchMetadataPlayerBookReward.from_dict(row) for row in book_reward
            ],
            death_details=[
                MatchMetadataPlayerDeathDetails.from_dict(row) for row in death_details
            ],
            items=[MatchMetadataPlayerItems.from_dict(row) for row in items],
        )


class MatchMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    match_id: int
    start_time: datetime.datetime
    winning_team: str
    duration_s: int
    match_outcome: str
    match_mode: str
    game_mode: str
    sample_time_s: list[int]
    stat_type: list[int]
    source_name: list[str]
    objectives_mask_team0: int
    objectives_mask_team1: int
    is_high_skill_range_parties: bool | None = Field(None)
    low_pri_pool: bool | None = Field(None)
    new_player_pool: bool | None = Field(None)
    objectives: list[MatchMetadataObjectives]
    mid_boss: list[MatchMetadataMidBoss]
    players: list[MatchMetadataPlayer]

    @classmethod
    def from_rows(
        cls,
        match_info: dict[str, any],
        match_players: list[dict[str, any]],
    ):
        objectives = {
            k.replace("objectives.", ""): v
            for k, v in match_info.items()
            if k.startswith("objectives.")
        }
        objectives = [dict(zip(objectives, t)) for t in zip(*objectives.values())]
        mid_boss = {
            k.replace("mid_boss.", ""): v
            for k, v in match_info.items()
            if k.startswith("mid_boss.")
        }
        mid_boss = [dict(zip(mid_boss, t)) for t in zip(*mid_boss.values())]
        return cls(
            match_id=match_info["match_id"],
            start_time=match_info["start_time"],
            winning_team=match_info["winning_team"],
            duration_s=match_info["duration_s"],
            match_outcome=match_info["match_outcome"],
            match_mode=match_info["match_mode"],
            game_mode=match_info["game_mode"],
            sample_time_s=match_info["sample_time_s"],
            stat_type=match_info["stat_type"],
            source_name=match_info["source_name"],
            objectives_mask_team0=match_info["objectives_mask_team0"],
            objectives_mask_team1=match_info["objectives_mask_team1"],
            is_high_skill_range_parties=match_info.get("is_high_skill_range_parties"),
            low_pri_pool=match_info.get("low_pri_pool"),
            new_player_pool=match_info.get("new_player_pool"),
            objectives=[MatchMetadataObjectives.from_dict(row) for row in objectives],
            mid_boss=[MatchMetadataMidBoss.from_dict(row) for row in mid_boss],
            players=[MatchMetadataPlayer.from_dict(row) for row in match_players],
        )


if __name__ == "__main__":
    from deadlock_analytics_api.globs import CH_POOL

    query = "SELECT * FROM match_info WHERE match_id = %(match_id)s LIMIT 1"
    with CH_POOL.get_client() as client:
        match_info, keys = client.execute(
            query, {"match_id": 22836410}, with_column_types=True
        )
    match_info = {k: v for (k, _), v in zip(keys, match_info[0])}

    query = "SELECT * FROM match_player WHERE match_id = %(match_id)s LIMIT 12"
    with CH_POOL.get_client() as client:
        match_players, keys = client.execute(
            query, {"match_id": 22836410}, with_column_types=True
        )
    match_players = [{k: v for (k, _), v in zip(keys, row)} for row in match_players]

    MatchMetadata.from_rows(match_info, match_players)
