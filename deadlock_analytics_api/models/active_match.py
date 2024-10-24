import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field

ACTIVE_MATCHES_KEYS = [
    "`players.team`",
    "`players.account_id`",
    "`players.abandoned`",
    "`players.hero_id`",
    "start_time",
    "winning_team",
    "match_id",
    "lobby_id",
    "net_worth_team_0",
    "net_worth_team_1",
    "duration_s",
    "spectators",
    "open_spectator_slots",
    "objectives_mask_team0",
    "objectives_mask_team1",
    "match_mode",
    "game_mode",
    "match_score",
    "region_mode",
    "scraped_at",
    "compat_version",
    "ranked_badge_level",
]

ACTIVE_MATCHES_REDUCED_KEYS = [
    "players.team",
    "players.account_id",
    "players.abandoned",
    "players.hero_id",
    "start_time",
    "match_id",
    "net_worth_team_0",
    "net_worth_team_1",
    "objectives_mask_team0",
    "objectives_mask_team1",
    "match_mode",
    "game_mode",
    "match_score",
    "region_mode",
    "winner",
    "compat_version",
    "ranked_badge_level",
]


class ActiveMatchPlayer(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    account_id: int
    team: int
    abandoned: bool
    hero_id: int


class ActiveMatchObjectives(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    core: bool
    tier1_lane1: bool
    tier1_lane2: bool
    tier1_lane3: bool
    tier1_lane4: bool
    tier2_lane1: bool
    tier2_lane2: bool
    tier2_lane3: bool
    tier2_lane4: bool
    titan: bool
    titan_shield_generator_1: bool
    titan_shield_generator_2: bool
    barrack_boss_lane1: bool
    barrack_boss_lane2: bool
    barrack_boss_lane3: bool
    barrack_boss_lane4: bool

    @classmethod
    def from_mask(cls, mask: int):
        return cls(
            core=bool(mask & (1 << 0)),
            tier1_lane1=bool(mask & (1 << 1)),
            tier1_lane2=bool(mask & (1 << 2)),
            tier1_lane3=bool(mask & (1 << 3)),
            tier1_lane4=bool(mask & (1 << 4)),
            tier2_lane1=bool(mask & (1 << 5)),
            tier2_lane2=bool(mask & (1 << 6)),
            tier2_lane3=bool(mask & (1 << 7)),
            tier2_lane4=bool(mask & (1 << 8)),
            titan=bool(mask & (1 << 9)),
            titan_shield_generator_1=bool(mask & (1 << 10)),
            titan_shield_generator_2=bool(mask & (1 << 11)),
            barrack_boss_lane1=bool(mask & (1 << 12)),
            barrack_boss_lane2=bool(mask & (1 << 13)),
            barrack_boss_lane3=bool(mask & (1 << 14)),
            barrack_boss_lane4=bool(mask & (1 << 15)),
        )


class ActiveMatch(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    match_id: int
    start_time: str
    scraped_at: str
    winning_team: int
    players: list[ActiveMatchPlayer]
    lobby_id: int
    net_worth_team_0: int
    net_worth_team_1: int
    duration_s: int
    spectators: int
    open_spectator_slots: int
    match_mode: str
    game_mode: str
    match_score: int
    region_mode: str
    objectives_mask_team0: int
    objectives_mask_team1: int
    compat_version: int | None = Field(None)
    ranked_badge_level: int | None = Field(None)

    @computed_field
    @property
    def objectives_team0(self) -> ActiveMatchObjectives | None:
        return ActiveMatchObjectives.from_mask(self.objectives_mask_team0)

    @computed_field
    @property
    def objectives_team1(self) -> ActiveMatchObjectives | None:
        return ActiveMatchObjectives.from_mask(self.objectives_mask_team1)

    @classmethod
    def from_row(cls, row) -> "ActiveMatch":
        return cls(
            **{
                k: col if not isinstance(col, datetime.datetime) else col.isoformat()
                for k, col in zip(ACTIVE_MATCHES_KEYS, row)
                if not "players" in k
            },
            players=[
                ActiveMatchPlayer(
                    account_id=account_id,
                    team=team,
                    abandoned=abandoned,
                    hero_id=hero_id,
                )
                for team, account_id, abandoned, hero_id in zip(
                    row[0], row[1], row[2], row[3]
                )
            ],
        )
