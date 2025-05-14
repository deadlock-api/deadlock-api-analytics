from typing import Literal

from pydantic import BaseModel, Field, computed_field


class PlayerLeaderboardV2(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    region_mode: Literal["Row", "Europe", "SEAsia", "SAmerica", "Russia", "Oceania"] | None = Field(
        None
    )
    leaderboard_rank: int
    wins: int
    matches_played: int
    kills: int
    deaths: int
    assists: int
    ranked_badge_level: int | None = None

    @computed_field
    @property
    def ranked_rank(self) -> int | None:
        return self.ranked_badge_level // 10 if self.ranked_badge_level is not None else None

    @computed_field
    @property
    def ranked_subrank(self) -> int | None:
        return self.ranked_badge_level % 10 if self.ranked_badge_level is not None else None


class PlayerHeroStat(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    hero_id: int
    matches: int
    wins: int
    ending_level: float
    kills: int
    deaths: int
    assists: int
    denies_per_match: float
    kills_per_min: float
    deaths_per_min: float
    assists_per_min: float
    denies_per_min: float
    networth_per_min: float
    last_hits_per_min: float
    damage_mitigated_per_min: float
    damage_taken_per_min: float
    creeps_per_min: float
    obj_damage_per_min: float
    accuracy: float
    crit_shot_rate: float


class PlayerItemStat(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    hero_id: int
    item_id: int
    wins: int
    matches: int


class HeroCombsWinLossStat(BaseModel):
    hero_ids: list[int]
    wins: int
    losses: int
    matches: int
    total_kills: int = Field(description="The total number of kills over all matches")
    total_deaths: int = Field(description="The total number of deaths over all matches")
    total_assists: int = Field(description="The total number of assists over all matches")


class HeroMatchUpWinLossStatMatchUp(BaseModel):
    hero_id: int
    wins: int
    losses: int
    matches: int
    total_kills: int = Field(description="The total number of kills over all matches")
    total_kills_received: int = Field(
        description="The total number of kills received over all matches"
    )
    total_deaths: int = Field(description="The total number of deaths over all matches")
    total_deaths_received: int = Field(
        description="The total number of deaths received over all matches"
    )
    total_assists: int = Field(description="The total number of assists over all matches")
    total_assists_received: int = Field(
        description="The total number of assists received over all matches"
    )


class HeroMatchUpWinLossStat(BaseModel):
    hero_id: int
    matchups: list[HeroMatchUpWinLossStatMatchUp]
