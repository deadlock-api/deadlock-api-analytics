from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator


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


class ItemWinLossStat(BaseModel):
    hero_id: int
    item_id: int
    wins: int
    losses: int


class PlayerCardSlot(BaseModel):
    slots_id: int
    hero_id: int
    hero_kills: int
    hero_wins: int
    # stat_id: int # Always 0
    # stat_score: int # Always 0


class PlayerCardHistoryEntry(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    created_at: datetime
    slots: list[PlayerCardSlot]
    ranked_badge_level: int

    @field_validator("created_at", mode="before")
    @classmethod
    def utc_created_at(cls, v: datetime) -> datetime:
        return v.astimezone(timezone.utc)

    @classmethod
    def from_row(cls, row) -> "PlayerCardHistoryEntry":
        return cls(
            slots=[
                PlayerCardSlot(
                    **{
                        k.replace("slots_", "", 1): v[0] if isinstance(v, list) else v
                        for k, v in row.items()
                        if k.startswith("slots_")
                    }
                )
            ],
            **{k: v for k, v in row.items() if not k.startswith("slots_")},
        )

    @computed_field
    @property
    def match_ranked_rank(self) -> int | None:
        return self.ranked_badge_level // 10 if self.ranked_badge_level is not None else None

    @computed_field
    @property
    def match_ranked_subrank(self) -> int | None:
        return self.ranked_badge_level % 10 if self.ranked_badge_level is not None else None


class PlayerMMRHistoryEntryV2(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    match_id: int
    won: bool = Field(False)
    source: str
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


class PlayerHeroStat(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    hero_id: int
    matches: int
    wins: int
    kills: int
    deaths: int
    assists: int
    ending_level: float
    denies_per_match: float
    networth_per_min: float
    damage_mitigated_per_min: float
    damage_taken_per_min: float
    creeps_per_min: float
    denies_per_min: float
    obj_damage_per_min: float
    accuracy: float
    crit_shot_rate: float
    highest_ranked_badge_level: int | None = None

    @computed_field
    @property
    def highest_ranked_rank(self) -> int | None:
        return (
            self.highest_ranked_badge_level // 10
            if self.highest_ranked_badge_level is not None
            else None
        )

    @computed_field
    @property
    def highest_ranked_subrank(self) -> int | None:
        return (
            self.highest_ranked_badge_level % 10
            if self.highest_ranked_badge_level is not None
            else None
        )


class PlayerItemStat(BaseModel):
    account_id: int = Field(description="The account id of the player, it's a SteamID3")
    hero_id: int
    item_id: int
    wins: int
    matches: int


class PlayerMate(BaseModel):
    mate_id: int = Field(description="The account id of the mate, it's a SteamID3")
    wins: int
    matches_played: int
    matches: list[int]


class PlayerParty(BaseModel):
    party_size: int
    wins: int
    matches_played: int
    matches: list[int]
