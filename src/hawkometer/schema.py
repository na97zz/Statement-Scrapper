from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


Direction = Literal["hawkish", "dovish", "neutral"]


@dataclass(frozen=True)
class PhraseEntry:
    phrase: str
    direction: Direction
    weight: float
    theme: str
    central_banks: list[str]
    notes: str
    enabled: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhraseEntry":
        return cls(
            phrase=str(data["phrase"]),
            direction=data["direction"],
            weight=float(data["weight"]),
            theme=str(data["theme"]),
            central_banks=[str(item) for item in data["central_banks"]],
            notes=str(data.get("notes", "")),
            enabled=bool(data["enabled"]),
        )


@dataclass(frozen=True)
class HawkometerConfig:
    metadata: dict[str, Any]
    phrase_library: list[PhraseEntry]
    themes: dict[str, Any]
    central_bank_overrides: dict[str, Any]
    regime_adjustments: dict[str, Any]
    scoring_config: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HawkometerConfig":
        return cls(
            metadata=dict(data["metadata"]),
            phrase_library=[PhraseEntry.from_dict(item) for item in data["phrase_library"]],
            themes=dict(data["themes"]),
            central_bank_overrides=dict(data["central_bank_overrides"]),
            regime_adjustments=dict(data["regime_adjustments"]),
            scoring_config=dict(data["scoring_config"]),
        )
