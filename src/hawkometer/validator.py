from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {
    "metadata",
    "phrase_library",
    "themes",
    "central_bank_overrides",
    "regime_adjustments",
    "scoring_config",
}

REQUIRED_PHRASE_KEYS = {
    "phrase",
    "direction",
    "weight",
    "theme",
    "central_banks",
    "notes",
    "enabled",
}

VALID_DIRECTIONS = {"hawkish", "dovish", "neutral"}


class HawkometerValidationError(ValueError):
    """Raised when a Hawkometer configuration is invalid."""


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_TOP_LEVEL_KEYS - set(config)
    for key in sorted(missing):
        errors.append(f"Missing top-level key: {key}")

    phrase_library = config.get("phrase_library", [])
    if not isinstance(phrase_library, list):
        errors.append("phrase_library must be a list")
        return errors

    themes = config.get("themes", {})
    for index, phrase in enumerate(phrase_library):
        if not isinstance(phrase, dict):
            errors.append(f"phrase_library[{index}] must be an object")
            continue
        phrase_missing = REQUIRED_PHRASE_KEYS - set(phrase)
        for key in sorted(phrase_missing):
            errors.append(f"phrase_library[{index}] missing key: {key}")
        direction = phrase.get("direction")
        if direction not in VALID_DIRECTIONS:
            errors.append(f"phrase_library[{index}] has invalid direction: {direction}")
        try:
            weight = float(phrase.get("weight"))
            if weight < -3 or weight > 3:
                errors.append(f"phrase_library[{index}] weight outside [-3, 3]: {weight}")
        except (TypeError, ValueError):
            errors.append(f"phrase_library[{index}] weight must be numeric")
        if not isinstance(phrase.get("central_banks"), list) or not phrase.get("central_banks"):
            errors.append(f"phrase_library[{index}] central_banks must be a non-empty list")
        if not isinstance(phrase.get("enabled"), bool):
            errors.append(f"phrase_library[{index}] enabled must be true/false")
        theme = phrase.get("theme")
        if isinstance(themes, dict) and theme and theme not in themes:
            errors.append(f"phrase_library[{index}] references unknown theme: {theme}")
    return errors


def validate_config_file(path: str | Path) -> list[str]:
    return validate_config(load_json(path))


def assert_valid_config(config: dict[str, Any]) -> None:
    errors = validate_config(config)
    if errors:
        raise HawkometerValidationError("; ".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a local Hawkometer JSON config.")
    parser.add_argument("config", nargs="?", default="data/hawkometer.json", help="Path to hawkometer.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.config)
    config = load_json(path)
    errors = validate_config(config)
    if errors:
        print(f"INVALID: {path}")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    phrases = config.get("phrase_library", [])
    themes = config.get("themes", {})
    directions = Counter(phrase.get("direction", "") for phrase in phrases)
    banks = sorted({bank for phrase in phrases for bank in phrase.get("central_banks", [])})
    print(f"OK: {path} is valid")
    print(f"phrases: {len(phrases)}")
    print(f"themes: {len(themes)}")
    print("directions: " + ", ".join(f"{key}={directions[key]}" for key in sorted(directions)))
    print("central_banks: " + ", ".join(banks))


if __name__ == "__main__":
    main()
