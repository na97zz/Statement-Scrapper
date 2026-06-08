from __future__ import annotations

import json
import math
import re
import argparse
from pathlib import Path
from typing import Any

from .classifier import classification_summary
from .schema import HawkometerConfig, PhraseEntry
from .validator import assert_valid_config


DEFAULT_CONFIG_PATH = Path("data/hawkometer.json")


class HawkometerScorer:
    """Deterministic phrase-based scorer for central bank communication."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        raw_config = self._load_config(self.config_path)
        assert_valid_config(raw_config)
        self.config = HawkometerConfig.from_dict(raw_config)
        self.score_min = float(self.config.scoring_config.get("score_min", -10))
        self.score_max = float(self.config.scoring_config.get("score_max", 10))

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def score(self, text: str, central_bank: str | None = None) -> dict[str, Any]:
        bank = (central_bank or "").upper()
        matches = []
        hawkish_score = 0.0
        dovish_score = 0.0

        for entry in self.config.phrase_library:
            if not entry.enabled:
                continue
            if bank and not self._entry_applies_to_bank(entry, bank):
                continue
            count = self._count_phrase(text, entry.phrase)
            if count == 0:
                continue
            scaled_count = self._scale_count(count)
            signed_contribution = self._signed_contribution(entry, scaled_count)
            if entry.direction == "hawkish":
                hawkish_score += abs(entry.weight) * scaled_count
            elif entry.direction == "dovish":
                dovish_score += abs(entry.weight) * scaled_count
            matches.append(
                {
                    "phrase": entry.phrase,
                    "direction": entry.direction,
                    "theme": entry.theme,
                    "count": count,
                    "weight": entry.weight,
                    "scaled_count": scaled_count,
                    "scaled_contribution": signed_contribution,
                    "notes": entry.notes,
                }
            )

        raw_score = hawkish_score - dovish_score
        final_score = self._clip(raw_score)
        classification = classification_summary(final_score, matches)
        return {
            "score": final_score,
            "classification": classification["classification"],
            "confidence": classification["confidence"],
            "lean": classification["lean"],
            "matched_phrases": matches,
            "hawkish_score": hawkish_score,
            "dovish_score": dovish_score,
            "raw_score": raw_score,
            "final_score": final_score,
            "score_min": self.score_min,
            "score_max": self.score_max,
        }

    @staticmethod
    def _entry_applies_to_bank(entry: PhraseEntry, bank: str) -> bool:
        banks = {item.upper() for item in entry.central_banks}
        return "ALL" in banks or bank in banks

    @staticmethod
    def _count_phrase(text: str, phrase: str) -> int:
        escaped = re.escape(phrase.strip())
        escaped = escaped.replace(r"\ ", r"\s+")
        pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)
        return len(pattern.findall(text or ""))

    @staticmethod
    def _scale_count(count: int) -> float:
        if count <= 0:
            return 0.0
        return 1.0 + math.log(count)

    @staticmethod
    def _signed_contribution(entry: PhraseEntry, scaled_count: float) -> float:
        if entry.direction == "neutral":
            return 0.0
        if entry.direction == "hawkish":
            return abs(entry.weight) * scaled_count
        return -abs(entry.weight) * scaled_count

    def _clip(self, score: float) -> float:
        return max(self.score_min, min(self.score_max, score))


def score_text(text: str, config_path: str | Path = DEFAULT_CONFIG_PATH, central_bank: str | None = None) -> dict[str, Any]:
    return HawkometerScorer(config_path).score(text, central_bank=central_bank)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score text with the local Hawkometer phrase scorer.")
    parser.add_argument("text", nargs="?", default="", help="Text to score. If omitted, use --file.")
    parser.add_argument("--file", help="Path to a text file to score.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to hawkometer.json.")
    parser.add_argument("--bank", default="", help="Optional central bank code, e.g. FED, ECB, BOE.")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    else:
        text = args.text
    if not text:
        raise SystemExit("No text provided. Pass text as an argument or use --file path/to/text.txt")

    result = score_text(text, config_path=args.config, central_bank=args.bank or None)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(f"score: {result['score']:.3f}")
    print(f"classification: {result['classification']}")
    print(f"confidence: {result['confidence']:.3f}")
    print(f"lean: {result['lean']}")
    print(f"raw_score: {result['raw_score']:.3f}")
    print(f"hawkish_score: {result['hawkish_score']:.3f}")
    print(f"dovish_score: {result['dovish_score']:.3f}")
    print(f"matched_phrases: {len(result['matched_phrases'])}")
    for match in result["matched_phrases"]:
        print(
            "- "
            f"{match['phrase']} | {match['direction']} | "
            f"count={match['count']} | weight={match['weight']} | "
            f"contribution={match['scaled_contribution']:.3f}"
        )


if __name__ == "__main__":
    main()
