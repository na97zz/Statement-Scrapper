from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .scorer import DEFAULT_CONFIG_PATH, HawkometerScorer


class HawkometerDiffScorer:
    """Compare two statements and score the directional wording impulse."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.scorer = HawkometerScorer(config_path)

    def compare(
        self,
        previous_text: str,
        current_text: str,
        central_bank: str | None = None,
    ) -> dict[str, Any]:
        previous_score = self.scorer.score(previous_text, central_bank=central_bank)
        current_score = self.scorer.score(current_text, central_bank=central_bank)

        previous_counts = self._phrase_counts(previous_text, central_bank)
        current_counts = self._phrase_counts(current_text, central_bank)

        added_matches = []
        removed_matches = []
        strengthened_wording = []
        softened_wording = []
        diff_score = 0.0

        for phrase in sorted(set(previous_counts) | set(current_counts)):
            previous = previous_counts.get(phrase, {})
            current = current_counts.get(phrase, {})
            previous_count = int(previous.get("count", 0))
            current_count = int(current.get("count", 0))
            if previous_count == current_count:
                continue

            metadata = current or previous
            direction = metadata["direction"]
            weight = float(metadata["weight"])
            delta_count = current_count - previous_count
            contribution = self._diff_contribution(direction, weight, delta_count)
            diff_score += contribution

            change = {
                "phrase": phrase,
                "direction": direction,
                "theme": metadata["theme"],
                "weight": weight,
                "previous_count": previous_count,
                "current_count": current_count,
                "delta_count": delta_count,
                "diff_contribution": contribution,
            }

            if delta_count > 0:
                added_matches.append(change)
                if contribution > 0:
                    strengthened_wording.append(change)
                elif contribution < 0:
                    softened_wording.append(change)
            else:
                removed_matches.append(change)
                if contribution > 0:
                    strengthened_wording.append(change)
                elif contribution < 0:
                    softened_wording.append(change)

        final_adjusted_score = self.scorer._clip(current_score["final_score"] + diff_score)
        return {
            "previous_score": previous_score,
            "current_score": current_score,
            "diff_score": diff_score,
            "final_adjusted_score": final_adjusted_score,
            "added_matches": added_matches,
            "removed_matches": removed_matches,
            "added_hawkish_phrases": self._filter_changes(added_matches, "hawkish"),
            "removed_hawkish_phrases": self._filter_changes(removed_matches, "hawkish"),
            "added_dovish_phrases": self._filter_changes(added_matches, "dovish"),
            "removed_dovish_phrases": self._filter_changes(removed_matches, "dovish"),
            "strengthened_wording": strengthened_wording,
            "softened_wording": softened_wording,
            "explanation_summary": self._explain(
                added_matches,
                removed_matches,
                strengthened_wording,
                softened_wording,
                diff_score,
                final_adjusted_score,
            ),
        }

    def _phrase_counts(self, text: str, central_bank: str | None) -> dict[str, dict[str, Any]]:
        bank = (central_bank or "").upper()
        counts: dict[str, dict[str, Any]] = {}
        for entry in self.scorer.config.phrase_library:
            if not entry.enabled:
                continue
            if bank and not self.scorer._entry_applies_to_bank(entry, bank):
                continue
            count = self.scorer._count_phrase(text, entry.phrase)
            if count <= 0:
                continue
            counts[entry.phrase] = {
                "count": count,
                "direction": entry.direction,
                "theme": entry.theme,
                "weight": entry.weight,
            }
        return counts

    @staticmethod
    def _diff_contribution(direction: str, weight: float, delta_count: int) -> float:
        if direction == "neutral":
            return 0.0
        impulse = abs(weight) * abs(delta_count)
        if direction == "hawkish":
            return impulse if delta_count > 0 else -impulse
        if direction == "dovish":
            return -impulse if delta_count > 0 else impulse
        return 0.0

    @staticmethod
    def _filter_changes(changes: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
        return [change for change in changes if change["direction"] == direction]

    @staticmethod
    def _top_phrases(changes: list[dict[str, Any]], limit: int = 3) -> str:
        if not changes:
            return "none"
        counts = Counter(change["phrase"] for change in changes)
        return ", ".join(phrase for phrase, _ in counts.most_common(limit))

    def _explain(
        self,
        added_matches: list[dict[str, Any]],
        removed_matches: list[dict[str, Any]],
        strengthened_wording: list[dict[str, Any]],
        softened_wording: list[dict[str, Any]],
        diff_score: float,
        final_adjusted_score: float,
    ) -> str:
        direction = "hawkish" if diff_score > 0 else "dovish" if diff_score < 0 else "neutral"
        return (
            f"Statement diff impulse is {direction} ({diff_score:.2f}). "
            f"Added phrases: {len(added_matches)}; removed phrases: {len(removed_matches)}. "
            f"Strengthened wording: {self._top_phrases(strengthened_wording)}. "
            f"Softened wording: {self._top_phrases(softened_wording)}. "
            f"Final adjusted score: {final_adjusted_score:.2f}."
        )


def score_statement_diff(
    previous_text: str,
    current_text: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    central_bank: str | None = None,
) -> dict[str, Any]:
    return HawkometerDiffScorer(config_path).compare(previous_text, current_text, central_bank=central_bank)
