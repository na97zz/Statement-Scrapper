from __future__ import annotations

from collections import Counter
from typing import Any


def classify_score(score: float) -> dict[str, str]:
    """Classify a clipped -10 to +10 Hawkometer score."""
    if score >= 8:
        classification = "Strongly Hawkish"
    elif score >= 3:
        classification = "Hawkish"
    elif score >= 0.5:
        classification = "Neutral-Hawkish"
    elif score >= -0.5:
        classification = "Neutral"
    elif score > -3:
        classification = "Neutral-Dovish"
    elif score > -8:
        classification = "Dovish"
    else:
        classification = "Strongly Dovish"

    if score > 0.5:
        lean = "hawkish"
    elif score < -0.5:
        lean = "dovish"
    else:
        lean = "none"
    return {"classification": classification, "lean": lean}


def confidence_score(score: float, matched_phrases: list[dict[str, Any]]) -> float:
    """Return deterministic confidence between 0 and 1.

    Confidence rises with phrase evidence, theme diversity, average absolute
    phrase weight, agreement between directional phrases, and distance from
    zero. It falls when evidence is sparse, conflicted, or concentrated.
    """
    if not matched_phrases:
        return 0.0

    directional = [m for m in matched_phrases if m.get("direction") in {"hawkish", "dovish"}]
    if not directional:
        return 0.15

    total_count = sum(int(m.get("count", 1)) for m in directional)
    themes = [str(m.get("theme", "")) for m in directional if m.get("theme")]
    unique_themes = len(set(themes))
    avg_abs_weight = sum(abs(float(m.get("weight", 0))) for m in directional) / len(directional)

    hawkish_count = sum(int(m.get("count", 1)) for m in directional if m.get("direction") == "hawkish")
    dovish_count = sum(int(m.get("count", 1)) for m in directional if m.get("direction") == "dovish")
    agreement = abs(hawkish_count - dovish_count) / max(hawkish_count + dovish_count, 1)

    evidence_component = min(total_count / 8.0, 1.0) * 0.25
    diversity_component = min(unique_themes / 5.0, 1.0) * 0.20
    weight_component = min(avg_abs_weight / 3.0, 1.0) * 0.20
    agreement_component = agreement * 0.20
    distance_component = min(abs(float(score)) / 5.0, 1.0) * 0.15

    confidence = (
        evidence_component
        + diversity_component
        + weight_component
        + agreement_component
        + distance_component
    )

    if len(directional) == 1:
        confidence *= 0.65
    if unique_themes == 1 and len(directional) > 2:
        confidence *= 0.85
    if hawkish_count and dovish_count:
        confidence *= 0.75
    if -0.5 <= score <= 0.5:
        confidence *= 0.75

    return round(max(0.0, min(1.0, confidence)), 3)


def classification_summary(score: float, matched_phrases: list[dict[str, Any]]) -> dict[str, Any]:
    result = classify_score(score)
    result["confidence"] = confidence_score(score, matched_phrases)
    result["theme_count"] = len({m.get("theme") for m in matched_phrases if m.get("theme")})
    result["direction_counts"] = dict(Counter(m.get("direction") for m in matched_phrases))
    return result
