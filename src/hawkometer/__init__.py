"""Local phrase-based Hawkometer scoring package."""

__all__ = [
    "HawkometerDiffScorer",
    "HawkometerScorer",
    "HawkometerConfig",
    "PhraseEntry",
    "classify_score",
    "confidence_score",
    "score_statement_diff",
    "score_text",
    "validate_config",
]


def __getattr__(name):
    if name in {"HawkometerScorer", "score_text"}:
        from .scorer import HawkometerScorer, score_text

        return {"HawkometerScorer": HawkometerScorer, "score_text": score_text}[name]
    if name in {"HawkometerDiffScorer", "score_statement_diff"}:
        from .diff_scorer import HawkometerDiffScorer, score_statement_diff

        return {
            "HawkometerDiffScorer": HawkometerDiffScorer,
            "score_statement_diff": score_statement_diff,
        }[name]
    if name in {"HawkometerConfig", "PhraseEntry"}:
        from .schema import HawkometerConfig, PhraseEntry

        return {"HawkometerConfig": HawkometerConfig, "PhraseEntry": PhraseEntry}[name]
    if name in {"classify_score", "confidence_score"}:
        from .classifier import classify_score, confidence_score

        return {"classify_score": classify_score, "confidence_score": confidence_score}[name]
    if name == "validate_config":
        from .validator import validate_config

        return validate_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
