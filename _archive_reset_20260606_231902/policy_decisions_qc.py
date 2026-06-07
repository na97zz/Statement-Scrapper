from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REVIEW_COLUMNS = [
    "bank",
    "meeting_date",
    "meeting_id",
    "current_decision",
    "current_rate_change_bps",
    "current_policy_rate_after",
    "current_confidence",
    "extraction_method",
    "needs_review",
    "review_reason",
    "qc_priority",
    "suggested_decision",
    "suggested_rate_change_bps",
    "suggested_policy_rate_after",
    "suggested_confidence",
    "suggestion_reason",
    "matched_pattern",
    "matched_excerpt",
    "source_title",
    "source_url",
    "notes",
]

SUGGESTED_COLUMNS = [
    "bank",
    "meeting_date",
    "meeting_id",
    "decision",
    "rate_change_bps",
    "policy_rate_after",
    "confidence",
    "override_reason",
    "matched_excerpt",
]

OVERRIDE_TEMPLATE_COLUMNS = [
    "bank",
    "meeting_date",
    "meeting_id",
    "override_decision",
    "override_rate_change_bps",
    "override_policy_rate_after",
    "override_confidence",
    "override_notes",
    "approved",
]

FINAL_EXTRA_COLUMNS = [
    "qc_status",
    "needs_review",
    "review_reason",
    "suggested_decision",
    "suggested_rate_change_bps",
    "suggested_policy_rate_after",
    "suggested_confidence",
    "override_notes",
]


@dataclass
class Suggestion:
    decision: str = ""
    rate_change_bps: float | None = None
    policy_rate_after: float | None = None
    confidence: float = 0.0
    reason: str = ""
    matched_pattern: str = ""
    matched_excerpt: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC and manual override workflow for policy_decisions.csv.")
    parser.add_argument("--input", default="data/external/policy_decisions.csv", help="Raw policy decisions CSV.")
    parser.add_argument("--qa", default="data/external/policy_decisions_qa.csv", help="Policy decisions QA CSV.")
    parser.add_argument("--meeting-linkage", default="processed/meeting_linkage", help="Meeting linkage folder.")
    parser.add_argument("--output-dir", default="data/external", help="Output folder for QC files.")
    parser.add_argument("--overrides", default="data/external/policy_decisions_overrides.csv", help="Optional approved overrides CSV.")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Confidence threshold for review.")
    parser.add_argument("--suggestion-threshold", type=float, default=0.75, help="Suggestion confidence threshold.")
    parser.add_argument("--start-date", default="2021-01-01", help="Start date filter.")
    parser.add_argument("--end-date", default="2026-12-31", help="End date filter.")
    return parser.parse_args()


def read_csv_safe(path: Path, optional: bool = False) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        if optional:
            print(f"Warning: optional file missing: {path}")
        else:
            print(f"Warning: input file missing: {path}")
        return pd.DataFrame(), False
    return pd.read_csv(path, dtype=str, keep_default_na=False), True


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def filter_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_date"] = pd.to_datetime(df.get("meeting_date", ""), errors="coerce")
    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise ValueError("--start-date and --end-date must be valid dates.")
    df = df[df["_date"].between(start, end, inclusive="both")].copy()
    df["meeting_date"] = df["_date"].dt.strftime("%Y-%m-%d")
    return df.drop(columns=["_date"])


def to_number(value: Any) -> float:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "approved"}


def decision_inconsistent(decision: str, rate_change_bps: Any) -> bool:
    decision_l = str(decision or "").strip().lower()
    rate = to_number(rate_change_bps)
    if pd.isna(rate) or decision_l == "unknown":
        return False
    if decision_l == "hike" and rate <= 0:
        return True
    if decision_l == "cut" and rate >= 0:
        return True
    if decision_l == "hold" and rate != 0:
        return True
    return False


def review_reasons(row: pd.Series, confidence_threshold: float) -> list[str]:
    reasons = []
    confidence = to_number(row.get("confidence", ""))
    decision = str(row.get("decision", "")).strip().lower()
    if pd.isna(confidence) or confidence < confidence_threshold:
        reasons.append(f"confidence below {confidence_threshold:g}")
    if decision == "unknown" or not decision:
        reasons.append("decision unknown")
    if str(row.get("rate_change_bps", "")).strip() == "":
        reasons.append("rate_change_bps missing")
    if str(row.get("policy_rate_after", "")).strip() == "":
        reasons.append("policy_rate_after missing")
    if "keyword" in str(row.get("extraction_method", "")).lower():
        reasons.append("keyword-only extraction")
    if str(row.get("source_title", "")).strip() == "":
        reasons.append("source_title missing")
    if str(row.get("text_excerpt", "")).strip() == "":
        reasons.append("text_excerpt missing")
    if decision_inconsistent(decision, row.get("rate_change_bps", "")):
        reasons.append("decision inconsistent with rate_change_bps")
    return reasons


def priority_for(reasons: list[str]) -> str:
    high = {"decision unknown", "decision inconsistent with rate_change_bps"}
    if any(reason in high for reason in reasons):
        return "high"
    medium_prefixes = ("confidence below", "rate_change_bps missing", "policy_rate_after missing")
    if any(reason.startswith(medium_prefixes) for reason in reasons):
        return "medium"
    return "low"


def parse_percent(value: str) -> float | None:
    text = str(value).strip().lower().replace("%", "").replace("percent", "")
    text = text.replace("¼", "/4").replace("½", "/2").replace("¾", "/4")
    text = text.replace("1/4", ".25").replace("1/2", ".5").replace("3/4", ".75").replace("-", " ")
    parts = text.split()
    try:
        if len(parts) == 2 and parts[1].startswith("."):
            return float(parts[0]) + float(parts[1])
        return float(text)
    except ValueError:
        return None


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = text.replace("â€“", "-").replace("â€”", "-").replace("â€™", "'")
    text = text.replace("Â½", "1/2").replace("Â¼", "1/4").replace("Â¾", "3/4")
    return re.sub(r"\s+", " ", text)


def make_excerpt(text: str, match: re.Match[str] | None, radius: int = 250) -> str:
    if not match:
        return ""
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return text[start:end].strip()


def extract_policy_rate(text: str) -> tuple[float | None, str, str, str]:
    patterns = [
        (r"(target range for the federal funds rate|federal funds rate)\s+at\s+([0-9][0-9./¼½¾ -]*)\s+to\s+([0-9][0-9./¼½¾ -]*)\s+percent", "fed_target_range"),
        (r"(policy rate|cash rate target|official cash rate|bank rate|deposit facility rate|main refinancing operations rate|key policy rate|sight deposit rate)\s+(?:at|to)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)", "policy_rate"),
        (r"(kept|maintained|left)\s+(?:the\s+)?(policy rate|cash rate target|official cash rate|bank rate|key policy rate)\s+(?:unchanged\s+)?at\s+([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)", "unchanged_policy_rate"),
        (r"(remains|remain)\s+at\s+([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)", "remains_at"),
    ]
    for pattern, label in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        if label == "fed_target_range":
            low = parse_percent(match.group(2))
            high = parse_percent(match.group(3))
            if low is None or high is None:
                continue
            midpoint = (low + high) / 2
            return midpoint, label, f"Fed target range {low:g} to {high:g}; midpoint used.", make_excerpt(text, match)
        value = parse_percent(match.groups()[-1])
        if value is not None:
            return value, label, "", make_excerpt(text, match)
    return None, "", "", ""


def extract_decision(text: str) -> tuple[str, float | None, str, str, str]:
    patterns = [
        ("hike", r"\b(raised|increased|lifted|tightened)\b.{0,90}?\bby\s+([0-9]+(?:\.[0-9]+)?)\s+(?:basis points|bps)\b", "explicit hike bps"),
        ("hike", r"\b(?:raise|increase)\s+(?:the\s+)?(?:policy rate|cash rate target|official cash rate|bank rate)\s+by\s+([0-9]+(?:\.[0-9]+)?)\s+(?:basis points|bps)\b", "explicit hike bps"),
        ("hike", r"\bdecided\s+to\s+(?:increase|raise)\b", "strong hike keyword"),
        ("cut", r"\b(cut|reduced|lowered|eased|decreased)\b.{0,90}?\bby\s+([0-9]+(?:\.[0-9]+)?)\s+(?:basis points|bps)\b", "explicit cut bps"),
        ("cut", r"\b(?:decrease|lower|reduce)\s+(?:the\s+)?(?:policy rate|cash rate target|official cash rate|bank rate)\s+by\s+([0-9]+(?:\.[0-9]+)?)\s+(?:basis points|bps)\b", "explicit cut bps"),
        ("cut", r"\bdecided\s+to\s+(?:lower|reduce)\b", "strong cut keyword"),
        ("hold", r"\b(left unchanged|kept unchanged|decided to maintain|decided to keep|no change to the policy rate|maintain the target range|target range\s+at\s+[0-9].{0,40}?to\s+[0-9])\b", "strong hold keyword"),
        ("hold", r"\b(kept|maintained)\s+(?:the\s+)?(?:policy rate|cash rate target|official cash rate|bank rate)\s+at\b", "strong hold keyword"),
        ("hold", r"\b(remains|remain)\s+at\b", "strong hold keyword"),
    ]
    for decision, pattern, label in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        bps = None
        for group in reversed(match.groups()):
            if group and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", str(group)):
                bps = float(group)
                break
        if decision == "cut" and bps is not None:
            bps = -bps
        if decision == "hold":
            bps = 0.0
        return decision, bps, label, pattern, make_excerpt(text, match)
    return "", None, "", "", ""


def confidence_for(decision: str, bps: float | None, rate_after: float | None, reason: str) -> float:
    explicit_bps = bps is not None and reason.startswith("explicit")
    strong_keyword = "strong" in reason
    if explicit_bps and rate_after is not None:
        return 0.95
    if decision == "hold" and rate_after is not None:
        return 0.90
    if explicit_bps:
        return 0.85
    if strong_keyword and decision:
        return 0.75
    return 0.0


def suggest_from_text(text: str) -> Suggestion:
    clean = normalize_text(text)
    decision, bps, reason, pattern, decision_excerpt = extract_decision(clean)
    rate_after, rate_pattern, rate_note, rate_excerpt = extract_policy_rate(clean)
    if not decision and rate_after is None:
        return Suggestion()
    conf = confidence_for(decision, bps, rate_after, reason)
    matched = decision_excerpt or rate_excerpt
    suggestion_reason = reason or ("policy rate found without decision" if rate_after is not None else "")
    if rate_note:
        suggestion_reason = f"{suggestion_reason}; {rate_note}".strip("; ")
    return Suggestion(
        decision=decision,
        rate_change_bps=bps,
        policy_rate_after=rate_after,
        confidence=conf,
        reason=suggestion_reason,
        matched_pattern=pattern or rate_pattern,
        matched_excerpt=matched,
    )


def build_text_lookup(meeting_documents: pd.DataFrame, master: pd.DataFrame, meeting_index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    md = ensure_columns(meeting_documents, ["meeting_id", "doc_id", "text_file", "title"])
    if not md.empty:
        rows.append(md[["meeting_id", "doc_id", "text_file", "title"]])
    m = ensure_columns(master, ["meeting_id", "doc_id", "text_file", "text_file_path", "title"])
    if not m.empty:
        m["text_file"] = m["text_file"].where(m["text_file"].astype(str).str.len() > 0, m["text_file_path"])
        rows.append(m[["meeting_id", "doc_id", "text_file", "title"]])
    mi = ensure_columns(meeting_index, ["meeting_id", "anchor_doc_id", "anchor_text_file", "anchor_title"])
    if not mi.empty:
        rows.append(pd.DataFrame({
            "meeting_id": mi["meeting_id"],
            "doc_id": mi["anchor_doc_id"],
            "text_file": mi["anchor_text_file"],
            "title": mi["anchor_title"],
        }))
    if not rows:
        return pd.DataFrame(columns=["meeting_id", "doc_id", "text_file", "title"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["meeting_id", "doc_id", "text_file"])


def resolve_text_file(row: pd.Series, lookup: pd.DataFrame, base_dir: Path) -> str:
    candidates = lookup[
        (lookup["doc_id"].astype(str).eq(str(row.get("source_doc_id", ""))))
        | (lookup["meeting_id"].astype(str).eq(str(row.get("meeting_id", ""))))
    ]
    for _, candidate in candidates.iterrows():
        path_text = str(candidate.get("text_file", ""))
        path = Path(path_text)
        if not path.is_absolute():
            path = base_dir / path
        if path.exists():
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return ""


def create_suggestions(policy: pd.DataFrame, qa: pd.DataFrame, lookup: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    qa_lookup = ensure_columns(qa, ["meeting_id", "text_excerpt"])[["meeting_id", "text_excerpt"]]
    df = policy.merge(qa_lookup, on="meeting_id", how="left", suffixes=("", "_qa"))
    suggestions = []
    for _, row in df.iterrows():
        text = str(row.get("text_excerpt", ""))
        suggestion = suggest_from_text(text)
        if suggestion.confidence < 0.75:
            full_text = resolve_text_file(row, lookup, base_dir)
            if full_text:
                full_suggestion = suggest_from_text(full_text)
                if full_suggestion.confidence > suggestion.confidence:
                    suggestion = full_suggestion
        suggestions.append(suggestion)
    out = policy.copy()
    out["suggested_decision"] = [s.decision for s in suggestions]
    out["suggested_rate_change_bps"] = ["" if s.rate_change_bps is None else s.rate_change_bps for s in suggestions]
    out["suggested_policy_rate_after"] = ["" if s.policy_rate_after is None else s.policy_rate_after for s in suggestions]
    out["suggested_confidence"] = [s.confidence for s in suggestions]
    out["suggestion_reason"] = [s.reason for s in suggestions]
    out["matched_pattern"] = [s.matched_pattern for s in suggestions]
    out["matched_excerpt"] = [s.matched_excerpt for s in suggestions]
    return out


def add_review_flags(policy_with_suggestions: pd.DataFrame, qa: pd.DataFrame, confidence_threshold: float) -> pd.DataFrame:
    qa_small = ensure_columns(qa, ["meeting_id", "text_excerpt"])[["meeting_id", "text_excerpt"]]
    df = policy_with_suggestions.merge(qa_small, on="meeting_id", how="left", suffixes=("", "_qa"))
    reasons = [review_reasons(row, confidence_threshold) for _, row in df.iterrows()]
    df["needs_review"] = [bool(r) for r in reasons]
    df["review_reason"] = ["; ".join(r) for r in reasons]
    df["qc_priority"] = [priority_for(r) if r else "low" for r in reasons]
    return df


def build_review(df: pd.DataFrame) -> pd.DataFrame:
    review = pd.DataFrame({
        "bank": df["bank"],
        "meeting_date": df["meeting_date"],
        "meeting_id": df["meeting_id"],
        "current_decision": df["decision"],
        "current_rate_change_bps": df["rate_change_bps"],
        "current_policy_rate_after": df["policy_rate_after"],
        "current_confidence": df["confidence"],
        "extraction_method": df["extraction_method"],
        "needs_review": df["needs_review"],
        "review_reason": df["review_reason"],
        "qc_priority": df["qc_priority"],
        "suggested_decision": df["suggested_decision"],
        "suggested_rate_change_bps": df["suggested_rate_change_bps"],
        "suggested_policy_rate_after": df["suggested_policy_rate_after"],
        "suggested_confidence": df["suggested_confidence"],
        "suggestion_reason": df["suggestion_reason"],
        "matched_pattern": df["matched_pattern"],
        "matched_excerpt": df["matched_excerpt"],
        "source_title": df["source_title"],
        "source_url": df["source_url"],
        "notes": df["notes"],
    })
    priority_order = {"high": 0, "medium": 1, "low": 2}
    review["_priority"] = review["qc_priority"].map(priority_order).fillna(9)
    review["current_confidence_num"] = pd.to_numeric(review["current_confidence"], errors="coerce")
    return review.sort_values(["_priority", "current_confidence_num", "bank", "meeting_date"]).drop(columns=["_priority", "current_confidence_num"])[REVIEW_COLUMNS]


def build_suggested_corrections(review: pd.DataFrame, suggestion_threshold: float) -> pd.DataFrame:
    df = review.copy()
    df["suggested_confidence_num"] = pd.to_numeric(df["suggested_confidence"], errors="coerce").fillna(0)
    df = df[(df["needs_review"].astype(bool)) & (df["suggested_confidence_num"] >= suggestion_threshold)].copy()
    return pd.DataFrame({
        "bank": df["bank"],
        "meeting_date": df["meeting_date"],
        "meeting_id": df["meeting_id"],
        "decision": df["suggested_decision"],
        "rate_change_bps": df["suggested_rate_change_bps"],
        "policy_rate_after": df["suggested_policy_rate_after"],
        "confidence": df["suggested_confidence"],
        "override_reason": df["suggestion_reason"],
        "matched_excerpt": df["matched_excerpt"],
    })[SUGGESTED_COLUMNS]


def build_override_template(corrections: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    if corrections.empty:
        base = review[review["needs_review"].astype(bool)].copy()
        return pd.DataFrame({
            "bank": base["bank"],
            "meeting_date": base["meeting_date"],
            "meeting_id": base["meeting_id"],
            "override_decision": "",
            "override_rate_change_bps": "",
            "override_policy_rate_after": "",
            "override_confidence": "",
            "override_notes": "",
            "approved": "false",
        })[OVERRIDE_TEMPLATE_COLUMNS]
    return pd.DataFrame({
        "bank": corrections["bank"],
        "meeting_date": corrections["meeting_date"],
        "meeting_id": corrections["meeting_id"],
        "override_decision": corrections["decision"],
        "override_rate_change_bps": corrections["rate_change_bps"],
        "override_policy_rate_after": corrections["policy_rate_after"],
        "override_confidence": corrections["confidence"],
        "override_notes": corrections["override_reason"],
        "approved": "false",
    })[OVERRIDE_TEMPLATE_COLUMNS]


def apply_overrides(final: pd.DataFrame, review: pd.DataFrame, overrides: pd.DataFrame, overrides_found: bool, suggestion_threshold: float) -> tuple[pd.DataFrame, int]:
    df = final.copy()
    df = df.merge(
        review[["meeting_id", "needs_review", "review_reason", "suggested_decision", "suggested_rate_change_bps", "suggested_policy_rate_after", "suggested_confidence"]],
        on="meeting_id",
        how="left",
    )
    df["qc_status"] = "original_extraction"
    df["override_notes"] = ""
    high_suggestion = pd.to_numeric(df["suggested_confidence"], errors="coerce").fillna(0) >= suggestion_threshold
    df.loc[high_suggestion, "qc_status"] = "suggestion_available_not_applied"
    applied = 0
    if overrides_found and not overrides.empty:
        overrides = ensure_columns(overrides, OVERRIDE_TEMPLATE_COLUMNS)
        approved = overrides[overrides["approved"].apply(truthy)].copy()
        for _, row in approved.iterrows():
            mask = df["meeting_id"].astype(str).eq(str(row.get("meeting_id", "")).strip())
            if not mask.any() and str(row.get("bank", "")).strip() and str(row.get("meeting_date", "")).strip():
                mask = df["bank"].astype(str).eq(str(row["bank"]).strip()) & df["meeting_date"].astype(str).eq(str(row["meeting_date"]).strip())
            if not mask.any():
                continue
            idxs = df.index[mask]
            for idx in idxs:
                if str(row.get("override_decision", "")).strip():
                    df.at[idx, "decision"] = row["override_decision"]
                if str(row.get("override_rate_change_bps", "")).strip():
                    df.at[idx, "rate_change_bps"] = row["override_rate_change_bps"]
                if str(row.get("override_policy_rate_after", "")).strip():
                    df.at[idx, "policy_rate_after"] = row["override_policy_rate_after"]
                if str(row.get("override_confidence", "")).strip():
                    df.at[idx, "confidence"] = row["override_confidence"]
                notes = str(row.get("override_notes", "")).strip()
                if notes:
                    existing = str(df.at[idx, "notes"]).strip()
                    df.at[idx, "notes"] = notes if not existing else f"{existing} {notes}"
                df.at[idx, "qc_status"] = "manual_override_applied"
                df.at[idx, "override_notes"] = notes
                applied += 1
    df = recalc_next(df)
    return df, applied


def recalc_next(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_date"] = pd.to_datetime(out["meeting_date"], errors="coerce")
    out = out.sort_values(["bank", "_date", "meeting_id"])
    out["next_decision"] = out.groupby("bank")["decision"].shift(-1).fillna("")
    out["next_rate_change_bps"] = out.groupby("bank")["rate_change_bps"].shift(-1).fillna("")
    cols = [c for c in out.columns if c != "_date"]
    return out[cols]


def build_summary(raw: pd.DataFrame, final: pd.DataFrame, review: pd.DataFrame, corrections: pd.DataFrame, approved_applied: int, suggestion_threshold: float) -> pd.DataFrame:
    high_suggestions = int((pd.to_numeric(corrections.get("confidence", pd.Series(dtype=float)), errors="coerce").fillna(0) >= suggestion_threshold).sum())
    rows = [
        ("total_policy_rows", len(raw)),
        ("rows_needing_review", int(review["needs_review"].astype(bool).sum())),
        ("high_priority_review_rows", int(review["qc_priority"].eq("high").sum())),
        ("medium_priority_review_rows", int(review["qc_priority"].eq("medium").sum())),
        ("unknown_decisions_before", int(raw["decision"].eq("unknown").sum())),
        ("unknown_decisions_after", int(final["decision"].eq("unknown").sum())),
        ("approved_overrides_applied", approved_applied),
        ("suggestions_created", len(corrections)),
        ("high_confidence_suggestions", high_suggestions),
        ("average_confidence_before", pd.to_numeric(raw["confidence"], errors="coerce").mean()),
        ("average_confidence_after", pd.to_numeric(final["confidence"], errors="coerce").mean()),
        ("final_hikes", int(final["decision"].eq("hike").sum())),
        ("final_holds", int(final["decision"].eq("hold").sum())),
        ("final_cuts", int(final["decision"].eq("cut").sum())),
        ("final_unknowns", int(final["decision"].eq("unknown").sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def write_readme(path: Path) -> None:
    text = """# Policy Decisions QC Workflow

This workflow reviews `policy_decisions.csv` without modifying it. The raw extraction remains unchanged. The backtest-ready output is `policy_decisions_final.csv`.

## Files

- `policy_decisions.csv`: raw automated extraction from scraped central bank documents.
- `policy_decisions_review.csv`: rows flagged for review, with reasons and suggested corrections.
- `policy_decisions_suggested_corrections.csv`: high-confidence automatic suggestions.
- `policy_decisions_overrides_template.csv`: editable override template. Suggestions are prefilled where available and `approved` is `false`.
- `policy_decisions_overrides.csv`: optional manual file you create by copying/editing the template.
- `policy_decisions_final.csv`: raw extraction plus approved overrides and QC metadata.

## Review Process

Open `policy_decisions_review.csv` and start with `qc_priority = high`. Check the current decision, the suggestion, and the matched excerpt. Unknown or inconsistent rows should be reviewed first.

## Overrides

Copy `policy_decisions_overrides_template.csv` to `policy_decisions_overrides.csv`, edit any rows you want to approve, and set `approved` to `true`. Approved rows replace decision, rate change, policy rate after, confidence, and notes in the final file.

## Rerun QC

```powershell
python policy_decisions_qc.py --input data/external/policy_decisions.csv --qa data/external/policy_decisions_qa.csv --meeting-linkage processed/meeting_linkage --output-dir data/external
```

## Rerun Phase 4 Backtest

```powershell
python hawkometer_backtest.py --hawkometer processed/hawkometer --analytics processed/hawkometer_analytics --meeting-linkage processed/meeting_linkage --output processed/hawkometer_backtest --policy-decisions data/external/policy_decisions_final.csv
```

Suggestions are aids, not truth. Review and approve corrections manually before treating them as validated policy decisions.
"""
    path.write_text(text, encoding="utf-8")


def run(
    input_path: Path,
    qa_path: Path,
    meeting_linkage: Path,
    output_dir: Path,
    overrides_path: Path,
    confidence_threshold: float,
    suggestion_threshold: float,
    start_date: str,
    end_date: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy, policy_found = read_csv_safe(input_path)
    qa, _ = read_csv_safe(qa_path)
    overrides, overrides_found = read_csv_safe(overrides_path, optional=True)
    meeting_documents, _ = read_csv_safe(meeting_linkage / "meeting_documents.csv", optional=True)
    master, _ = read_csv_safe(meeting_linkage / "updated_master_index.csv", optional=True)
    meeting_index, _ = read_csv_safe(meeting_linkage / "meeting_index.csv", optional=True)

    if not policy_found or policy.empty:
        raise FileNotFoundError(f"Cannot run QC without policy decisions input: {input_path}")

    policy = ensure_columns(policy, ["bank", "meeting_date", "meeting_id", "decision", "rate_change_bps", "policy_rate_after", "next_decision", "next_rate_change_bps", "source_doc_id", "source_title", "source_url", "confidence", "extraction_method", "notes"])
    qa = ensure_columns(qa, ["meeting_id", "text_excerpt"])
    policy = filter_dates(policy, start_date, end_date)
    lookup = build_text_lookup(meeting_documents, master, meeting_index)
    suggested = create_suggestions(policy, qa, lookup, Path.cwd())
    flagged = add_review_flags(suggested, qa, confidence_threshold)
    review = build_review(flagged)
    corrections = build_suggested_corrections(review, suggestion_threshold)
    template = build_override_template(corrections, review)
    final, approved_applied = apply_overrides(policy, review, overrides, overrides_found, suggestion_threshold)
    summary = build_summary(policy, final, review, corrections, approved_applied, suggestion_threshold)

    review.to_csv(output_dir / "policy_decisions_review.csv", index=False)
    corrections.to_csv(output_dir / "policy_decisions_suggested_corrections.csv", index=False)
    template.to_csv(output_dir / "policy_decisions_overrides_template.csv", index=False)
    final.to_csv(output_dir / "policy_decisions_final.csv", index=False)
    summary.to_csv(output_dir / "policy_decisions_qc_summary.csv", index=False)
    write_readme(output_dir / "README_policy_decisions_qc.md")

    print(f"rows loaded: {len(policy)}")
    print(f"rows needing review: {int(review['needs_review'].astype(bool).sum())}")
    print(f"suggestions created: {len(corrections)}")
    print(f"high-confidence suggestions: {int((pd.to_numeric(corrections.get('confidence', pd.Series(dtype=float)), errors='coerce').fillna(0) >= suggestion_threshold).sum())}")
    print(f"overrides file found or missing: {'found' if overrides_found else 'missing'}")
    print(f"approved overrides applied: {approved_applied}")
    print(f"final unknown decisions: {int(final['decision'].eq('unknown').sum())}")
    print("output files created:")
    for name in [
        "policy_decisions_review.csv",
        "policy_decisions_suggested_corrections.csv",
        "policy_decisions_overrides_template.csv",
        "policy_decisions_final.csv",
        "policy_decisions_qc_summary.csv",
        "README_policy_decisions_qc.md",
    ]:
        print(f"  {output_dir / name}")


def main() -> None:
    args = parse_args()
    run(
        input_path=Path(args.input),
        qa_path=Path(args.qa),
        meeting_linkage=Path(args.meeting_linkage),
        output_dir=Path(args.output_dir),
        overrides_path=Path(args.overrides),
        confidence_threshold=args.confidence_threshold,
        suggestion_threshold=args.suggestion_threshold,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()
