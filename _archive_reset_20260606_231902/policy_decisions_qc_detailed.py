from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DETAILED_COLUMNS = [
    "bank",
    "meeting_date",
    "meeting_id",
    "qc_priority",
    "needs_review",
    "review_reason",
    "current_decision",
    "current_rate_change_bps",
    "current_policy_rate_after",
    "current_confidence",
    "extraction_method",
    "suggested_decision",
    "suggested_rate_change_bps",
    "suggested_policy_rate_after",
    "suggested_confidence",
    "suggestion_reason",
    "matched_pattern",
    "matched_excerpt",
    "full_context",
    "context_status",
    "source_doc_id",
    "source_title",
    "source_url",
    "text_file",
    "approved",
    "final_decision",
    "final_rate_change_bps",
    "final_policy_rate_after",
    "final_confidence",
    "reviewer_notes",
]

OVERRIDE_COLUMNS = [
    "meeting_id",
    "override_decision",
    "override_rate_change_bps",
    "override_policy_rate_after",
    "override_confidence",
    "override_notes",
    "approved",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/apply a self-contained policy decisions review spreadsheet.")
    parser.add_argument("--mode", choices=["build-review", "apply-review"], default="build-review")
    parser.add_argument("--input", default="data/external/policy_decisions.csv")
    parser.add_argument("--qa", default="data/external/policy_decisions_qa.csv")
    parser.add_argument("--meeting-linkage", default="processed/meeting_linkage")
    parser.add_argument("--output-dir", default="data/external")
    parser.add_argument("--review", default="data/external/policy_decisions_review_detailed.csv")
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-12-31")
    return parser.parse_args()


def read_csv(path: Path, optional: bool = False) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        print(f"Warning: {'optional ' if optional else ''}file missing: {path}")
        return pd.DataFrame(), False
    return pd.read_csv(path, dtype=str, keep_default_na=False), True


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def num(value: Any) -> float:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "approved"}


def filter_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_date"] = pd.to_datetime(df.get("meeting_date", ""), errors="coerce")
    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")
    df = df[df["_date"].between(start, end, inclusive="both")].copy()
    df["meeting_date"] = df["_date"].dt.strftime("%Y-%m-%d")
    return df.drop(columns=["_date"])


def inconsistent(decision: str, rate_change: Any) -> bool:
    decision = str(decision or "").strip().lower()
    rate = num(rate_change)
    if pd.isna(rate) or decision == "unknown":
        return False
    return (
        (decision == "hike" and rate <= 0)
        or (decision == "cut" and rate >= 0)
        or (decision == "hold" and rate != 0)
    )


def review_reasons(row: pd.Series, confidence_threshold: float) -> list[str]:
    reasons = []
    confidence = num(row.get("confidence", ""))
    if pd.isna(confidence) or confidence < confidence_threshold:
        reasons.append(f"confidence < {confidence_threshold:g}")
    if str(row.get("decision", "")).strip().lower() in {"", "unknown"}:
        reasons.append("decision unknown")
    if str(row.get("rate_change_bps", "")).strip() == "":
        reasons.append("rate_change_bps missing")
    if str(row.get("policy_rate_after", "")).strip() == "":
        reasons.append("policy_rate_after missing")
    if "keyword" in str(row.get("extraction_method", "")).lower():
        reasons.append("keyword extraction")
    if str(row.get("source_title", "")).strip() == "":
        reasons.append("source_title missing")
    if str(row.get("text_excerpt", "")).strip() == "":
        reasons.append("text_excerpt missing")
    if inconsistent(row.get("decision", ""), row.get("rate_change_bps", "")):
        reasons.append("decision inconsistent with rate_change_bps")
    return reasons


def priority(reasons: list[str]) -> str:
    if "decision unknown" in reasons or "decision inconsistent with rate_change_bps" in reasons:
        return "high"
    if any(r.startswith("confidence") or "missing" in r for r in reasons):
        return "medium"
    return "low"


def clean_text(text: str) -> str:
    text = str(text or "")
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "â€“": "-",
        "â€”": "-",
        "â€™": "'",
        "Â½": "1/2",
        "Â¼": "1/4",
        "Â¾": "3/4",
        "\xa0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def parse_percent(value: str) -> float | None:
    text = str(value).lower().replace("%", "").replace("percent", "").strip()
    text = text.replace("¼", "/4").replace("½", "/2").replace("¾", "/4")
    text = text.replace("1/4", ".25").replace("1/2", ".5").replace("3/4", ".75").replace("-", " ")
    parts = text.split()
    try:
        if len(parts) == 2 and parts[1].startswith("."):
            return float(parts[0]) + float(parts[1])
        return float(text)
    except ValueError:
        return None


def context_around(text: str, match: re.Match[str] | None, radius: int, fallback: int) -> str:
    if not text:
        return ""
    if not match:
        return text[:fallback].strip()
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return text[start:end].strip()


def extract_policy_rate(text: str) -> tuple[float | None, str, str, re.Match[str] | None]:
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
            return (low + high) / 2, label, f"Fed target range {low:g} to {high:g}; midpoint used.", match
        rate = parse_percent(match.groups()[-1])
        if rate is not None:
            return rate, label, "", match
    return None, "", "", None


def extract_decision(text: str) -> tuple[str, float | None, str, str, re.Match[str] | None]:
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
    for decision, pattern, reason in patterns:
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
        return decision, bps, reason, pattern, match
    return "", None, "", "", None


def suggestion_confidence(decision: str, bps: float | None, rate_after: float | None, reason: str) -> float:
    explicit = reason.startswith("explicit")
    strong = reason.startswith("strong")
    if explicit and rate_after is not None:
        return 0.95
    if decision == "hold" and rate_after is not None:
        return 0.90
    if explicit:
        return 0.85
    if strong:
        return 0.75
    return 0.0


def run_strong_extraction(text: str) -> dict[str, Any]:
    text = clean_text(text)
    decision, bps, reason, pattern, decision_match = extract_decision(text)
    rate, rate_pattern, rate_note, rate_match = extract_policy_rate(text)
    match = decision_match or rate_match
    if not decision and rate is None:
        return {
            "suggested_decision": "",
            "suggested_rate_change_bps": "",
            "suggested_policy_rate_after": "",
            "suggested_confidence": 0.0,
            "suggestion_reason": "",
            "matched_pattern": "",
            "matched_excerpt": "",
            "full_context": context_around(text, None, 1000, 2500),
        }
    conf = suggestion_confidence(decision, bps, rate, reason)
    note = reason
    if rate_note:
        note = f"{note}; {rate_note}".strip("; ")
    return {
        "suggested_decision": decision,
        "suggested_rate_change_bps": "" if bps is None else bps,
        "suggested_policy_rate_after": "" if rate is None else rate,
        "suggested_confidence": conf,
        "suggestion_reason": note,
        "matched_pattern": pattern or rate_pattern,
        "matched_excerpt": context_around(text, match, 250, 500),
        "full_context": context_around(text, match, 1000, 2500),
    }


def build_lookup(meeting_index: pd.DataFrame, meeting_documents: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mi = ensure_columns(meeting_index, ["meeting_id", "anchor_doc_id", "anchor_title", "anchor_text_file"])
    if not mi.empty:
        rows.append(pd.DataFrame({
            "meeting_id": mi["meeting_id"],
            "doc_id": mi["anchor_doc_id"],
            "title": mi["anchor_title"],
            "text_file": mi["anchor_text_file"],
            "priority": 1,
        }))
    md = ensure_columns(meeting_documents, ["meeting_id", "doc_id", "document_subtype", "category", "title", "text_file"])
    if not md.empty:
        md = md.copy()
        md["priority"] = md.apply(lambda r: 2 if str(r["doc_id"]) else 5, axis=1)
        rows.append(md[["meeting_id", "doc_id", "title", "text_file", "priority"]])
    m = ensure_columns(master, ["meeting_id", "doc_id", "document_subtype", "category", "title", "text_file", "text_file_path"])
    if not m.empty:
        m = m.copy()
        m["text_file"] = m["text_file"].where(m["text_file"].astype(str).str.len() > 0, m["text_file_path"])
        m["priority"] = 4
        rows.append(m[["meeting_id", "doc_id", "title", "text_file", "priority"]])
    if not rows:
        return pd.DataFrame(columns=["meeting_id", "doc_id", "title", "text_file", "priority"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["meeting_id", "doc_id", "text_file"]).sort_values(["meeting_id", "priority"])


def read_source_text(row: pd.Series, lookup: pd.DataFrame, base: Path) -> tuple[str, str, str]:
    source_doc_id = str(row.get("source_doc_id", "")).strip()
    meeting_id = str(row.get("meeting_id", "")).strip()
    choices = pd.DataFrame()
    if source_doc_id:
        choices = lookup[lookup["doc_id"].astype(str).eq(source_doc_id)]
    if choices.empty:
        choices = lookup[lookup["meeting_id"].astype(str).eq(meeting_id)]
    for _, item in choices.iterrows():
        path_text = str(item.get("text_file", ""))
        if not path_text:
            continue
        path = Path(path_text)
        if not path.is_absolute():
            path = base / path
        if not path.exists():
            continue
        try:
            return path.read_text(encoding="utf-8", errors="replace"), str(path_text), "full_context_loaded"
        except OSError as exc:
            return "", path_text, f"context_read_error: {exc}"
    return "", "", "context_missing_text_file"


def default_final(suggested: Any, current: Any) -> Any:
    return suggested if str(suggested).strip() not in {"", "nan", "None"} else current


def build_review(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy, found = read_csv(Path(args.input))
    if not found or policy.empty:
        raise FileNotFoundError(args.input)
    qa, _ = read_csv(Path(args.qa), optional=True)
    meeting_index, _ = read_csv(Path(args.meeting_linkage) / "meeting_index.csv", optional=True)
    meeting_documents, _ = read_csv(Path(args.meeting_linkage) / "meeting_documents.csv", optional=True)
    master, _ = read_csv(Path(args.meeting_linkage) / "updated_master_index.csv", optional=True)

    policy = ensure_columns(policy, ["bank", "meeting_date", "meeting_id", "decision", "rate_change_bps", "policy_rate_after", "source_doc_id", "source_title", "source_url", "confidence", "extraction_method", "notes"])
    policy = filter_dates(policy, args.start_date, args.end_date)
    qa = ensure_columns(qa, ["meeting_id", "text_excerpt", "source_doc_id"])
    merged = policy.merge(qa[["meeting_id", "text_excerpt"]], on="meeting_id", how="left")
    reasons = [review_reasons(row, args.confidence_threshold) for _, row in merged.iterrows()]
    merged["needs_review"] = [bool(r) for r in reasons]
    merged["review_reason"] = ["; ".join(r) for r in reasons]
    merged["qc_priority"] = [priority(r) if r else "low" for r in reasons]
    review_base = merged[merged["needs_review"]].copy()

    lookup = build_lookup(meeting_index, meeting_documents, master)
    rows = []
    for _, row in review_base.iterrows():
        text, text_file, context_status = read_source_text(row, lookup, Path.cwd())
        extraction = run_strong_extraction(text if text else row.get("text_excerpt", ""))
        if not text and row.get("text_excerpt", ""):
            context_status = "qa_excerpt_only"
        rows.append({
            "bank": row["bank"],
            "meeting_date": row["meeting_date"],
            "meeting_id": row["meeting_id"],
            "qc_priority": row["qc_priority"],
            "needs_review": True,
            "review_reason": row["review_reason"],
            "current_decision": row["decision"],
            "current_rate_change_bps": row["rate_change_bps"],
            "current_policy_rate_after": row["policy_rate_after"],
            "current_confidence": row["confidence"],
            "extraction_method": row["extraction_method"],
            **extraction,
            "context_status": context_status,
            "source_doc_id": row["source_doc_id"],
            "source_title": row["source_title"],
            "source_url": row["source_url"],
            "text_file": text_file,
            "approved": "false",
            "final_decision": default_final(extraction["suggested_decision"], row["decision"]),
            "final_rate_change_bps": default_final(extraction["suggested_rate_change_bps"], row["rate_change_bps"]),
            "final_policy_rate_after": default_final(extraction["suggested_policy_rate_after"], row["policy_rate_after"]),
            "final_confidence": default_final(extraction["suggested_confidence"] if extraction["suggested_confidence"] else "", row["confidence"]),
            "reviewer_notes": "",
        })
    detailed = pd.DataFrame(rows, columns=DETAILED_COLUMNS)
    pr_order = {"high": 0, "medium": 1, "low": 2}
    detailed["_prio"] = detailed["qc_priority"].map(pr_order).fillna(9)
    detailed["_conf"] = pd.to_numeric(detailed["current_confidence"], errors="coerce")
    detailed = detailed.sort_values(["_prio", "_conf", "bank", "meeting_date"]).drop(columns=["_prio", "_conf"])

    csv_path = out / "policy_decisions_review_detailed.csv"
    detailed.to_csv(csv_path, index=False)
    write_excel(out / "policy_decisions_review_detailed.xlsx", detailed)
    overrides_template = pd.DataFrame({
        "meeting_id": detailed["meeting_id"],
        "override_decision": detailed["final_decision"],
        "override_rate_change_bps": detailed["final_rate_change_bps"],
        "override_policy_rate_after": detailed["final_policy_rate_after"],
        "override_confidence": detailed["final_confidence"],
        "override_notes": detailed["reviewer_notes"],
        "approved": "false",
    })[OVERRIDE_COLUMNS]
    overrides_template.to_csv(out / "policy_decisions_overrides_from_detailed_template.csv", index=False)
    write_readme(out / "README_policy_decisions_review_detailed.md")

    summary = make_summary(policy, detailed, approved=0)
    summary.to_csv(out / "policy_decisions_review_detailed_summary.csv", index=False)
    print(f"total rows loaded: {len(policy)}")
    print(f"rows needing review: {len(detailed)}")
    print(f"high priority rows: {int(detailed['qc_priority'].eq('high').sum())}")
    print(f"suggestions created: {int(pd.to_numeric(detailed['suggested_confidence'], errors='coerce').fillna(0).gt(0).sum())}")
    print(f"rows with full context: {int(detailed['context_status'].eq('full_context_loaded').sum())}")
    print(f"rows missing context: {int(detailed['context_status'].str.contains('missing|error', case=False, na=False).sum())}")
    print(f"review CSV path: {csv_path}")
    xlsx_path = out / "policy_decisions_review_detailed.xlsx"
    print(f"review XLSX path: {xlsx_path if xlsx_path.exists() else 'not created (openpyxl unavailable)'}")


def write_excel(path: Path, detailed: pd.DataFrame) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("Warning: openpyxl not installed; skipping XLSX output.")
        return
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        detailed.to_excel(writer, index=False, sheet_name="Review")
        instructions = pd.DataFrame({
            "Instructions": [
                "Filter qc_priority = high first.",
                "Read full_context and matched_excerpt.",
                "Edit final_decision, final_rate_change_bps, final_policy_rate_after, and final_confidence if needed.",
                "Set approved = TRUE only when you are happy with the final fields.",
                "Run apply-review mode on the saved CSV or XLSX to create policy_decisions_overrides.csv and policy_decisions_final.csv.",
            ]
        })
        instructions.to_excel(writer, index=False, sheet_name="Instructions")
        ws = writer.book["Review"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        widths = {
            "A": 12, "B": 14, "C": 20, "D": 12, "F": 35, "L": 18,
            "Q": 35, "R": 60, "S": 90, "V": 45, "X": 45,
            "Y": 10, "Z": 16, "AD": 35,
        }
        for col, width in widths.items():
            ws.column_dimensions[col].width = width
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if cell.column_letter in {"R", "S", "F", "AD"}:
                    cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")


def read_review(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, sheet_name="Review", dtype=str).fillna("")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def apply_review(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy, found = read_csv(Path(args.input))
    if not found or policy.empty:
        raise FileNotFoundError(args.input)
    review = read_review(Path(args.review))
    review = ensure_columns(review, DETAILED_COLUMNS)
    approved = review[review["approved"].apply(truthy)].copy()
    overrides = pd.DataFrame({
        "meeting_id": approved["meeting_id"],
        "override_decision": approved["final_decision"],
        "override_rate_change_bps": approved["final_rate_change_bps"],
        "override_policy_rate_after": approved["final_policy_rate_after"],
        "override_confidence": approved["final_confidence"],
        "override_notes": approved["reviewer_notes"],
        "approved": "true",
    })[OVERRIDE_COLUMNS]
    overrides.to_csv(out / "policy_decisions_overrides.csv", index=False)

    final = policy.copy()
    final["qc_status"] = "original_extraction"
    for _, row in overrides.iterrows():
        mask = final["meeting_id"].astype(str).eq(str(row["meeting_id"]))
        for idx in final.index[mask]:
            final.at[idx, "decision"] = row["override_decision"]
            final.at[idx, "rate_change_bps"] = row["override_rate_change_bps"]
            final.at[idx, "policy_rate_after"] = row["override_policy_rate_after"]
            final.at[idx, "confidence"] = row["override_confidence"]
            notes = str(row["override_notes"]).strip()
            if notes:
                existing = str(final.at[idx, "notes"]).strip()
                final.at[idx, "notes"] = notes if not existing else f"{existing} {notes}"
            final.at[idx, "qc_status"] = "manual_override_applied"
    final = recalc_next(final)
    final.to_csv(out / "policy_decisions_final.csv", index=False)
    summary = make_summary(policy, review, approved=len(overrides), final=final)
    summary.to_csv(out / "policy_decisions_review_detailed_summary.csv", index=False)
    print(f"approved rows found: {len(overrides)}")
    print(f"overrides file created: {out / 'policy_decisions_overrides.csv'}")
    print(f"final policy decisions file created: {out / 'policy_decisions_final.csv'}")
    print(f"unknown decisions before: {int(policy['decision'].eq('unknown').sum())}")
    print(f"unknown decisions after: {int(final['decision'].eq('unknown').sum())}")
    print(f"average confidence before: {pd.to_numeric(policy['confidence'], errors='coerce').mean():.3f}")
    print(f"average confidence after: {pd.to_numeric(final['confidence'], errors='coerce').mean():.3f}")


def recalc_next(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_date"] = pd.to_datetime(out["meeting_date"], errors="coerce")
    out = out.sort_values(["bank", "_date", "meeting_id"])
    out["next_decision"] = out.groupby("bank")["decision"].shift(-1).fillna("")
    out["next_rate_change_bps"] = out.groupby("bank")["rate_change_bps"].shift(-1).fillna("")
    return out.drop(columns=["_date"])


def make_summary(policy: pd.DataFrame, detailed: pd.DataFrame, approved: int, final: pd.DataFrame | None = None) -> pd.DataFrame:
    final_df = final if final is not None else policy
    rows = [
        ("total_policy_rows", len(policy)),
        ("rows_needing_review", len(detailed)),
        ("high_priority_rows", int(detailed.get("qc_priority", pd.Series(dtype=str)).eq("high").sum())),
        ("medium_priority_rows", int(detailed.get("qc_priority", pd.Series(dtype=str)).eq("medium").sum())),
        ("low_priority_rows", int(detailed.get("qc_priority", pd.Series(dtype=str)).eq("low").sum())),
        ("suggestions_created", int(pd.to_numeric(detailed.get("suggested_confidence", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0).sum())),
        ("rows_with_full_context", int(detailed.get("context_status", pd.Series(dtype=str)).eq("full_context_loaded").sum())),
        ("rows_missing_context", int(detailed.get("context_status", pd.Series(dtype=str)).str.contains("missing|error", case=False, na=False).sum())),
        ("approved_rows_applied", approved),
        ("unknown_decisions_before", int(policy["decision"].eq("unknown").sum())),
        ("unknown_decisions_after", int(final_df["decision"].eq("unknown").sum())),
        ("average_confidence_before", pd.to_numeric(policy["confidence"], errors="coerce").mean()),
        ("average_confidence_after", pd.to_numeric(final_df["confidence"], errors="coerce").mean()),
        ("output_created_at", datetime.now(timezone.utc).isoformat()),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def write_readme(path: Path) -> None:
    text = """# Detailed Policy Decisions Review

`policy_decisions_review_detailed.csv` and `.xlsx` are self-contained review files for policy decision QC. They include the current extraction, suggested correction, matched phrase, source metadata, and enough context to review decisions without opening source documents.

## How To Review

Open the Excel file and filter `qc_priority = high` first. Read `full_context`, compare current and suggested fields, then edit:

- `final_decision`
- `final_rate_change_bps`
- `final_policy_rate_after`
- `final_confidence`
- `reviewer_notes`

Set `approved` to `TRUE` only for rows you want applied.

## Apply Approved Rows

CSV:

```powershell
python policy_decisions_qc_detailed.py --mode apply-review --input data/external/policy_decisions.csv --review data/external/policy_decisions_review_detailed.csv --output-dir data/external
```

Excel:

```powershell
python policy_decisions_qc_detailed.py --mode apply-review --input data/external/policy_decisions.csv --review data/external/policy_decisions_review_detailed.xlsx --output-dir data/external
```

This creates `policy_decisions_overrides.csv` and `policy_decisions_final.csv`.

## Rerun Phase 4 Backtest

```powershell
python hawkometer_backtest.py --hawkometer processed/hawkometer --analytics processed/hawkometer_analytics --meeting-linkage processed/meeting_linkage --output processed/hawkometer_backtest --policy-decisions data/external/policy_decisions_final.csv
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.mode == "build-review":
        build_review(args)
    else:
        apply_review(args)


if __name__ == "__main__":
    main()
