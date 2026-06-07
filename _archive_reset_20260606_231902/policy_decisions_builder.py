from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "bank",
    "meeting_date",
    "meeting_id",
    "decision",
    "rate_change_bps",
    "policy_rate_after",
    "next_decision",
    "next_rate_change_bps",
    "source_doc_id",
    "source_title",
    "source_url",
    "confidence",
    "extraction_method",
    "notes",
]

QA_COLUMNS = [
    "bank",
    "meeting_date",
    "meeting_id",
    "decision",
    "rate_change_bps",
    "policy_rate_after",
    "confidence",
    "extraction_method",
    "source_title",
    "notes",
    "text_excerpt",
]


@dataclass
class ExtractionResult:
    decision: str = "unknown"
    rate_change_bps: float | None = None
    policy_rate_after: float | None = None
    confidence: float = 0.0
    extraction_method: str = "unknown"
    notes: str = ""
    text_excerpt: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build policy_decisions.csv from scraped central bank meeting documents.")
    parser.add_argument("--meeting-linkage", default="processed/meeting_linkage", help="Meeting linkage folder.")
    parser.add_argument("--output", default="data/external/policy_decisions.csv", help="Output policy decisions CSV.")
    parser.add_argument("--start-date", default="2021-01-01", help="Start date filter.")
    parser.add_argument("--end-date", default="2026-12-31", help="End date filter.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence rows to write.")
    parser.add_argument("--qa-output", default="data/external/policy_decisions_qa.csv", help="QA output CSV.")
    return parser.parse_args()


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Warning: missing input file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def parse_date_filter(df: pd.DataFrame, date_col: str, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_date"] = pd.to_datetime(df.get(date_col, ""), errors="coerce")
    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise ValueError("--start-date and --end-date must be valid dates.")
    df = df[df["_date"].between(start, end, inclusive="both")].copy()
    df[date_col] = df["_date"].dt.strftime("%Y-%m-%d")
    return df


def resolve_text_path(path_text: str, base_dir: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    if path.exists():
        return path
    alternate = find_alternate_text_path(path)
    return alternate


def normalize_path_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def find_alternate_text_path(path: Path) -> Path | None:
    parent = path.parent
    if not parent.exists():
        return None
    target = normalize_path_match(path.stem)
    candidates = []
    for candidate in parent.glob("*.txt"):
        candidate_norm = normalize_path_match(candidate.stem)
        if target and (target in candidate_norm or candidate_norm in target):
            candidates.append(candidate)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (len(item.name), item.name))[0]


def read_text(path_text: str, base_dir: Path) -> tuple[str, str]:
    path = resolve_text_path(path_text, base_dir)
    if path is None:
        return "", f"text file missing: {path_text}"
    try:
        return path.read_text(encoding="utf-8", errors="replace"), ""
    except OSError as exc:
        return "", str(exc)


def normalize_text(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = text.replace("â€“", "-").replace("â€”", "-").replace("â€™", "'")
    return re.sub(r"\s+", " ", text)


def excerpt(text: str, match: re.Match[str] | None, radius: int = 250) -> str:
    if not match:
        return ""
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return text[start:end].strip()


def parse_percent(value: str) -> float | None:
    value = str(value).strip().lower().replace("percent", "").replace("%", "")
    value = value.replace("¼", "/4").replace("½", "/2").replace("¾", "/4")
    value = value.replace("1/4", ".25").replace("1/2", ".5").replace("3/4", ".75")
    value = value.replace("-", " ")
    parts = value.split()
    try:
        if len(parts) == 2 and re.fullmatch(r"\d+", parts[0]) and parts[1].startswith("."):
            return float(parts[0]) + float(parts[1])
        return float(value)
    except ValueError:
        return None


def extract_fed_target_range(text: str) -> tuple[float | None, str, re.Match[str] | None]:
    patterns = [
        r"target range for the federal funds rate at\s+([0-9][0-9./¼½¾ -]*)\s+to\s+([0-9][0-9./¼½¾ -]*)\s+percent",
        r"federal funds rate at\s+([0-9][0-9./¼½¾ -]*)\s+to\s+([0-9][0-9./¼½¾ -]*)\s+percent",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        low = parse_percent(match.group(1))
        high = parse_percent(match.group(2))
        if low is None or high is None:
            continue
        midpoint = (low + high) / 2
        return midpoint, f"Fed target range {low:g} to {high:g} percent; midpoint stored.", match
    return None, "", None


def extract_policy_rate_after(text: str, bank: str) -> tuple[float | None, str, re.Match[str] | None]:
    if bank.upper() == "FED":
        midpoint, notes, match = extract_fed_target_range(text)
        if midpoint is not None:
            return midpoint, notes, match

    rate_patterns = [
        r"(?:policy rate|cash rate target|official cash rate|bank rate|key policy rate|sight deposit rate|target for the overnight rate|overnight rate)\s+(?:to|at)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)",
        r"(?:target for the overnight rate|overnight rate)\s+to\s+([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)",
        r"(?:main refinancing operations rate|deposit facility rate|marginal lending facility rate)\s+(?:to|at|will be)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)",
        r"(?:kept|keeps|left|leaves|maintain|maintains|maintained)\s+(?:the\s+)?(?:policy rate|cash rate target|official cash rate|bank rate|key policy rate|sight deposit rate|target for the overnight rate|overnight rate)\s+(?:unchanged\s+)?at\s+([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)",
        r"(?:set|sets)\s+(?:the\s+)?(?:policy rate|cash rate target|official cash rate|bank rate|key policy rate)\s+at\s+([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)",
    ]
    for pattern in rate_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)), "", match
    return None, "", None


def extract_explicit_bps(text: str) -> tuple[float | None, str, re.Match[str] | None]:
    bps_pattern = (
        r"\b(raised|raising|increase(?:d)?|increases|increasing|tightened|tightens|tightening|lowered|lowering|reduced|reduces|reducing|cut|cuts|cutting|decrease(?:d)?|decreases|decreasing|eased|eases|easing)"
        r"\b.{0,80}?\bby\s+([0-9]+(?:\.[0-9]+)?)\s+(?:basis points|bps)\b"
    )
    match = re.search(bps_pattern, text, flags=re.IGNORECASE)
    if match:
        verb = match.group(1).lower()
        value = float(match.group(2))
        if verb.startswith(("lower", "reduc", "cut", "decreas", "eas")):
            value = -value
        return value, "explicit_bps_change", match

    unchanged_pattern = r"\b(unchanged|maintained|maintain|kept|keep|left unchanged|leave unchanged|hold|holds)\b"
    match = re.search(unchanged_pattern, text, flags=re.IGNORECASE)
    if match:
        return 0.0, "explicit_unchanged_keyword", match
    return None, "", None


def infer_decision(rate_change_bps: float | None, method: str) -> str:
    if rate_change_bps is None or pd.isna(rate_change_bps):
        return "unknown"
    if rate_change_bps > 0:
        return "hike"
    if rate_change_bps < 0:
        return "cut"
    if rate_change_bps == 0:
        return "hold"
    return "unknown"


def confidence_for(rate_change_bps: float | None, rate_after: float | None, method: str) -> float:
    if method == "calculated_from_policy_rate_after":
        return 0.7
    if rate_change_bps is not None and rate_after is not None and method == "explicit_bps_change":
        return 1.0
    if rate_change_bps == 0 and rate_after is not None:
        return 0.9
    if rate_change_bps is not None and method == "explicit_bps_change":
        return 0.8
    if rate_change_bps is not None and method == "explicit_unchanged_keyword":
        return 0.5
    return 0.0


def extract_decision(text: str, bank: str) -> ExtractionResult:
    clean = normalize_text(text)
    rate_after, notes, rate_match = extract_policy_rate_after(clean, bank)
    rate_change, change_method, change_match = extract_explicit_bps(clean)
    method = change_method or ("policy_rate_after_only" if rate_after is not None else "unknown")
    decision = infer_decision(rate_change, method)
    conf = confidence_for(rate_change, rate_after, method)
    chosen_match = change_match or rate_match
    return ExtractionResult(
        decision=decision,
        rate_change_bps=rate_change,
        policy_rate_after=rate_after,
        confidence=conf,
        extraction_method=method,
        notes=notes,
        text_excerpt=excerpt(clean, chosen_match),
    )


def candidate_priority(row: pd.Series) -> int:
    subtype = str(row.get("document_subtype", "")).lower()
    title = str(row.get("title", "")).lower()
    category = str(row.get("category", "")).lower()
    if subtype == "statement":
        return 0
    if "monetary policy decision" in title:
        return 1
    if "interest rate announcement" in title:
        return 2
    if "press release" in title:
        return 3
    if "monetary" in category or "monetary policy" in title:
        return 4
    if subtype == "minutes" or "minutes" in title:
        return 5
    return 99


def build_candidate_docs(meeting_index: pd.DataFrame, meeting_documents: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    docs = ensure_columns(
        meeting_documents,
        ["meeting_id", "doc_id", "document_subtype", "category", "title", "text_file", "source_url"],
    )
    master = ensure_columns(
        master,
        ["meeting_id", "doc_id", "document_subtype", "category", "title", "text_file", "text_file_path", "source_url"],
    )
    if not docs.empty:
        rows.append(docs[["meeting_id", "doc_id", "document_subtype", "category", "title", "text_file", "source_url"]])
    if not master.empty:
        m = master.copy()
        m["text_file"] = m["text_file"].where(m["text_file"].astype(str).str.len() > 0, m["text_file_path"])
        rows.append(m[["meeting_id", "doc_id", "document_subtype", "category", "title", "text_file", "source_url"]])

    anchors = ensure_columns(meeting_index, ["meeting_id", "anchor_doc_id", "anchor_title", "anchor_text_file"])
    if not anchors.empty:
        anchor_docs = pd.DataFrame(
            {
                "meeting_id": anchors["meeting_id"],
                "doc_id": anchors["anchor_doc_id"],
                "document_subtype": "statement",
                "category": "Monetary Policy",
                "title": anchors["anchor_title"],
                "text_file": anchors["anchor_text_file"],
                "source_url": "",
            }
        )
        rows.append(anchor_docs)

    if not rows:
        return pd.DataFrame(columns=["meeting_id", "doc_id", "document_subtype", "category", "title", "text_file", "source_url"])

    out = pd.concat(rows, ignore_index=True).drop_duplicates(["meeting_id", "doc_id", "text_file", "title"])
    out["_priority"] = out.apply(candidate_priority, axis=1)
    out = out[out["_priority"] <= 5].copy()
    return out.sort_values(["meeting_id", "_priority", "title"])


def build_policy_decisions(
    meeting_index: pd.DataFrame,
    candidates: pd.DataFrame,
    base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    candidates_by_meeting = {mid: group.copy() for mid, group in candidates.groupby("meeting_id")} if not candidates.empty else {}

    for _, meeting in meeting_index.iterrows():
        meeting_id = str(meeting.get("meeting_id", ""))
        bank = str(meeting.get("bank", ""))
        selected = None
        result = ExtractionResult(notes="No readable source document found.")
        text_error = ""

        for _, candidate in candidates_by_meeting.get(meeting_id, pd.DataFrame()).iterrows():
            text, error = read_text(str(candidate.get("text_file", "")), base_dir)
            if error:
                text_error = error
                continue
            selected = candidate
            result = extract_decision(text, bank)
            if result.decision != "unknown" or result.policy_rate_after is not None:
                break

        if selected is None:
            selected = pd.Series({"doc_id": meeting.get("anchor_doc_id", ""), "title": meeting.get("anchor_title", ""), "source_url": ""})
            if text_error:
                result.notes = text_error

        row = {
            "bank": bank,
            "meeting_date": meeting.get("meeting_date", ""),
            "meeting_id": meeting_id,
            "decision": result.decision,
            "rate_change_bps": "" if result.rate_change_bps is None else result.rate_change_bps,
            "policy_rate_after": "" if result.policy_rate_after is None else result.policy_rate_after,
            "next_decision": "",
            "next_rate_change_bps": "",
            "source_doc_id": selected.get("doc_id", ""),
            "source_title": selected.get("title", ""),
            "source_url": selected.get("source_url", ""),
            "confidence": result.confidence,
            "extraction_method": result.extraction_method,
            "notes": result.notes,
        }
        output_rows.append(row)
        qa_rows.append({**{column: row.get(column, "") for column in QA_COLUMNS if column != "text_excerpt"}, "text_excerpt": result.text_excerpt})

    decisions = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    decisions = fill_calculated_changes(decisions)
    decisions = add_next_decisions(decisions)

    qa = pd.DataFrame(qa_rows)
    qa = qa.drop(columns=[c for c in ["next_decision", "next_rate_change_bps", "source_doc_id", "source_url"] if c in qa.columns], errors="ignore")
    qa = decisions[["bank", "meeting_date", "meeting_id", "decision", "rate_change_bps", "policy_rate_after", "confidence", "extraction_method", "source_title", "notes"]].merge(
        qa[["meeting_id", "text_excerpt"]], on="meeting_id", how="left"
    )
    return decisions, qa[QA_COLUMNS]


def fill_calculated_changes(decisions: pd.DataFrame) -> pd.DataFrame:
    df = decisions.copy()
    df["_date"] = pd.to_datetime(df["meeting_date"], errors="coerce")
    df["policy_rate_after_num"] = pd.to_numeric(df["policy_rate_after"], errors="coerce")
    df["rate_change_num"] = pd.to_numeric(df["rate_change_bps"], errors="coerce")
    df = df.sort_values(["bank", "_date", "meeting_id"])
    for bank, group in df.groupby("bank"):
        previous_rate = None
        for idx, row in group.iterrows():
            current_rate = row["policy_rate_after_num"]
            change = row["rate_change_num"]
            if pd.isna(change) and previous_rate is not None and not pd.isna(current_rate):
                calculated = round((current_rate - previous_rate) * 100, 6)
                df.at[idx, "rate_change_bps"] = calculated
                df.at[idx, "decision"] = infer_decision(calculated, "calculated_from_policy_rate_after")
                df.at[idx, "confidence"] = max(float(row["confidence"]), 0.7)
                df.at[idx, "extraction_method"] = "calculated_from_policy_rate_after"
                df.at[idx, "notes"] = append_note(str(row["notes"]), "Rate change calculated from consecutive policy_rate_after values.")
            if not pd.isna(current_rate):
                previous_rate = current_rate
    return df.drop(columns=["_date", "policy_rate_after_num", "rate_change_num"])


def append_note(existing: str, note: str) -> str:
    existing = str(existing or "").strip()
    return note if not existing else f"{existing} {note}"


def add_next_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    df = decisions.copy()
    df["_date"] = pd.to_datetime(df["meeting_date"], errors="coerce")
    df = df.sort_values(["bank", "_date", "meeting_id"])
    df["next_decision"] = df.groupby("bank")["decision"].shift(-1).fillna("")
    df["next_rate_change_bps"] = df.groupby("bank")["rate_change_bps"].shift(-1).fillna("")
    return df.drop(columns=["_date"])[OUTPUT_COLUMNS]


def write_readme(path: Path) -> None:
    text = """# Policy Decisions Builder

`policy_decisions.csv` is an external validation input for Phase 4 Hawkometer backtesting. It is generated from the scraped central bank meeting documents and can later be replaced or augmented with official rate datasets.

## Extraction Method

The builder starts from `processed/meeting_linkage/meeting_index.csv`, then looks for linked statement, monetary policy decision, interest rate announcement, press release, monetary policy, and minutes documents. It reads the selected text file and searches for policy-rate and basis-point decision phrases.

## Fed Target Ranges

For Federal Reserve target ranges, the midpoint is stored in `policy_rate_after`. The original target range is noted in `notes`.

## Confidence

- `1.0`: explicit basis-point change and policy rate after found
- `0.9`: explicit hold/unchanged and policy rate after found
- `0.8`: explicit basis-point change found but no rate after
- `0.7`: rate change calculated from consecutive policy rates
- `0.5`: decision inferred from hold keywords only
- `0.0`: unknown

## Manual Review

Review low-confidence rows in `policy_decisions_qa.csv`, especially rows with `unknown`, blank rates, or confidence below `0.7`. The QA file includes an excerpt around the matched phrase.

## Rerun Phase 4

```powershell
python hawkometer_backtest.py --hawkometer processed/hawkometer --analytics processed/hawkometer_analytics --meeting-linkage processed/meeting_linkage --output processed/hawkometer_backtest --policy-decisions data/external/policy_decisions.csv
```
"""
    path.write_text(text, encoding="utf-8")


def run(
    meeting_linkage_dir: Path,
    output_path: Path,
    start_date: str,
    end_date: str,
    min_confidence: float,
    qa_output: Path,
) -> None:
    meeting_index = read_csv_safe(meeting_linkage_dir / "meeting_index.csv")
    master = read_csv_safe(meeting_linkage_dir / "updated_master_index.csv")
    meeting_documents = read_csv_safe(meeting_linkage_dir / "meeting_documents.csv")

    meeting_index = ensure_columns(meeting_index, ["meeting_id", "bank", "meeting_date", "anchor_doc_id", "anchor_title", "anchor_text_file"])
    meeting_index = parse_date_filter(meeting_index, "meeting_date", start_date, end_date)
    candidates = build_candidate_docs(meeting_index, meeting_documents, master)
    decisions, qa = build_policy_decisions(meeting_index, candidates, Path.cwd())

    decisions["confidence"] = pd.to_numeric(decisions["confidence"], errors="coerce").fillna(0)
    filtered = decisions[decisions["confidence"] >= min_confidence].copy()
    qa = qa[qa["meeting_id"].isin(filtered["meeting_id"])].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    qa_output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    qa.to_csv(qa_output, index=False)
    write_readme(output_path.parent / "README_policy_decisions.md")

    counts = filtered["decision"].value_counts()
    extracted = int(filtered["decision"].ne("unknown").sum())
    unknown = int(filtered["decision"].eq("unknown").sum())
    print(f"meetings loaded: {len(meeting_index)}")
    print(f"decisions extracted: {extracted}")
    print(f"unknown decisions: {unknown}")
    print(f"hikes: {int(counts.get('hike', 0))}")
    print(f"holds: {int(counts.get('hold', 0))}")
    print(f"cuts: {int(counts.get('cut', 0))}")
    print(f"average confidence: {filtered['confidence'].mean():.3f}" if len(filtered) else "average confidence: n/a")
    print(f"output path: {output_path}")
    print(f"qa output path: {qa_output}")


def main() -> None:
    args = parse_args()
    run(
        meeting_linkage_dir=Path(args.meeting_linkage),
        output_path=Path(args.output),
        start_date=args.start_date,
        end_date=args.end_date,
        min_confidence=args.min_confidence,
        qa_output=Path(args.qa_output),
    )


if __name__ == "__main__":
    main()
