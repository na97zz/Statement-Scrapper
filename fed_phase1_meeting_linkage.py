from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd


FED_BANK = "FED"

MASTER_COLUMNS = [
    "doc_id",
    "bank",
    "category",
    "document_subtype",
    "title",
    "date_au",
    "speaker",
    "source_url",
    "raw_file_path",
    "text_file_path",
    "document_type",
    "status",
    "scraped_at",
    "meeting_id",
    "meeting_relation",
    "meeting_date",
    "days_after_meeting",
    "linkage_confidence",
    "linkage_reason",
]

MEETING_COLUMNS = [
    "meeting_id",
    "bank",
    "meeting_date",
    "year",
    "month",
    "anchor_doc_id",
    "anchor_title",
    "anchor_text_file",
    "next_meeting_date",
    "meeting_window_start",
    "meeting_window_end",
]

MEETING_DOCUMENT_COLUMNS = [
    "meeting_id",
    "bank",
    "meeting_date",
    "doc_id",
    "document_subtype",
    "category",
    "title",
    "speaker",
    "date_au",
    "days_after_meeting",
    "text_file_path",
    "source_url",
    "meeting_relation",
    "linkage_confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FED-only Phase 1 meeting linkage outputs.")
    parser.add_argument("--input", default="data/master_index.csv", help="Input raw master_index.csv.")
    parser.add_argument("--output", default="processed/fed", help="Output folder for FED Phase 1 files.")
    parser.add_argument("--fallback-window-days", type=int, default=45, help="Window end for latest meeting.")
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def parse_date(value: Any) -> pd.Timestamp:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none", "unknown date"}:
        return pd.NaT
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", text):
        return pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", text):
        return pd.to_datetime(text, dayfirst=True, errors="coerce")
    compact = re.search(r"(20\d{2})[._/-]?(\d{2})[._/-]?(\d{2})", text)
    if compact:
        return pd.to_datetime("-".join(compact.groups()), format="%Y-%m-%d", errors="coerce")
    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def normalize_master(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(
        df,
        [
            "bank",
            "category",
            "title",
            "date_au",
            "speaker",
            "source_url",
            "raw_file_path",
            "text_file_path",
            "document_type",
            "status",
            "scraped_at",
            "doc_id",
        ],
    )
    text_cols = [
        "bank",
        "category",
        "title",
        "speaker",
        "source_url",
        "raw_file_path",
        "text_file_path",
        "document_type",
        "status",
        "scraped_at",
        "doc_id",
    ]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df["bank"].str.upper().eq(FED_BANK)].copy()
    df["bank"] = FED_BANK
    df["date"] = df["date_au"].apply(parse_date).dt.normalize()
    df["date_parse_failed"] = df["date"].isna()
    df["date_au"] = df["date"].dt.strftime("%Y-%m-%d")
    df.loc[df["date_parse_failed"], "date_au"] = ""
    df["document_subtype"] = df.apply(classify_document, axis=1)
    missing_doc_id = df["doc_id"].eq("")
    df.loc[missing_doc_id, "doc_id"] = df[missing_doc_id].apply(make_doc_id, axis=1)
    return df.sort_values(["date", "category", "title", "doc_id"]).reset_index(drop=True)


def make_doc_id(row: pd.Series) -> str:
    basis = "|".join(
        str(row.get(col, ""))
        for col in ["bank", "category", "date_au", "title", "source_url", "text_file_path"]
    )
    return "FED_DOC_" + hashlib.sha1(basis.encode("utf-8", errors="ignore")).hexdigest()[:12].upper()


def classify_document(row: pd.Series) -> str:
    category = str(row.get("category", "")).lower()
    title = str(row.get("title", "")).lower()
    url = str(row.get("source_url", "")).lower()
    if "minutes" in category or "minutes" in title:
        return "minutes"
    if "summary of economic projections" in title or re.search(r"\bsep\b", title):
        return "sep"
    if "press conference" in title or "transcript" in title:
        return "press_conference"
    if "speech" in category:
        return "speech"
    if "testimony" in title or "testimony" in url:
        return "testimony"
    if "monetary" in category and ("fomc statement" in title or "statement" in title):
        return "statement"
    return "other"


def is_fomc_statement_anchor(row: pd.Series) -> bool:
    title = str(row.get("title", "")).lower()
    category = str(row.get("category", "")).lower()
    if row.get("document_subtype") != "statement":
        return False
    if "monetary" not in category:
        return False
    if "longer-run goals" in title or "monetary policy strategy" in title:
        return False
    return "fomc statement" in title or title.strip() == "statement"


def build_meeting_index(fed: pd.DataFrame, fallback_window_days: int) -> pd.DataFrame:
    anchors = fed[fed.apply(is_fomc_statement_anchor, axis=1) & fed["date"].notna()].copy()
    anchors = anchors.sort_values(["date", "title", "doc_id"]).drop_duplicates(["date"], keep="first")
    if anchors.empty:
        return pd.DataFrame(columns=MEETING_COLUMNS)
    meetings = pd.DataFrame(
        {
            "meeting_id": anchors["date"].dt.strftime("FED_%Y_%m"),
            "bank": FED_BANK,
            "meeting_date": anchors["date"],
            "year": anchors["date"].dt.year.astype(int),
            "month": anchors["date"].dt.month.astype(int),
            "anchor_doc_id": anchors["doc_id"],
            "anchor_title": anchors["title"],
            "anchor_text_file": anchors["text_file_path"],
        }
    ).sort_values("meeting_date")
    meetings["next_meeting_date"] = meetings["meeting_date"].shift(-1)
    meetings["meeting_window_start"] = meetings["meeting_date"]
    meetings["meeting_window_end"] = meetings["next_meeting_date"] - pd.Timedelta(days=1)
    no_next = meetings["meeting_window_end"].isna()
    meetings.loc[no_next, "meeting_window_end"] = meetings.loc[no_next, "meeting_date"] + pd.to_timedelta(
        fallback_window_days, unit="D"
    )
    for col in ["meeting_date", "next_meeting_date", "meeting_window_start", "meeting_window_end"]:
        meetings[col] = meetings[col].dt.strftime("%Y-%m-%d").fillna("")
    return meetings[MEETING_COLUMNS].reset_index(drop=True)


def meeting_dates(meetings: pd.DataFrame) -> pd.DataFrame:
    out = meetings.copy()
    for col in ["meeting_date", "meeting_window_start", "meeting_window_end"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def nearest_same_day_or_month(meetings: pd.DataFrame, doc_date: pd.Timestamp) -> pd.Series | None:
    if pd.isna(doc_date) or meetings.empty:
        return None
    same_day = meetings[meetings["meeting_date"].eq(doc_date)]
    if not same_day.empty:
        return same_day.iloc[0]
    same_month = meetings[
        (meetings["meeting_date"].dt.year == doc_date.year) & (meetings["meeting_date"].dt.month == doc_date.month)
    ].copy()
    if same_month.empty:
        return None
    same_month["_distance"] = (same_month["meeting_date"] - doc_date).abs()
    return same_month.sort_values(["_distance", "meeting_date"]).iloc[0]


def latest_on_or_before(meetings: pd.DataFrame, doc_date: pd.Timestamp) -> pd.Series | None:
    if pd.isna(doc_date) or meetings.empty:
        return None
    prior = meetings[meetings["meeting_date"] <= doc_date]
    if prior.empty:
        return None
    return prior.sort_values("meeting_date").iloc[-1]


def in_meeting_window(meetings: pd.DataFrame, doc_date: pd.Timestamp) -> pd.Series | None:
    if pd.isna(doc_date) or meetings.empty:
        return None
    candidates = meetings[(meetings["meeting_window_start"] <= doc_date) & (meetings["meeting_window_end"] >= doc_date)]
    if candidates.empty:
        return None
    return candidates.sort_values("meeting_date").iloc[-1]


def link_documents(fed: pd.DataFrame, meetings: pd.DataFrame) -> pd.DataFrame:
    linked = fed.copy()
    meetings_dt = meeting_dates(meetings)
    anchor_map = meetings_dt.set_index("anchor_doc_id", drop=False) if not meetings_dt.empty else pd.DataFrame()
    for col, default in [
        ("meeting_id", ""),
        ("meeting_relation", "unlinked"),
        ("meeting_date", ""),
        ("days_after_meeting", ""),
        ("linkage_confidence", 0.0),
        ("linkage_reason", ""),
    ]:
        linked[col] = default

    for idx, row in linked.iterrows():
        selected = None
        subtype = row["document_subtype"]
        relation = "unlinked"
        confidence = 0.0
        reason = ""
        if row["doc_id"] in anchor_map.index:
            selected = anchor_map.loc[row["doc_id"]]
            relation = "anchor_statement"
            confidence = 1.0
            reason = "Official FOMC statement anchor."
        elif subtype == "minutes":
            selected = latest_on_or_before(meetings_dt, row["date"])
            relation = "minutes_for_meeting" if selected is not None else "unlinked"
            confidence = 0.9 if selected is not None else 0.0
            reason = "Linked to latest FOMC meeting on or before minutes date." if selected is not None else "No prior FOMC meeting found."
        elif subtype == "sep":
            selected = nearest_same_day_or_month(meetings_dt, row["date"])
            relation = "sep_for_meeting" if selected is not None else "unlinked"
            confidence = 0.9 if selected is not None else 0.0
            reason = "Linked to same-day or same-month FOMC meeting." if selected is not None else "No same-day/month FOMC meeting found."
        elif subtype == "press_conference":
            selected = nearest_same_day_or_month(meetings_dt, row["date"])
            relation = "press_conference_for_meeting" if selected is not None else "unlinked"
            confidence = 0.9 if selected is not None else 0.0
            reason = "Linked to same-day or same-month FOMC meeting." if selected is not None else "No same-day/month FOMC meeting found."
        elif subtype == "speech":
            selected = in_meeting_window(meetings_dt, row["date"])
            relation = "speech_after_meeting" if selected is not None else "unlinked"
            confidence = 0.7 if selected is not None else 0.0
            reason = "Speech falls inside a FOMC meeting window." if selected is not None else "Speech outside all FOMC meeting windows."
        else:
            selected = in_meeting_window(meetings_dt, row["date"])
            relation = "other_in_meeting_window" if selected is not None else "unlinked"
            confidence = 0.4 if selected is not None else 0.0
            reason = "Document falls inside a FOMC meeting window." if selected is not None else "Document outside all FOMC meeting windows."

        if selected is not None:
            meeting_date = selected["meeting_date"]
            linked.at[idx, "meeting_id"] = selected["meeting_id"]
            linked.at[idx, "meeting_date"] = meeting_date.strftime("%Y-%m-%d")
            linked.at[idx, "days_after_meeting"] = str(int((row["date"] - meeting_date).days))
        linked.at[idx, "meeting_relation"] = relation
        linked.at[idx, "linkage_confidence"] = confidence
        linked.at[idx, "linkage_reason"] = reason
    return linked


def build_meeting_documents(linked: pd.DataFrame) -> pd.DataFrame:
    docs = linked[linked["meeting_id"].astype(str).str.len() > 0].copy()
    if docs.empty:
        return pd.DataFrame(columns=MEETING_DOCUMENT_COLUMNS)
    return docs[MEETING_DOCUMENT_COLUMNS].sort_values(["meeting_date", "date_au", "document_subtype", "title"])


def build_summary(linked: pd.DataFrame, meetings: pd.DataFrame, missing_text_files: int) -> pd.DataFrame:
    linked_count = int(linked["meeting_id"].astype(str).str.len().gt(0).sum())
    rows = [
        ("bank", FED_BANK),
        ("fed_rows_loaded", len(linked)),
        ("meetings_created", len(meetings)),
        ("documents_linked", linked_count),
        ("documents_unlinked", len(linked) - linked_count),
        ("missing_text_files", missing_text_files),
        ("date_min", linked["date_au"].replace("", pd.NA).dropna().min() if len(linked) else ""),
        ("date_max", linked["date_au"].replace("", pd.NA).dropna().max() if len(linked) else ""),
        ("statement_count", int(linked["document_subtype"].eq("statement").sum())),
        ("minutes_count", int(linked["document_subtype"].eq("minutes").sum())),
        ("speech_count", int(linked["document_subtype"].eq("speech").sum())),
        ("press_conference_count", int(linked["document_subtype"].eq("press_conference").sum())),
        ("sep_count", int(linked["document_subtype"].eq("sep").sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def missing_text_count(fed: pd.DataFrame, root: Path) -> int:
    missing = 0
    for value in fed["text_file_path"].fillna("").astype(str):
        if not value:
            missing += 1
            continue
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            missing += 1
    return missing


def write_readme(output: Path) -> None:
    text = """# FED Phase 1 Meeting Linkage

This folder contains a FED-only Phase 1 meeting linkage dataset built from `data/master_index.csv` and the raw FED archive.

## Files

- `fed_master_index.csv`: FED-only master index with document subtype and meeting linkage columns.
- `fed_meeting_index.csv`: one row per FOMC statement anchor meeting.
- `fed_meeting_documents.csv`: one row per linked FED document.
- `fed_linkage_summary.csv`: basic counts and linkage coverage.

## Meeting IDs

Meeting IDs use:

```text
FED_YYYY_MM
```

## Linkage Rules

- FOMC statements are meeting anchors.
- Minutes link to the latest FOMC meeting on or before the minutes date.
- SEP and press conference documents link to the nearest same-day or same-month meeting.
- Speeches link to the meeting window from the latest FOMC statement through the day before the next statement.

## Run

```powershell
python fed_phase1_meeting_linkage.py --input data/master_index.csv --output processed/fed
```
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def run(input_path: Path, output: Path, fallback_window_days: int) -> None:
    root = Path.cwd()
    raw_fed = root / "data" / "FED"
    if not raw_fed.exists():
        alt = root / "FED"
        if not alt.exists():
            raise FileNotFoundError("FED raw folder not found at data/FED or FED.")
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    fed = normalize_master(df)
    missing_text_files = missing_text_count(fed, root)
    meetings = build_meeting_index(fed, fallback_window_days)
    linked = link_documents(fed, meetings)
    meeting_documents = build_meeting_documents(linked)
    summary = build_summary(linked, meetings, missing_text_files)

    output.mkdir(parents=True, exist_ok=True)
    linked[MASTER_COLUMNS].to_csv(output / "fed_master_index.csv", index=False)
    meetings.to_csv(output / "fed_meeting_index.csv", index=False)
    meeting_documents.to_csv(output / "fed_meeting_documents.csv", index=False)
    summary.to_csv(output / "fed_linkage_summary.csv", index=False)
    write_readme(output)

    linked_count = int(linked["meeting_id"].astype(str).str.len().gt(0).sum())
    date_min = linked["date_au"].replace("", pd.NA).dropna().min() if len(linked) else ""
    date_max = linked["date_au"].replace("", pd.NA).dropna().max() if len(linked) else ""
    print(f"FED rows loaded: {len(fed)}")
    print(f"meetings created: {len(meetings)}")
    print(f"documents linked: {linked_count}")
    print(f"documents unlinked: {len(linked) - linked_count}")
    print(f"missing text files: {missing_text_files}")
    print(f"date range: {date_min} to {date_max}")
    print(f"output folder: {output}")


def main() -> None:
    args = parse_args()
    run(Path(args.input), Path(args.output), args.fallback_window_days)


if __name__ == "__main__":
    main()
