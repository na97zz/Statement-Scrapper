from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


OUTPUT_COLUMNS = [
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
    "text_file",
    "source_url",
    "meeting_relation",
    "linkage_confidence",
]


@dataclass
class LinkageStats:
    total_rows_loaded: int
    total_meetings_created: int
    documents_linked: int
    documents_unlinked: int
    documents_missing_dates: int
    duplicate_meeting_ids: int
    banks_with_no_meeting_anchors: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build meeting-cycle linkage tables for central bank documents."
    )
    parser.add_argument("--input", default="data/master_index.csv", help="Input master_index.csv path.")
    parser.add_argument(
        "--output",
        default="processed/meeting_linkage",
        help="Output folder for meeting linkage CSV files.",
    )
    parser.add_argument(
        "--speech-window-days",
        type=int,
        default=45,
        help="Fallback meeting window length when there is no later meeting.",
    )
    parser.add_argument("--min-date", default="2021-01-01", help="Minimum document date to link.")
    parser.add_argument("--max-date", default="2026-12-31", help="Maximum document date to link.")
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in [
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
    ]:
        if column not in df.columns:
            df[column] = ""

    if "raw_file" not in df.columns:
        df["raw_file"] = df["raw_file_path"]
    if "text_file" not in df.columns:
        df["text_file"] = df["text_file_path"]
    if "document" not in df.columns:
        df["document"] = df["document_type"]

    text_columns = [
        "bank",
        "category",
        "title",
        "speaker",
        "source_url",
        "raw_file",
        "raw_file_path",
        "text_file",
        "text_file_path",
        "document",
        "document_type",
        "status",
        "scraped_at",
    ]
    for column in text_columns:
        df[column] = df[column].fillna("").astype(str).str.strip()
    return df


def parse_one_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "unknown date"}:
        return pd.NaT
    if set(text) == {"#"}:
        return pd.NaT

    # Excel serial dates can appear when CSVs are round-tripped through Excel.
    if re.fullmatch(r"\d+(\.0+)?", text):
        serial = float(text)
        if 20000 <= serial <= 60000:
            return pd.to_datetime(serial, unit="D", origin="1899-12-30", errors="coerce")

    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", text):
        return pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")

    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", text):
        return pd.to_datetime(text, dayfirst=True, errors="coerce")

    compact = re.search(r"(20\d{2})[._/-]?(\d{2})[._/-]?(\d{2})", text)
    if compact:
        return pd.to_datetime("-".join(compact.groups()), format="%Y-%m-%d", errors="coerce")

    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def parse_full_date_from_title(title: str) -> pd.Timestamp:
    text = str(title or "").strip()
    match = re.fullmatch(
        r"(\d{1,2})\s+"
        r"(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+(20\d{2})",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return pd.NaT
    day, month, year = match.groups()
    return pd.to_datetime(f"{day} {month} {year}", dayfirst=True, errors="coerce")


def month_year_from_title(title: str) -> tuple[int, int] | None:
    text = str(title or "").strip().lower()
    match = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+(20\d{2})\b",
        text,
    )
    if not match:
        return None
    month_name, year = match.groups()
    month = pd.to_datetime(month_name, format="%B").month
    return month, int(year)


def parse_compact_url_date(source_url: str, title: str) -> pd.Timestamp:
    title_month_year = month_year_from_title(title)
    if not title_month_year:
        return pd.NaT
    expected_month, expected_year = title_month_year

    for day, month, year_short in re.findall(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)", str(source_url or "")):
        year = 2000 + int(year_short)
        if int(month) == expected_month and year == expected_year:
            return pd.to_datetime(f"{year}-{month}-{day}", format="%Y-%m-%d", errors="coerce")
    return pd.NaT


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed = df["date_au"].apply(parse_one_date)
    title_dates = df["title"].apply(parse_full_date_from_title)
    use_title_date = title_dates.notna()
    parsed.loc[use_title_date] = title_dates.loc[use_title_date]
    url_dates = df.apply(lambda row: parse_compact_url_date(row["source_url"], row["title"]), axis=1)
    use_url_date = url_dates.notna()
    parsed.loc[use_url_date] = url_dates.loc[use_url_date]
    df["date"] = parsed.dt.normalize()
    df["date_parse_failed"] = df["date"].isna()
    df["date_au"] = df["date"].dt.strftime("%Y-%m-%d")
    df.loc[df["date_parse_failed"], "date_au"] = ""
    return df


def make_doc_id(row: pd.Series) -> str:
    basis = "|".join(
        str(row.get(column, ""))
        for column in ["bank", "category", "date_au", "title", "source_url", "text_file"]
    )
    digest = hashlib.sha1(basis.encode("utf-8", errors="ignore")).hexdigest()[:12].upper()
    return f"DOC_{digest}"


def classify_document(category: str, title: str) -> str:
    category_l = (category or "").lower()
    title_l = (title or "").lower()

    if "minutes" in category_l or "minutes" in title_l:
        return "minutes"
    if "summary of economic projections" in title_l or re.search(r"\bsep\b", title_l):
        return "sep"
    if "press conference" in title_l or "transcript" in title_l:
        return "press_conference"

    statement_terms = [
        "statement",
        "policy statement",
        "rate statement",
        "monetary policy decision",
        "monetary policy decisions",
        "interest rate announcement",
        "monetary policy report",
        "monetary policy assessment",
        "pengepolitisk rapport",
    ]
    if "monetary" in category_l and any(term in title_l for term in statement_terms):
        return "statement"

    # RBA statements are often titled only by meeting date.
    if "monetary" in category_l and re.fullmatch(r"\d{1,2}\s+[a-z]+\s+20\d{2}", title_l):
        return "statement"

    if "speech" in category_l:
        return "speech"
    if "testimony" in title_l:
        return "testimony"
    return "other"


def is_anchor_document(row: pd.Series) -> bool:
    if bool(row.get("date_parse_failed", False)):
        return False
    if row.get("document_subtype") != "statement":
        return False
    category_l = str(row.get("category", "")).lower()
    title_l = str(row.get("title", "")).lower()
    if "monetary" not in category_l:
        return False
    anchor_terms = [
        "statement",
        "monetary policy decision",
        "monetary policy decisions",
        "interest rate announcement",
        "monetary policy report",
        "monetary policy assessment",
        "pengepolitisk rapport",
    ]
    return any(term in title_l for term in anchor_terms) or bool(
        re.fullmatch(r"\d{1,2}\s+[a-z]+\s+20\d{2}", title_l)
    )


def suffix_for_position(position: int) -> str:
    letters = []
    n = position
    while True:
        n, remainder = divmod(n, 26)
        letters.append(chr(ord("A") + remainder))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(letters))


def create_meeting_index(df: pd.DataFrame, speech_window_days: int) -> pd.DataFrame:
    anchors = df[df["is_meeting_anchor"]].copy()
    if anchors.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    anchors = anchors.sort_values(["bank", "date", "title", "doc_id"]).reset_index(drop=True)
    anchors["year"] = anchors["date"].dt.year.astype(int)
    anchors["month"] = anchors["date"].dt.month.astype(int)
    anchors["_month_key"] = anchors["date"].dt.strftime("%Y_%m")
    anchors["_month_count"] = anchors.groupby(["bank", "_month_key"])["_month_key"].transform("size")
    anchors["_month_position"] = anchors.groupby(["bank", "_month_key"]).cumcount()

    def build_meeting_id(row: pd.Series) -> str:
        base = f"{row['bank']}_{row['_month_key']}"
        if int(row["_month_count"]) > 1:
            return f"{base}_{suffix_for_position(int(row['_month_position']))}"
        return base

    anchors["meeting_id"] = anchors.apply(build_meeting_id, axis=1)
    anchors["meeting_date"] = anchors["date"]
    anchors["anchor_doc_id"] = anchors["doc_id"]
    anchors["anchor_title"] = anchors["title"]
    anchors["anchor_text_file"] = anchors["text_file"]

    meetings = anchors[
        [
            "meeting_id",
            "bank",
            "meeting_date",
            "year",
            "month",
            "anchor_doc_id",
            "anchor_title",
            "anchor_text_file",
        ]
    ].copy()

    meetings = meetings.sort_values(["bank", "meeting_date", "meeting_id"]).reset_index(drop=True)
    meetings["next_meeting_date"] = meetings.groupby("bank")["meeting_date"].shift(-1)
    meetings["meeting_window_start"] = meetings["meeting_date"]
    meetings["meeting_window_end"] = meetings["next_meeting_date"] - pd.Timedelta(days=1)
    no_next = meetings["meeting_window_end"].isna()
    meetings.loc[no_next, "meeting_window_end"] = (
        meetings.loc[no_next, "meeting_date"] + pd.to_timedelta(speech_window_days, unit="D")
    )

    for column in ["meeting_date", "next_meeting_date", "meeting_window_start", "meeting_window_end"]:
        meetings[column] = meetings[column].dt.strftime("%Y-%m-%d")
        meetings[column] = meetings[column].fillna("")
    return meetings[OUTPUT_COLUMNS]


def meeting_dates(meetings: pd.DataFrame) -> pd.DataFrame:
    out = meetings.copy()
    for column in ["meeting_date", "meeting_window_start", "meeting_window_end"]:
        out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def choose_nearest_same_day_or_month(bank_meetings: pd.DataFrame, doc_date: pd.Timestamp) -> pd.Series | None:
    if bank_meetings.empty or pd.isna(doc_date):
        return None
    same_day = bank_meetings[bank_meetings["meeting_date"].eq(doc_date)]
    if not same_day.empty:
        return same_day.iloc[0]
    same_month = bank_meetings[
        (bank_meetings["meeting_date"].dt.year == doc_date.year)
        & (bank_meetings["meeting_date"].dt.month == doc_date.month)
    ].copy()
    if same_month.empty:
        return None
    same_month["_distance"] = (same_month["meeting_date"] - doc_date).abs()
    return same_month.sort_values(["_distance", "meeting_date", "meeting_id"]).iloc[0]


def choose_prior_meeting(bank_meetings: pd.DataFrame, doc_date: pd.Timestamp) -> pd.Series | None:
    if bank_meetings.empty or pd.isna(doc_date):
        return None
    prior = bank_meetings[bank_meetings["meeting_date"] <= doc_date]
    if prior.empty:
        return None
    return prior.sort_values(["meeting_date", "meeting_id"]).iloc[-1]


def choose_window_meeting(bank_meetings: pd.DataFrame, doc_date: pd.Timestamp) -> pd.Series | None:
    if bank_meetings.empty or pd.isna(doc_date):
        return None
    in_window = bank_meetings[
        (bank_meetings["meeting_window_start"] <= doc_date)
        & (bank_meetings["meeting_window_end"] >= doc_date)
    ]
    if in_window.empty:
        return None
    return in_window.sort_values(["meeting_date", "meeting_id"]).iloc[-1]


def link_documents(df: pd.DataFrame, meetings: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    meetings_dt = meeting_dates(meetings)
    by_bank = {bank: group.copy() for bank, group in meetings_dt.groupby("bank")}
    anchor_by_doc_id = meetings_dt.set_index("anchor_doc_id", drop=False) if not meetings_dt.empty else pd.DataFrame()

    defaults = {
        "meeting_id": "",
        "meeting_relation": "unlinked",
        "meeting_date": "",
        "days_after_meeting": "",
        "linkage_confidence": 0.0,
        "linkage_reason": "",
    }
    for column, value in defaults.items():
        df[column] = value

    for idx, row in df.iterrows():
        doc_date = row["date"]
        bank = row["bank"]
        subtype = row["document_subtype"]
        bank_meetings = by_bank.get(bank, pd.DataFrame())
        selected = None
        relation = "unlinked"
        confidence = 0.0
        reason = ""

        if bool(row.get("date_parse_failed", False)):
            reason = "Date could not be parsed."
        elif not bool(row.get("in_date_range", True)):
            reason = "Document date is outside the configured date range."
        elif row["doc_id"] in anchor_by_doc_id.index:
            selected = anchor_by_doc_id.loc[row["doc_id"]]
            if isinstance(selected, pd.DataFrame):
                selected = selected.iloc[0]
            relation = "anchor_statement"
            confidence = 1.0
            reason = "Document is an official meeting anchor."
        elif subtype == "minutes":
            selected = choose_prior_meeting(bank_meetings, doc_date)
            relation = "minutes_for_prior_meeting" if selected is not None else "unlinked"
            confidence = 0.9 if selected is not None else 0.0
            reason = "Linked to the most recent prior same-bank meeting." if selected is not None else "No prior same-bank meeting found."
        elif subtype == "sep":
            selected = choose_nearest_same_day_or_month(bank_meetings, doc_date)
            relation = "sep_for_meeting" if selected is not None else "unlinked"
            confidence = 0.9 if selected is not None and selected["meeting_date"] == doc_date else 0.8 if selected is not None else 0.0
            reason = "Linked to nearest same-day or same-month same-bank meeting." if selected is not None else "No same-day or same-month meeting found."
        elif subtype == "press_conference":
            selected = choose_nearest_same_day_or_month(bank_meetings, doc_date)
            relation = "press_conference_for_meeting" if selected is not None else "unlinked"
            confidence = 0.9 if selected is not None and selected["meeting_date"] == doc_date else 0.8 if selected is not None else 0.0
            reason = "Linked to nearest same-day or same-month same-bank meeting." if selected is not None else "No same-day or same-month meeting found."
        elif subtype == "speech":
            selected = choose_window_meeting(bank_meetings, doc_date)
            relation = "speech_after_meeting" if selected is not None else "unlinked"
            confidence = 0.7 if selected is not None else 0.0
            reason = "Speech falls inside a same-bank meeting window." if selected is not None else "Speech does not fall inside a meeting window."
        else:
            selected = choose_window_meeting(bank_meetings, doc_date)
            relation = "other_in_meeting_window" if selected is not None else "unlinked"
            confidence = 0.4 if selected is not None else 0.0
            reason = "Document falls inside a same-bank meeting window." if selected is not None else "Document does not fall inside a meeting window."

        if selected is not None:
            meeting_date = selected["meeting_date"]
            df.at[idx, "meeting_id"] = selected["meeting_id"]
            df.at[idx, "meeting_date"] = meeting_date.strftime("%Y-%m-%d")
            df.at[idx, "days_after_meeting"] = str(int((doc_date - meeting_date).days))
        df.at[idx, "meeting_relation"] = relation
        df.at[idx, "linkage_confidence"] = confidence
        df.at[idx, "linkage_reason"] = reason

    return df


def create_meeting_documents(linked: pd.DataFrame) -> pd.DataFrame:
    docs = linked[linked["meeting_id"].astype(str).str.len() > 0].copy()
    if docs.empty:
        return pd.DataFrame(columns=MEETING_DOCUMENT_COLUMNS)
    return docs[MEETING_DOCUMENT_COLUMNS].sort_values(["bank", "meeting_date", "date_au", "document_subtype", "title"])


def create_linkage_summary(linked: pd.DataFrame, meetings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_banks: Iterable[str] = sorted(set(linked["bank"].dropna().astype(str)) | set(meetings["bank"].dropna().astype(str)))
    for bank in all_banks:
        bank_docs = linked[linked["bank"].eq(bank)]
        bank_meetings = meetings[meetings["bank"].eq(bank)]
        linked_docs = bank_docs[bank_docs["meeting_id"].astype(str).str.len() > 0]
        number_of_meetings = len(bank_meetings)
        rows.append(
            {
                "bank": bank,
                "number_of_meetings": number_of_meetings,
                "total_documents": len(bank_docs),
                "linked_documents": len(linked_docs),
                "unlinked_documents": len(bank_docs) - len(linked_docs),
                "statement_count": int(bank_docs["document_subtype"].eq("statement").sum()),
                "minutes_count": int(bank_docs["document_subtype"].eq("minutes").sum()),
                "speech_count": int(bank_docs["document_subtype"].eq("speech").sum()),
                "press_conference_count": int(bank_docs["document_subtype"].eq("press_conference").sum()),
                "sep_count": int(bank_docs["document_subtype"].eq("sep").sum()),
                "average_documents_per_meeting": round(len(linked_docs) / number_of_meetings, 2)
                if number_of_meetings
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def print_validation(stats: LinkageStats) -> None:
    print(f"total rows loaded: {stats.total_rows_loaded}")
    print(f"total meetings created: {stats.total_meetings_created}")
    print(f"documents linked: {stats.documents_linked}")
    print(f"documents unlinked: {stats.documents_unlinked}")
    print(f"documents missing dates: {stats.documents_missing_dates}")
    print(f"duplicate meeting_ids: {stats.duplicate_meeting_ids}")
    if stats.banks_with_no_meeting_anchors:
        print("banks with no meeting anchors: " + ", ".join(stats.banks_with_no_meeting_anchors))
    else:
        print("banks with no meeting anchors: none")


def resolve_input_path(input_path: Path) -> Path:
    if input_path.exists():
        return input_path
    fallback = Path("data") / input_path.name
    if input_path.name == "master_index.csv" and fallback.exists():
        return fallback
    raise FileNotFoundError(f"Input file not found: {input_path}")


def run(input_path: Path, output_dir: Path, speech_window_days: int, min_date: str, max_date: str) -> LinkageStats:
    input_path = resolve_input_path(input_path)
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    total_rows = len(df)

    df = normalize_columns(df)
    df = clean_dates(df)
    df["doc_id"] = df.apply(make_doc_id, axis=1)
    df["document_subtype"] = df.apply(lambda row: classify_document(row["category"], row["title"]), axis=1)

    min_ts = pd.to_datetime(min_date, errors="coerce")
    max_ts = pd.to_datetime(max_date, errors="coerce")
    if pd.isna(min_ts) or pd.isna(max_ts):
        raise ValueError("--min-date and --max-date must be parseable dates.")
    df["in_date_range"] = df["date"].between(min_ts, max_ts, inclusive="both")
    df.loc[df["date_parse_failed"], "in_date_range"] = False
    df["is_meeting_anchor"] = df.apply(is_anchor_document, axis=1) & df["in_date_range"]

    meetings = create_meeting_index(df, speech_window_days=speech_window_days)
    linked = link_documents(df, meetings)
    meeting_documents = create_meeting_documents(linked)
    summary = create_linkage_summary(linked, meetings)

    output_dir.mkdir(parents=True, exist_ok=True)
    linked.to_csv(output_dir / "updated_master_index.csv", index=False)
    meetings.to_csv(output_dir / "meeting_index.csv", index=False)
    meeting_documents.to_csv(output_dir / "meeting_documents.csv", index=False)
    summary.to_csv(output_dir / "linkage_summary.csv", index=False)

    all_banks = set(df["bank"].dropna().astype(str))
    anchor_banks = set(meetings["bank"].dropna().astype(str)) if not meetings.empty else set()
    stats = LinkageStats(
        total_rows_loaded=total_rows,
        total_meetings_created=len(meetings),
        documents_linked=int(linked["meeting_id"].astype(str).str.len().gt(0).sum()),
        documents_unlinked=int(linked["meeting_id"].astype(str).str.len().eq(0).sum()),
        documents_missing_dates=int(linked["date_parse_failed"].sum()),
        duplicate_meeting_ids=int(meetings["meeting_id"].duplicated().sum()) if not meetings.empty else 0,
        banks_with_no_meeting_anchors=sorted(bank for bank in all_banks if bank and bank not in anchor_banks),
    )
    print_validation(stats)
    print(f"output folder: {output_dir}")
    return stats


def main() -> None:
    args = parse_args()
    run(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        speech_window_days=args.speech_window_days,
        min_date=args.min_date,
        max_date=args.max_date,
    )


if __name__ == "__main__":
    main()
