from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHODOLOGY_VERSION = "phrase_hawkometer_v1"

DEFAULT_PHRASE_LIBRARY = {
    "hawkish": {
        "inflation persistent": 2.5,
        "inflation remains elevated": 2.2,
        "further tightening": 2.4,
        "additional tightening": 2.4,
        "more work to do": 2.0,
        "premature to declare victory": 2.0,
        "not the time to cut": 1.9,
        "upside risks to inflation": 1.8,
        "inflation expectations rising": 1.8,
        "policy must remain restrictive": 1.7,
        "sticky inflation": 1.6,
        "overheating": 1.6,
        "restrictive stance is appropriate": 1.5,
        "vigilance": 1.4,
        "vigilant": 1.4,
        "hawkish hold": 1.4,
    },
    "dovish": {
        "appropriate to begin easing": -2.4,
        "scope to ease": -2.2,
        "disinflation is well advanced": -2.0,
        "conditions for easing": -2.0,
        "rate cuts on the table": -2.0,
        "dovish pivot": -2.0,
        "approaching neutral": -1.8,
        "close to neutral": -1.6,
        "policy is sufficiently restrictive": -1.6,
        "disinflation continues": -1.5,
        "labour market cooling": -1.4,
        "labor market cooling": -1.4,
        "recession risk": -1.4,
        "downside risks": -1.4,
        "financial conditions have tightened": -1.2,
        "growth is slowing": -1.2,
        "data dependent": -0.7,
        "data-dependent": -0.7,
    },
}

REQUIRED_COLUMNS = [
    "doc_id",
    "bank",
    "category",
    "document_subtype",
    "date_au",
    "speaker",
    "title",
    "meeting_id",
    "text_file",
    "source_url",
]

DOCUMENT_SCORE_COLUMNS = [
    "doc_id",
    "bank",
    "category",
    "document_subtype",
    "date_au",
    "speaker",
    "title",
    "meeting_id",
    "text_file",
    "source_url",
    "hawkish_score",
    "dovish_score",
    "raw_score",
    "clipped_score",
    "classification",
    "total_phrase_matches",
    "matched_hawkish_phrases",
    "matched_dovish_phrases",
    "scoring_status",
    "scoring_error",
]

SPEAKER_ROLLING_COLUMNS = [
    "bank",
    "speaker",
    "date_au",
    "doc_id",
    "clipped_score",
    "rolling_90d_score",
    "rolling_30d_score",
    "prior_60d_score",
    "shift_30_vs_prior_60",
    "shift_direction",
]

BANK_COMMITTEE_COLUMNS = [
    "bank",
    "date_au",
    "committee_score",
    "committee_lean",
    "number_of_speakers",
    "number_of_documents",
    "average_document_score",
]

MEETING_SCORE_COLUMNS = [
    "meeting_id",
    "bank",
    "meeting_date",
    "statement_score",
    "minutes_score",
    "press_conference_score",
    "speech_avg_score",
    "meeting_cycle_score",
    "meeting_cycle_lean",
    "documents_scored",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic phrase-based Hawkometer scoring.")
    parser.add_argument(
        "--input",
        default="processed/meeting_linkage/updated_master_index.csv",
        help="Input updated master index CSV.",
    )
    parser.add_argument(
        "--output",
        default="processed/hawkometer",
        help="Output folder for Hawkometer CSV/JSON files.",
    )
    parser.add_argument("--phrase-library", default="", help="Optional phrase_library.json path.")
    parser.add_argument("--start-date", default="2021-01-01", help="Start date filter.")
    parser.add_argument("--end-date", default="2026-12-31", help="End date filter.")
    parser.add_argument(
        "--document-types",
        nargs="*",
        default=None,
        help="Optional list of document_subtype values to score.",
    )
    parser.add_argument(
        "--min-phrase-matches",
        type=int,
        default=0,
        help="Minimum phrase matches required for rows included in aggregate outputs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print scoring progress every N documents. Use 0 to disable.",
    )
    return parser.parse_args()


def load_phrase_library(path: str | Path | None) -> dict[str, dict[str, float]]:
    if not path:
        return DEFAULT_PHRASE_LIBRARY
    phrase_path = Path(path)
    with phrase_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    library = {"hawkish": {}, "dovish": {}}
    for side in library:
        phrases = data.get(side, {})
        if not isinstance(phrases, dict):
            raise ValueError(f"Phrase library key '{side}' must be an object of phrase: weight pairs.")
        library[side] = {str(phrase).lower(): float(weight) for phrase, weight in phrases.items()}
    return library


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    if "text_file_path" in df.columns:
        df["text_file"] = df["text_file"].where(df["text_file"].astype(str).str.len() > 0, df["text_file_path"])

    for column in REQUIRED_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()

    if df["doc_id"].eq("").any():
        missing = df["doc_id"].eq("")
        df.loc[missing, "doc_id"] = [
            f"DOC_{i:07d}_{safe_id_part(row.bank)}_{safe_id_part(row.category)}_{safe_id_part(row.date_au)}"
            for i, row in df[missing].reset_index(drop=False).iterrows()
        ]
    return df


def safe_id_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_")
    return cleaned[:24] or "NA"


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date_au"], errors="coerce", dayfirst=False)
    needs_dayfirst = df["date"].isna()
    if needs_dayfirst.any():
        df.loc[needs_dayfirst, "date"] = pd.to_datetime(
            df.loc[needs_dayfirst, "date_au"], errors="coerce", dayfirst=True
        )
    df["date_au"] = df["date"].dt.strftime("%Y-%m-%d")
    df.loc[df["date"].isna(), "date_au"] = ""
    return df


def filter_rows(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    document_types: list[str] | None,
) -> pd.DataFrame:
    df = df.copy()
    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise ValueError("--start-date and --end-date must be valid dates.")
    df = df[df["date"].between(start, end, inclusive="both")]
    if document_types:
        wanted = {value.lower() for value in document_types}
        df = df[df["document_subtype"].str.lower().isin(wanted)]
    return df.reset_index(drop=True)


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    parts = re.split(r"[\s-]+", phrase.strip().lower())
    escaped = [re.escape(part) for part in parts if part]
    # Spaces and hyphens vary across PDFs; allow either between phrase tokens.
    body = r"[\s\-]+".join(escaped)
    return re.compile(rf"(?<!\w){body}(?!\w)", flags=re.IGNORECASE)


def compile_phrase_library(library: dict[str, dict[str, float]]) -> dict[str, list[tuple[str, float, re.Pattern[str]]]]:
    return {
        side: [(phrase, float(weight), phrase_pattern(phrase)) for phrase, weight in phrases.items()]
        for side, phrases in library.items()
    }


def resolve_text_path(text_file: str, base_dir: Path) -> Path:
    path = Path(str(text_file or ""))
    if path.is_absolute():
        return path
    return base_dir / path


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
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return sorted(candidates, key=lambda item: (len(item.name), item.name))[0]
    return None


def read_text(text_file: str, base_dir: Path) -> tuple[str, str, str]:
    if not text_file:
        return "", "text_file path is blank", ""
    path = resolve_text_path(text_file, base_dir)
    recovered_path = ""
    if not path.exists():
        alternate = find_alternate_text_path(path)
        if alternate is None:
            return "", f"text file missing: {path}", ""
        path = alternate
        recovered_path = str(path.relative_to(base_dir)) if path.is_relative_to(base_dir) else str(path)
    try:
        return path.read_text(encoding="utf-8", errors="replace"), "", recovered_path
    except OSError as exc:
        return "", str(exc), recovered_path


def scaled_count(count: int) -> float:
    return math.log1p(count) / math.log(2)


def score_text(text: str, compiled_library: dict[str, list[tuple[str, float, re.Pattern[str]]]]) -> dict[str, Any]:
    lower_text = text.lower()
    matched: dict[str, dict[str, Any]] = {"hawkish": {}, "dovish": {}}
    side_scores = {"hawkish": 0.0, "dovish": 0.0}
    total_matches = 0

    for side, phrases in compiled_library.items():
        for phrase, weight, pattern in phrases:
            count = len(pattern.findall(lower_text))
            if count <= 0:
                continue
            contribution = weight * scaled_count(count)
            side_scores[side] += contribution
            total_matches += count
            matched[side][phrase] = {
                "count": int(count),
                "weight": float(weight),
                "contribution": round(float(contribution), 6),
            }

    raw_score = side_scores["hawkish"] + side_scores["dovish"]
    clipped_score = float(np.clip(raw_score, -10, 10))
    return {
        "hawkish_score": float(side_scores["hawkish"]),
        "dovish_score": float(side_scores["dovish"]),
        "raw_score": float(raw_score),
        "clipped_score": clipped_score,
        "classification": classify_score(clipped_score),
        "total_phrase_matches": int(total_matches),
        "matched_hawkish_phrases": json.dumps(matched["hawkish"], ensure_ascii=False, sort_keys=True),
        "matched_dovish_phrases": json.dumps(matched["dovish"], ensure_ascii=False, sort_keys=True),
    }


def classify_score(score: float) -> str:
    if score >= 2.5:
        return "hawkish"
    if score >= 1.0:
        return "leaning hawkish"
    if score <= -2.5:
        return "dovish"
    if score <= -1.0:
        return "leaning dovish"
    return "neutral"


def score_documents(
    df: pd.DataFrame,
    compiled_library: dict[str, list[tuple[str, float, re.Pattern[str]]]],
    base_dir: Path,
    progress_every: int = 100,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(df)
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        if progress_every and (position == 1 or position % progress_every == 0 or position == total):
            print(f"scoring documents: {position}/{total}")
        output = {column: row.get(column, "") for column in DOCUMENT_SCORE_COLUMNS}
        output.update(
            {
                "hawkish_score": 0.0,
                "dovish_score": 0.0,
                "raw_score": 0.0,
                "clipped_score": 0.0,
                "classification": "neutral",
                "total_phrase_matches": 0,
                "matched_hawkish_phrases": "{}",
                "matched_dovish_phrases": "{}",
                "scoring_status": "scored",
                "scoring_error": "",
            }
        )

        text, error, recovered_path = read_text(row.get("text_file", ""), base_dir)
        if recovered_path:
            output["text_file"] = recovered_path
        if error:
            output["scoring_status"] = "missing_text"
            output["scoring_error"] = error
            rows.append(output)
            continue

        try:
            output.update(score_text(text, compiled_library))
        except Exception as exc:  # Keep scoring robust across malformed text.
            output["scoring_status"] = "error"
            output["scoring_error"] = str(exc)
        rows.append(output)

    scores = pd.DataFrame(rows, columns=DOCUMENT_SCORE_COLUMNS)
    numeric_columns = [
        "hawkish_score",
        "dovish_score",
        "raw_score",
        "clipped_score",
        "total_phrase_matches",
    ]
    for column in numeric_columns:
        scores[column] = pd.to_numeric(scores[column], errors="coerce").fillna(0)
    return scores


def aggregate_ready(document_scores: pd.DataFrame, min_phrase_matches: int) -> pd.DataFrame:
    ready = document_scores[document_scores["scoring_status"].eq("scored")].copy()
    ready = ready[ready["total_phrase_matches"] >= min_phrase_matches].copy()
    ready["date"] = pd.to_datetime(ready["date_au"], errors="coerce")
    return ready.dropna(subset=["date"])


def rolling_time_average(group: pd.DataFrame, window: str) -> pd.Series:
    group = group.sort_values("date")
    series = group.set_index("date")["clipped_score"].astype(float)
    rolled = series.rolling(window, min_periods=1).mean()
    return pd.Series(rolled.to_numpy(), index=group.index)


def prior_window_average(group: pd.DataFrame, prior_days: int = 60, offset_days: int = 30) -> pd.Series:
    group = group.sort_values("date")
    values = []
    for _, row in group.iterrows():
        end = row["date"] - pd.Timedelta(days=offset_days)
        start = row["date"] - pd.Timedelta(days=offset_days + prior_days)
        prior = group[(group["date"] >= start) & (group["date"] < end)]
        values.append(float(prior["clipped_score"].mean()) if not prior.empty else np.nan)
    return pd.Series(values, index=group.index)


def create_speaker_rolling_scores(document_scores: pd.DataFrame, min_phrase_matches: int) -> pd.DataFrame:
    ready = aggregate_ready(document_scores, min_phrase_matches)
    ready = ready[ready["speaker"].astype(str).str.strip().ne("")].copy()
    if ready.empty:
        return pd.DataFrame(columns=SPEAKER_ROLLING_COLUMNS)

    ready = ready.sort_values(["bank", "speaker", "date", "doc_id"])
    ready["rolling_90d_score"] = ready.groupby(["bank", "speaker"], group_keys=False).apply(
        lambda group: rolling_time_average(group, "90D")
    )
    ready["rolling_30d_score"] = ready.groupby(["bank", "speaker"], group_keys=False).apply(
        lambda group: rolling_time_average(group, "30D")
    )
    ready["prior_60d_score"] = ready.groupby(["bank", "speaker"], group_keys=False).apply(prior_window_average)
    ready["shift_30_vs_prior_60"] = ready["rolling_30d_score"] - ready["prior_60d_score"]
    ready["shift_direction"] = ready["shift_30_vs_prior_60"].apply(classify_shift)

    out = ready[SPEAKER_ROLLING_COLUMNS].copy()
    return out.sort_values(["bank", "speaker", "date_au", "doc_id"])


def classify_shift(value: float) -> str:
    if pd.isna(value):
        return "stable"
    if value >= 1.0:
        return "hawkish shift"
    if value <= -1.0:
        return "dovish shift"
    return "stable"


def voter_weight(value: str) -> float:
    normalized = str(value or "").strip().lower()
    if normalized == "voter":
        return 1.0
    if normalized in {"non_voter", "non-voter", "observer"}:
        return 0.55
    return 1.0


def create_bank_committee_scores(document_scores: pd.DataFrame, speaker_rolling: pd.DataFrame) -> pd.DataFrame:
    ready = aggregate_ready(document_scores, min_phrase_matches=0)
    if ready.empty:
        return pd.DataFrame(columns=BANK_COMMITTEE_COLUMNS)

    voter_lookup = pd.DataFrame()
    if "voter_status" in document_scores.columns:
        voter_lookup = document_scores[["bank", "speaker", "voter_status"]].drop_duplicates()

    rows: list[dict[str, Any]] = []
    if speaker_rolling.empty:
        speaker_rolling_dt = pd.DataFrame(columns=list(speaker_rolling.columns) + ["date"])
    else:
        speaker_rolling_dt = speaker_rolling.copy()
        speaker_rolling_dt["date"] = pd.to_datetime(speaker_rolling_dt["date_au"], errors="coerce")

    for (bank, date), day_docs in ready.groupby(["bank", "date"]):
        prior = speaker_rolling_dt[(speaker_rolling_dt["bank"].eq(bank)) & (speaker_rolling_dt["date"] <= date)]
        latest = pd.DataFrame()
        if not prior.empty:
            latest = prior.sort_values(["speaker", "date"]).groupby("speaker", as_index=False).tail(1)
            if not voter_lookup.empty:
                latest = latest.merge(voter_lookup[voter_lookup["bank"].eq(bank)], on=["bank", "speaker"], how="left")
            else:
                latest["voter_status"] = ""
            latest["weight"] = latest["voter_status"].apply(voter_weight)
            committee_score = float(np.average(latest["rolling_90d_score"].astype(float), weights=latest["weight"]))
        else:
            committee_score = float(day_docs["clipped_score"].mean())

        rows.append(
            {
                "bank": bank,
                "date_au": date.strftime("%Y-%m-%d"),
                "committee_score": committee_score,
                "committee_lean": classify_score(committee_score),
                "number_of_speakers": int(latest["speaker"].nunique()) if not latest.empty else 0,
                "number_of_documents": int(len(day_docs)),
                "average_document_score": float(day_docs["clipped_score"].mean()),
            }
        )

    return pd.DataFrame(rows, columns=BANK_COMMITTEE_COLUMNS).sort_values(["bank", "date_au"])


def weighted_available_score(components: dict[str, tuple[float, float | None]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for weight, value in components.values():
        if value is None or pd.isna(value):
            continue
        numerator += weight * float(value)
        denominator += weight
    return numerator / denominator if denominator else 0.0


def create_meeting_scores(document_scores: pd.DataFrame) -> pd.DataFrame:
    ready = document_scores[
        document_scores["scoring_status"].eq("scored") & document_scores["meeting_id"].astype(str).str.strip().ne("")
    ].copy()
    if ready.empty:
        return pd.DataFrame(columns=MEETING_SCORE_COLUMNS)

    ready["meeting_date"] = pd.to_datetime(ready.get("meeting_date", ready["date_au"]), errors="coerce")
    ready["date"] = pd.to_datetime(ready["date_au"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (meeting_id, bank), group in ready.groupby(["meeting_id", "bank"]):
        statement = mean_for_subtypes(group, ["statement"])
        minutes = mean_for_subtypes(group, ["minutes"])
        press_conference = mean_for_subtypes(group, ["press_conference"])
        speech = mean_for_subtypes(group, ["speech"])
        meeting_cycle_score = weighted_available_score(
            {
                "statement": (0.45, statement),
                "minutes": (0.25, minutes),
                "press_conference": (0.20, press_conference),
                "speeches": (0.10, speech),
            }
        )
        meeting_date = group["meeting_date"].dropna().min()
        if pd.isna(meeting_date):
            meeting_date = group["date"].dropna().min()
        rows.append(
            {
                "meeting_id": meeting_id,
                "bank": bank,
                "meeting_date": meeting_date.strftime("%Y-%m-%d") if not pd.isna(meeting_date) else "",
                "statement_score": blank_if_none(statement),
                "minutes_score": blank_if_none(minutes),
                "press_conference_score": blank_if_none(press_conference),
                "speech_avg_score": blank_if_none(speech),
                "meeting_cycle_score": meeting_cycle_score,
                "meeting_cycle_lean": classify_score(meeting_cycle_score),
                "documents_scored": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=MEETING_SCORE_COLUMNS).sort_values(["bank", "meeting_date", "meeting_id"])


def mean_for_subtypes(group: pd.DataFrame, subtypes: list[str]) -> float | None:
    subset = group[group["document_subtype"].isin(subtypes)]
    if subset.empty:
        return None
    return float(subset["clipped_score"].mean())


def blank_if_none(value: float | None) -> float | str:
    if value is None or pd.isna(value):
        return ""
    return float(value)


def create_summary(document_scores: pd.DataFrame, total_rows_loaded: int, rows_after_filters: int) -> pd.DataFrame:
    scored = document_scores[document_scores["scoring_status"].eq("scored")].copy()
    missing = document_scores[document_scores["scoring_status"].eq("missing_text")]
    zero_match = scored[scored["total_phrase_matches"].eq(0)]

    avg_bank = scored.groupby("bank")["clipped_score"].mean().round(6).to_dict()
    avg_subtype = scored.groupby("document_subtype")["clipped_score"].mean().round(6).to_dict()
    most_hawkish = document_extreme(scored, ascending=False)
    most_dovish = document_extreme(scored, ascending=True)

    rows = [
        {"metric": "total_documents_loaded", "value": str(total_rows_loaded)},
        {"metric": "total_documents_after_filters", "value": str(rows_after_filters)},
        {"metric": "documents_filtered_out", "value": str(total_rows_loaded - rows_after_filters)},
        {"metric": "total_documents_scored", "value": str(len(scored))},
        {"metric": "documents_missing_text", "value": str(len(missing))},
        {"metric": "documents_with_zero_phrase_matches", "value": str(len(zero_match))},
        {"metric": "average_score_by_bank", "value": json.dumps(avg_bank, sort_keys=True)},
        {"metric": "average_score_by_document_subtype", "value": json.dumps(avg_subtype, sort_keys=True)},
        {"metric": "most_hawkish_document", "value": json.dumps(most_hawkish, ensure_ascii=False, sort_keys=True)},
        {"metric": "most_dovish_document", "value": json.dumps(most_dovish, ensure_ascii=False, sort_keys=True)},
    ]
    return pd.DataFrame(rows)


def document_extreme(scored: pd.DataFrame, ascending: bool) -> dict[str, Any]:
    if scored.empty:
        return {}
    row = scored.sort_values("clipped_score", ascending=ascending).iloc[0]
    return {
        "doc_id": row["doc_id"],
        "bank": row["bank"],
        "date_au": row["date_au"],
        "title": row["title"],
        "score": float(row["clipped_score"]),
        "classification": row["classification"],
    }


def write_json_output(
    output_path: Path,
    phrase_library: dict[str, dict[str, float]],
    document_scores: pd.DataFrame,
    speaker_rolling: pd.DataFrame,
    committee_scores: pd.DataFrame,
    meeting_scores: pd.DataFrame,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology_version": METHODOLOGY_VERSION,
        "phrase_library": phrase_library,
        "document_scores": frame_records(document_scores),
        "speaker_rolling_scores": frame_records(speaker_rolling),
        "bank_committee_scores": frame_records(committee_scores),
        "meeting_scores": frame_records(meeting_scores),
        "methodology_notes": [
            "Phrase-based deterministic scorer only; no LLM is used.",
            "Phrase counts are scaled with log1p(count) / log(2).",
            "Document scores are clipped to the range [-10, 10].",
            "Meeting scores re-normalise component weights when components are missing.",
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def frame_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


def write_readme(output_dir: Path) -> None:
    readme = """# Phrase-Based Hawkometer

This folder contains Phase 2 Hawkometer outputs. This phase is fully deterministic and does not use any local or remote LLM.

## What It Measures

The Hawkometer measures the balance of hawkish and dovish language in central bank documents using a fixed phrase library. Positive scores indicate tighter or more inflation-concerned language. Negative scores indicate easier or more growth/labour-market-concerned language.

## What It Does Not Measure

It does not forecast policy decisions, infer causality, understand strategy, or replace expert reading. It is a reproducible language signal designed to support later analysis.

## Phrase Scoring

Each phrase has a signed weight. Hawkish phrases are positive and dovish phrases are negative. Counts are scaled as `log1p(count) / log(2)`, then summed and clipped between -10 and +10.

Classification thresholds:

- `>= 2.5`: hawkish
- `1.0` to `< 2.5`: leaning hawkish
- `>-1.0` to `< 1.0`: neutral
- `>-2.5` to `<= -1.0`: leaning dovish
- `<= -2.5`: dovish

## Speaker Rolling Scores

Speaker scores are calculated for documents with a speaker name. The script calculates 90-day rolling average, 30-day rolling average, prior 60-day average, and the shift between recent and prior tone.

## Committee Scores

Committee scores are bank-date aggregates using the latest available 90-day rolling score per speaker. If `voter_status` is present, voters receive weight 1.0 and non-voters/observers receive 0.55.

## Meeting Scores

Meeting scores combine linked documents from the meeting-cycle layer:

- statement: 45%
- minutes: 25%
- press conference: 20%
- speeches: 10%

If a component is missing, weights are re-normalised across available components.

## Known Limitations

- Context blindness
- Weak handling of negation
- Weak handling of hypotheticals
- Novel phrases are not detected
- Translation effects across central banks
- No causal prediction

## How To Run

```powershell
python hawkometer_scorer.py --input processed/meeting_linkage/updated_master_index.csv --output processed/hawkometer
```

Optional:

```powershell
python hawkometer_scorer.py --phrase-library phrase_library.json --start-date 2021-01-01 --end-date 2026-12-31 --document-types speech statement minutes press_conference --min-phrase-matches 0
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def print_validation(document_scores: pd.DataFrame, total_rows_loaded: int) -> None:
    scored = document_scores[document_scores["scoring_status"].eq("scored")]
    missing = document_scores[document_scores["scoring_status"].eq("missing_text")]
    skipped = document_scores[~document_scores["scoring_status"].eq("scored")]
    zero_match = scored[scored["total_phrase_matches"].eq(0)]
    print(f"rows loaded: {total_rows_loaded}")
    print(f"rows scored: {len(scored)}")
    print(f"rows skipped: {total_rows_loaded - len(scored)}")
    print(f"rows after filters: {len(document_scores)}")
    print(f"missing text files: {len(missing)}")
    print(f"zero-match documents: {len(zero_match)}")
    print("average score by bank:")
    if scored.empty:
        print("  none")
    else:
        for bank, value in scored.groupby("bank")["clipped_score"].mean().sort_index().items():
            print(f"  {bank}: {value:.3f}")
    print("most hawkish document:")
    print("  " + json.dumps(document_extreme(scored, ascending=False), ensure_ascii=False))
    print("most dovish document:")
    print("  " + json.dumps(document_extreme(scored, ascending=True), ensure_ascii=False))


def run(
    input_path: Path,
    output_dir: Path,
    phrase_library_path: str,
    start_date: str,
    end_date: str,
    document_types: list[str] | None,
    min_phrase_matches: int,
    progress_every: int,
) -> None:
    base_dir = Path.cwd()
    phrase_library = load_phrase_library(phrase_library_path)
    compiled_library = compile_phrase_library(phrase_library)

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    total_rows_loaded = len(df)
    df = normalize_columns(df)
    df = parse_dates(df)
    df = filter_rows(df, start_date=start_date, end_date=end_date, document_types=document_types)

    output_dir.mkdir(parents=True, exist_ok=True)
    document_scores = score_documents(
        df,
        compiled_library,
        base_dir=base_dir,
        progress_every=progress_every,
    )
    speaker_rolling = create_speaker_rolling_scores(document_scores, min_phrase_matches=min_phrase_matches)
    committee_scores = create_bank_committee_scores(document_scores, speaker_rolling)
    meeting_scores = create_meeting_scores(document_scores)
    summary = create_summary(document_scores, total_rows_loaded=total_rows_loaded, rows_after_filters=len(df))

    document_scores.to_csv(output_dir / "document_scores.csv", index=False)
    speaker_rolling.to_csv(output_dir / "speaker_rolling_scores.csv", index=False)
    committee_scores.to_csv(output_dir / "bank_committee_scores.csv", index=False)
    meeting_scores.to_csv(output_dir / "meeting_hawkometer_scores.csv", index=False)
    summary.to_csv(output_dir / "hawkometer_summary.csv", index=False)
    write_json_output(
        output_dir / "hawkometer.json",
        phrase_library,
        document_scores,
        speaker_rolling,
        committee_scores,
        meeting_scores,
    )
    write_readme(output_dir)
    print_validation(document_scores, total_rows_loaded=total_rows_loaded)
    print(f"output folder: {output_dir}")


def main() -> None:
    args = parse_args()
    run(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        phrase_library_path=args.phrase_library,
        start_date=args.start_date,
        end_date=args.end_date,
        document_types=args.document_types,
        min_phrase_matches=args.min_phrase_matches,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
