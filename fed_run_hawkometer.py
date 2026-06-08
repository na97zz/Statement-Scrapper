from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.hawkometer.classifier import classify_score
from src.hawkometer.scorer import HawkometerScorer


PRESERVED_COLUMNS = [
    "doc_id",
    "bank",
    "date_au",
    "speaker",
    "title",
    "category",
    "document_subtype",
    "meeting_id",
    "text_file",
    "source_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FED-only Hawkometer scoring using the existing scorer.")
    parser.add_argument("--input", default="processed/fed/fed_master_index.csv", help="FED Phase 1 master index.")
    parser.add_argument(
        "--meeting-docs",
        default="processed/fed/fed_meeting_documents.csv",
        help="FED Phase 1 meeting documents file.",
    )
    parser.add_argument("--phrase-library", default="hawkometer.json", help="Path to Hawkometer phrase library.")
    parser.add_argument("--output", default="processed/fed/hawkometer", help="Output folder.")
    return parser.parse_args()


def resolve_phrase_library(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    if path.name == "hawkometer.json":
        fallback = Path("data") / "hawkometer.json"
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Phrase library not found: {path_text}")


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def normalize_input(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = ensure_columns(
        df,
        [
            "doc_id",
            "bank",
            "date_au",
            "speaker",
            "title",
            "category",
            "document_subtype",
            "meeting_id",
            "text_file",
            "text_file_path",
            "source_url",
        ],
    )
    if df["text_file"].astype(str).str.strip().eq("").all():
        df["text_file"] = df["text_file_path"]
    non_fed = int((~df["bank"].astype(str).str.upper().eq("FED")).sum())
    fed = df[df["bank"].astype(str).str.upper().eq("FED")].copy()
    fed["bank"] = "FED"
    for column in PRESERVED_COLUMNS:
        fed[column] = fed[column].fillna("").astype(str)
    return fed.reset_index(drop=True), non_fed


def read_text_file(path_text: str, root: Path) -> tuple[str, str]:
    if not path_text:
        return "", "text_file missing"
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return "", f"text file not found: {path}"
    try:
        return path.read_text(encoding="utf-8", errors="replace"), ""
    except OSError as exc:
        return "", str(exc)


def score_documents(fed: pd.DataFrame, scorer: HawkometerScorer, root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in fed.iterrows():
        base = {column: row.get(column, "") for column in PRESERVED_COLUMNS}
        text, error = read_text_file(base["text_file"], root)
        if error:
            rows.append(
                {
                    **base,
                    "score": 0.0,
                    "classification": "Neutral",
                    "confidence": 0.0,
                    "lean": "none",
                    "hawkish_score": 0.0,
                    "dovish_score": 0.0,
                    "raw_score": 0.0,
                    "matched_phrase_count": 0,
                    "total_phrase_matches": 0,
                    "matched_phrases_json": "[]",
                    "scoring_status": "missing_text",
                    "scoring_error": error,
                }
            )
            continue

        result = scorer.score(text, central_bank="FED")
        total_matches = sum(int(match.get("count", 0)) for match in result["matched_phrases"])
        rows.append(
            {
                **base,
                "score": result["score"],
                "classification": result["classification"],
                "confidence": result["confidence"],
                "lean": result["lean"],
                "hawkish_score": result["hawkish_score"],
                "dovish_score": result["dovish_score"],
                "raw_score": result["raw_score"],
                "matched_phrase_count": len(result["matched_phrases"]),
                "total_phrase_matches": total_matches,
                "matched_phrases_json": json.dumps(result["matched_phrases"], ensure_ascii=False),
                "scoring_status": "scored",
                "scoring_error": "",
            }
        )
    return pd.DataFrame(rows)


def build_speaker_scores(document_scores: pd.DataFrame) -> pd.DataFrame:
    scored = document_scores[
        document_scores["scoring_status"].eq("scored")
        & document_scores["speaker"].astype(str).str.strip().ne("")
    ].copy()
    if scored.empty:
        return pd.DataFrame(
            columns=[
                "bank",
                "speaker",
                "documents_scored",
                "average_score",
                "average_confidence",
                "latest_date",
                "latest_score",
                "most_hawkish_score",
                "most_dovish_score",
            ]
        )
    scored["date"] = pd.to_datetime(scored["date_au"], errors="coerce")
    rows = []
    for (bank, speaker), group in scored.groupby(["bank", "speaker"]):
        group = group.sort_values("date")
        latest = group.iloc[-1]
        rows.append(
            {
                "bank": bank,
                "speaker": speaker,
                "documents_scored": len(group),
                "average_score": group["score"].mean(),
                "average_confidence": group["confidence"].mean(),
                "latest_date": latest["date_au"],
                "latest_score": latest["score"],
                "most_hawkish_score": group["score"].max(),
                "most_dovish_score": group["score"].min(),
            }
        )
    return pd.DataFrame(rows).sort_values(["average_score", "documents_scored"], ascending=[False, False])


def component_average(group: pd.DataFrame, subtype: str) -> float | None:
    subset = group[group["document_subtype"].eq(subtype)]
    if subset.empty:
        return None
    return float(subset["score"].mean())


def weighted_meeting_score(components: dict[str, tuple[float, float | None]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for weight, value in components.values():
        if value is None:
            continue
        numerator += weight * value
        denominator += weight
    return numerator / denominator if denominator else 0.0


def build_meeting_scores(document_scores: pd.DataFrame, meeting_docs: pd.DataFrame) -> pd.DataFrame:
    scored = document_scores[
        document_scores["scoring_status"].eq("scored")
        & document_scores["meeting_id"].astype(str).str.strip().ne("")
    ].copy()
    if scored.empty:
        return pd.DataFrame()

    meeting_docs = ensure_columns(meeting_docs, ["meeting_id", "meeting_date"])
    meeting_dates = meeting_docs[["meeting_id", "meeting_date"]].drop_duplicates("meeting_id")
    rows = []
    for meeting_id, group in scored.groupby("meeting_id"):
        statement = component_average(group, "statement")
        minutes = component_average(group, "minutes")
        speech = component_average(group, "speech")
        press = component_average(group, "press_conference")
        sep = component_average(group, "sep")
        meeting_score = weighted_meeting_score(
            {
                "statement": (0.50, statement),
                "minutes": (0.25, minutes),
                "press_conference": (0.15, press),
                "sep": (0.05, sep),
                "speech": (0.05, speech),
            }
        )
        meeting_classification = classify_score(meeting_score)
        date_row = meeting_dates[meeting_dates["meeting_id"].eq(meeting_id)]
        meeting_date = date_row["meeting_date"].iloc[0] if not date_row.empty else group["date_au"].min()
        rows.append(
            {
                "meeting_id": meeting_id,
                "bank": "FED",
                "meeting_date": meeting_date,
                "meeting_score": meeting_score,
                "classification": meeting_classification["classification"],
                "lean": meeting_classification["lean"],
                "average_document_score": group["score"].mean(),
                "average_confidence": group["confidence"].mean(),
                "statement_score": "" if statement is None else statement,
                "minutes_score": "" if minutes is None else minutes,
                "press_conference_score": "" if press is None else press,
                "sep_score": "" if sep is None else sep,
                "speech_avg_score": "" if speech is None else speech,
                "documents_scored": len(group),
                "zero_match_documents": int(group["total_phrase_matches"].eq(0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("meeting_date")


def document_extreme(scores: pd.DataFrame, ascending: bool) -> dict[str, Any]:
    scored = scores[scores["scoring_status"].eq("scored")].copy()
    if scored.empty:
        return {}
    row = scored.sort_values("score", ascending=ascending).iloc[0]
    return {
        "doc_id": row["doc_id"],
        "date_au": row["date_au"],
        "title": row["title"],
        "category": row["category"],
        "score": float(row["score"]),
        "classification": row["classification"],
    }


def meeting_extreme(scores: pd.DataFrame, ascending: bool) -> dict[str, Any]:
    if scores.empty:
        return {}
    row = scores.sort_values("meeting_score", ascending=ascending).iloc[0]
    return {
        "meeting_id": row["meeting_id"],
        "meeting_date": row["meeting_date"],
        "meeting_score": float(row["meeting_score"]),
        "documents_scored": int(row["documents_scored"]),
    }


def build_summary(document_scores: pd.DataFrame, speaker_scores: pd.DataFrame, meeting_scores: pd.DataFrame) -> pd.DataFrame:
    scored = document_scores[document_scores["scoring_status"].eq("scored")]
    rows = [
        ("fed_rows_loaded", len(document_scores)),
        ("fed_rows_scored", len(scored)),
        ("missing_text_files", int(document_scores["scoring_status"].eq("missing_text").sum())),
        ("zero_match_documents", int(scored["total_phrase_matches"].eq(0).sum())),
        ("speakers_scored", len(speaker_scores)),
        ("meetings_scored", len(meeting_scores)),
        ("average_score", scored["score"].mean() if not scored.empty else 0.0),
        ("most_hawkish_fed_document", json.dumps(document_extreme(document_scores, ascending=False), ensure_ascii=False)),
        ("most_dovish_fed_document", json.dumps(document_extreme(document_scores, ascending=True), ensure_ascii=False)),
        ("most_hawkish_fed_meeting", json.dumps(meeting_extreme(meeting_scores, ascending=False), ensure_ascii=False)),
        ("most_dovish_fed_meeting", json.dumps(meeting_extreme(meeting_scores, ascending=True), ensure_ascii=False)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def write_json_output(
    path: Path,
    document_scores: pd.DataFrame,
    speaker_scores: pd.DataFrame,
    meeting_scores: pd.DataFrame,
    summary: pd.DataFrame,
    phrase_library_path: Path,
) -> None:
    def records(df: pd.DataFrame) -> list[dict[str, Any]]:
        return json.loads(df.to_json(orient="records"))

    def summary_value(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        return value

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bank": "FED",
        "phrase_library": str(phrase_library_path),
        "document_scores": records(document_scores),
        "speaker_scores": records(speaker_scores),
        "meeting_scores": records(meeting_scores),
        "summary": {row["metric"]: summary_value(row["value"]) for _, row in summary.iterrows()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(output: Path) -> None:
    text = """# FED Hawkometer Scores

This folder contains FED-only Hawkometer outputs built from `processed/fed/fed_master_index.csv` using the existing `src.hawkometer.scorer.HawkometerScorer` logic and local phrase library.

## Run

```powershell
python fed_run_hawkometer.py --input processed/fed/fed_master_index.csv --meeting-docs processed/fed/fed_meeting_documents.csv --phrase-library hawkometer.json --output processed/fed/hawkometer
```

If `hawkometer.json` is not present at the project root, the runner automatically uses `data/hawkometer.json`.

## Outputs

- `fed_document_scores.csv`: one row per FED document.
- `fed_speaker_scores.csv`: speaker-level averages.
- `fed_meeting_scores.csv`: meeting-level weighted scores.
- `fed_hawkometer_summary.csv`: key run summary.
- `fed_hawkometer.json`: JSON bundle of all outputs.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def run(input_path: Path, meeting_docs_path: Path, phrase_library: Path, output: Path) -> None:
    phrase_path = resolve_phrase_library(str(phrase_library))
    scorer = HawkometerScorer(phrase_path)
    master = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    fed, non_fed = normalize_input(master)
    if non_fed:
        print(f"Warning: filtered out {non_fed} non-FED rows.")
    meeting_docs = pd.read_csv(meeting_docs_path, dtype=str, keep_default_na=False) if meeting_docs_path.exists() else pd.DataFrame()
    if not meeting_docs.empty and "bank" in meeting_docs.columns:
        non_fed_meeting = int((~meeting_docs["bank"].astype(str).str.upper().eq("FED")).sum())
        if non_fed_meeting:
            print(f"Warning: filtered out {non_fed_meeting} non-FED meeting document rows.")
        meeting_docs = meeting_docs[meeting_docs["bank"].astype(str).str.upper().eq("FED")].copy()

    output.mkdir(parents=True, exist_ok=True)
    document_scores = score_documents(fed, scorer, Path.cwd())
    speaker_scores = build_speaker_scores(document_scores)
    meeting_scores = build_meeting_scores(document_scores, meeting_docs)
    summary = build_summary(document_scores, speaker_scores, meeting_scores)

    document_scores.to_csv(output / "fed_document_scores.csv", index=False)
    speaker_scores.to_csv(output / "fed_speaker_scores.csv", index=False)
    meeting_scores.to_csv(output / "fed_meeting_scores.csv", index=False)
    summary.to_csv(output / "fed_hawkometer_summary.csv", index=False)
    write_json_output(output / "fed_hawkometer.json", document_scores, speaker_scores, meeting_scores, summary, phrase_path)
    write_readme(output)

    scored = document_scores[document_scores["scoring_status"].eq("scored")]
    print(f"FED rows loaded: {len(fed)}")
    print(f"FED rows scored: {len(scored)}")
    print(f"missing text files: {int(document_scores['scoring_status'].eq('missing_text').sum())}")
    print(f"zero-match documents: {int(scored['total_phrase_matches'].eq(0).sum())}")
    print(f"speakers scored: {len(speaker_scores)}")
    print(f"meetings scored: {len(meeting_scores)}")
    print(f"average score: {(scored['score'].mean() if not scored.empty else 0.0):.3f}")
    print(f"most hawkish FED document: {document_extreme(document_scores, ascending=False)}")
    print(f"most dovish FED document: {document_extreme(document_scores, ascending=True)}")
    print(f"most hawkish FED meeting: {meeting_extreme(meeting_scores, ascending=False)}")
    print(f"most dovish FED meeting: {meeting_extreme(meeting_scores, ascending=True)}")
    print(f"output folder: {output}")


def main() -> None:
    args = parse_args()
    run(Path(args.input), Path(args.meeting_docs), Path(args.phrase_library), Path(args.output))


if __name__ == "__main__":
    main()
