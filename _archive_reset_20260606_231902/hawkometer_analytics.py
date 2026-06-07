from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BANK_TRENDS_COLUMNS = [
    "bank",
    "date_au",
    "daily_avg_score",
    "rolling_30d_score",
    "rolling_90d_score",
    "rolling_180d_score",
    "momentum_30_vs_90",
    "momentum_classification",
    "documents_count",
]

SPEAKER_ALERT_COLUMNS = [
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
    "alert_strength",
]

MEETING_ANALYSIS_COLUMNS = [
    "meeting_id",
    "bank",
    "meeting_date",
    "meeting_cycle_score",
    "previous_meeting_score",
    "meeting_score_change",
    "rolling_3_meeting_score",
    "rolling_6_meeting_score",
    "meeting_cycle_lean",
    "meeting_shift_classification",
    "documents_scored",
]

CROSS_BANK_COLUMNS = [
    "rank",
    "bank",
    "latest_date",
    "latest_30d_score",
    "latest_90d_score",
    "latest_meeting_score",
    "momentum_30_vs_90",
    "overall_current_score",
    "current_lean",
]

DOCUMENT_BREAKDOWN_COLUMNS = [
    "bank",
    "document_subtype",
    "category",
    "document_count",
    "average_score",
    "median_score",
    "max_score",
    "min_score",
    "zero_match_count",
    "average_phrase_matches",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 3 Hawkometer analytics outputs.")
    parser.add_argument("--input", default="processed/hawkometer", help="Phase 2 Hawkometer output folder.")
    parser.add_argument(
        "--meeting-linkage",
        default="processed/meeting_linkage",
        help="Phase 1 meeting linkage output folder.",
    )
    parser.add_argument(
        "--output",
        default="processed/hawkometer_analytics",
        help="Output folder for Phase 3 analytics.",
    )
    parser.add_argument("--start-date", default="2021-01-01", help="Start date filter.")
    parser.add_argument("--end-date", default="2026-12-31", help="End date filter.")
    parser.add_argument("--shift-threshold", type=float, default=1.0, help="Shift alert threshold.")
    parser.add_argument(
        "--latest-window-days",
        type=int,
        default=120,
        help="Window for considering current/latest bank scores.",
    )
    return parser.parse_args()


def read_csv_safe(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        message = f"Warning: missing {'required' if required else 'optional'} file: {path}"
        print(message)
        if required:
            return pd.DataFrame()
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def parse_date_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    if column not in df.columns:
        df[column] = ""
    df["_date"] = pd.to_datetime(df[column], errors="coerce")
    return df


def filter_by_date(df: pd.DataFrame, date_column: str, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = parse_date_column(df, date_column)
    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise ValueError("--start-date and --end-date must be valid dates.")
    return df[df["_date"].between(start, end, inclusive="both")].copy()


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def classify_lean(score: float) -> str:
    if pd.isna(score):
        return "neutral"
    if score >= 2.5:
        return "hawkish"
    if score >= 1.0:
        return "leaning hawkish"
    if score <= -2.5:
        return "dovish"
    if score <= -1.0:
        return "leaning dovish"
    return "neutral"


def classify_momentum(value: float, threshold: float) -> str:
    if pd.isna(value):
        return "stable"
    if value >= threshold:
        return "hawkish acceleration"
    if value <= -threshold:
        return "dovish acceleration"
    return "stable"


def classify_shift(value: float, threshold: float) -> str:
    if pd.isna(value):
        return "stable"
    if value >= threshold:
        return "hawkish shift"
    if value <= -threshold:
        return "dovish shift"
    return "stable"


def alert_strength(value: float) -> str:
    absolute = abs(value)
    if absolute >= 3.5:
        return "extreme"
    if absolute >= 2.0:
        return "strong"
    if absolute >= 1.0:
        return "mild"
    return ""


def build_bank_trends(document_scores: pd.DataFrame, shift_threshold: float) -> pd.DataFrame:
    if document_scores.empty:
        return pd.DataFrame(columns=BANK_TRENDS_COLUMNS)

    df = ensure_columns(document_scores, ["bank", "date_au", "clipped_score", "scoring_status"])
    df = df[df["scoring_status"].eq("scored")].copy()
    if df.empty:
        return pd.DataFrame(columns=BANK_TRENDS_COLUMNS)
    df["_score"] = numeric(df, "clipped_score")
    df = parse_date_column(df, "date_au").dropna(subset=["_date", "_score"])

    daily = (
        df.groupby(["bank", "_date"], as_index=False)
        .agg(daily_avg_score=("_score", "mean"), documents_count=("_score", "size"))
        .sort_values(["bank", "_date"])
    )

    pieces = []
    for bank, group in daily.groupby("bank"):
        group = group.sort_values("_date").set_index("_date")
        group["rolling_30d_score"] = group["daily_avg_score"].rolling("30D", min_periods=1).mean()
        group["rolling_90d_score"] = group["daily_avg_score"].rolling("90D", min_periods=1).mean()
        group["rolling_180d_score"] = group["daily_avg_score"].rolling("180D", min_periods=1).mean()
        group["momentum_30_vs_90"] = group["rolling_30d_score"] - group["rolling_90d_score"]
        group["bank"] = bank
        pieces.append(group.reset_index())

    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=BANK_TRENDS_COLUMNS)
    out["date_au"] = out["_date"].dt.strftime("%Y-%m-%d")
    out["momentum_classification"] = out["momentum_30_vs_90"].apply(lambda value: classify_momentum(value, shift_threshold))
    return out[BANK_TRENDS_COLUMNS].sort_values(["bank", "date_au"])


def build_speaker_shift_alerts(speaker_rolling: pd.DataFrame, shift_threshold: float) -> pd.DataFrame:
    if speaker_rolling.empty:
        return pd.DataFrame(columns=SPEAKER_ALERT_COLUMNS)
    df = ensure_columns(speaker_rolling, SPEAKER_ALERT_COLUMNS[:-1])
    df["shift_30_vs_prior_60"] = numeric(df, "shift_30_vs_prior_60")
    alerts = df[df["shift_30_vs_prior_60"].abs() >= shift_threshold].copy()
    if alerts.empty:
        return pd.DataFrame(columns=SPEAKER_ALERT_COLUMNS)
    alerts["shift_direction"] = alerts["shift_30_vs_prior_60"].apply(lambda value: classify_shift(value, shift_threshold))
    alerts["alert_strength"] = alerts["shift_30_vs_prior_60"].apply(alert_strength)
    for column in ["clipped_score", "rolling_90d_score", "rolling_30d_score", "prior_60d_score"]:
        alerts[column] = numeric(alerts, column)
    return alerts[SPEAKER_ALERT_COLUMNS].sort_values(
        ["bank", "date_au", "alert_strength", "shift_30_vs_prior_60"],
        ascending=[True, True, True, False],
    )


def build_meeting_cycle_analysis(meeting_scores: pd.DataFrame, meeting_index: pd.DataFrame, shift_threshold: float) -> pd.DataFrame:
    if meeting_scores.empty:
        return pd.DataFrame(columns=MEETING_ANALYSIS_COLUMNS)

    scores = ensure_columns(
        meeting_scores,
        ["meeting_id", "bank", "meeting_date", "meeting_cycle_score", "meeting_cycle_lean", "documents_scored"],
    )
    if not meeting_index.empty and "meeting_id" in meeting_index.columns:
        idx = ensure_columns(meeting_index, ["meeting_id", "meeting_date"])[["meeting_id", "meeting_date"]]
        scores = scores.merge(idx, on="meeting_id", how="left", suffixes=("", "_index"))
        scores["meeting_date"] = scores["meeting_date"].where(scores["meeting_date"].astype(str).str.len() > 0, scores["meeting_date_index"])

    scores = parse_date_column(scores, "meeting_date").dropna(subset=["_date"])
    scores["meeting_cycle_score"] = numeric(scores, "meeting_cycle_score")
    scores["documents_scored"] = numeric(scores, "documents_scored").fillna(0).astype(int)
    scores = scores.sort_values(["bank", "_date", "meeting_id"])
    scores["previous_meeting_score"] = scores.groupby("bank")["meeting_cycle_score"].shift(1)
    scores["meeting_score_change"] = scores["meeting_cycle_score"] - scores["previous_meeting_score"]
    scores["rolling_3_meeting_score"] = scores.groupby("bank")["meeting_cycle_score"].transform(
        lambda series: series.rolling(3, min_periods=1).mean()
    )
    scores["rolling_6_meeting_score"] = scores.groupby("bank")["meeting_cycle_score"].transform(
        lambda series: series.rolling(6, min_periods=1).mean()
    )
    scores["meeting_cycle_lean"] = scores["meeting_cycle_score"].apply(classify_lean)
    scores["meeting_shift_classification"] = scores["meeting_score_change"].apply(lambda value: classify_shift(value, shift_threshold))
    scores["meeting_date"] = scores["_date"].dt.strftime("%Y-%m-%d")
    return scores[MEETING_ANALYSIS_COLUMNS].sort_values(["bank", "meeting_date", "meeting_id"])


def latest_in_window(df: pd.DataFrame, bank: str, date_column: str, latest_date: pd.Timestamp, window_days: int) -> pd.DataFrame:
    if df.empty or date_column not in df.columns:
        return pd.DataFrame()
    temp = parse_date_column(df[df["bank"].eq(bank)].copy(), date_column)
    start = latest_date - pd.Timedelta(days=window_days)
    return temp[temp["_date"].between(start, latest_date, inclusive="both")]


def weighted_score(values: dict[str, tuple[float, float | None]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for weight, value in values.values():
        if value is None or pd.isna(value):
            continue
        numerator += weight * float(value)
        denominator += weight
    return numerator / denominator if denominator else np.nan


def build_cross_bank_ranking(bank_trends: pd.DataFrame, meeting_analysis: pd.DataFrame, latest_window_days: int) -> pd.DataFrame:
    if bank_trends.empty:
        return pd.DataFrame(columns=CROSS_BANK_COLUMNS)

    trends = parse_date_column(bank_trends, "date_au").dropna(subset=["_date"])
    meetings = parse_date_column(meeting_analysis, "meeting_date") if not meeting_analysis.empty else pd.DataFrame()
    rows = []
    for bank, group in trends.groupby("bank"):
        group = group.sort_values("_date")
        latest_date = group["_date"].max()
        current_window = latest_in_window(trends, bank, "date_au", latest_date, latest_window_days)
        latest_trend = current_window.sort_values("_date").iloc[-1] if not current_window.empty else group.iloc[-1]

        meeting_window = latest_in_window(meetings, bank, "meeting_date", latest_date, latest_window_days) if not meetings.empty else pd.DataFrame()
        latest_meeting_score = np.nan
        if not meeting_window.empty:
            latest_meeting_score = pd.to_numeric(meeting_window.sort_values("_date").iloc[-1]["meeting_cycle_score"], errors="coerce")

        latest_30d = pd.to_numeric(latest_trend["rolling_30d_score"], errors="coerce")
        latest_90d = pd.to_numeric(latest_trend["rolling_90d_score"], errors="coerce")
        momentum = pd.to_numeric(latest_trend["momentum_30_vs_90"], errors="coerce")
        overall = weighted_score(
            {
                "latest_90d_score": (0.50, latest_90d),
                "latest_meeting_score": (0.30, latest_meeting_score),
                "latest_30d_score": (0.20, latest_30d),
            }
        )
        rows.append(
            {
                "bank": bank,
                "latest_date": latest_date.strftime("%Y-%m-%d"),
                "latest_30d_score": latest_30d,
                "latest_90d_score": latest_90d,
                "latest_meeting_score": latest_meeting_score,
                "momentum_30_vs_90": momentum,
                "overall_current_score": overall,
                "current_lean": classify_lean(overall),
            }
        )

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values("overall_current_score", ascending=False, na_position="last").reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    return ranking[CROSS_BANK_COLUMNS]


def build_document_type_breakdown(document_scores: pd.DataFrame) -> pd.DataFrame:
    if document_scores.empty:
        return pd.DataFrame(columns=DOCUMENT_BREAKDOWN_COLUMNS)
    df = ensure_columns(document_scores, ["bank", "document_subtype", "category", "clipped_score", "total_phrase_matches", "scoring_status"])
    df = df[df["scoring_status"].eq("scored")].copy()
    if df.empty:
        return pd.DataFrame(columns=DOCUMENT_BREAKDOWN_COLUMNS)
    df["_score"] = numeric(df, "clipped_score")
    df["_matches"] = numeric(df, "total_phrase_matches").fillna(0)
    grouped = (
        df.groupby(["bank", "document_subtype", "category"], as_index=False)
        .agg(
            document_count=("_score", "size"),
            average_score=("_score", "mean"),
            median_score=("_score", "median"),
            max_score=("_score", "max"),
            min_score=("_score", "min"),
            zero_match_count=("_matches", lambda series: int((series == 0).sum())),
            average_phrase_matches=("_matches", "mean"),
        )
        .sort_values(["bank", "document_subtype", "category"])
    )
    return grouped[DOCUMENT_BREAKDOWN_COLUMNS]


def frame_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    clean = df.replace({np.nan: None})
    return json.loads(clean.to_json(orient="records", date_format="iso"))


def strongest_row(df: pd.DataFrame, column: str, ascending: bool) -> dict[str, Any]:
    if df.empty or column not in df.columns:
        return {}
    temp = df.copy()
    temp[column] = pd.to_numeric(temp[column], errors="coerce")
    temp = temp.dropna(subset=[column])
    if temp.empty:
        return {}
    return temp.sort_values(column, ascending=ascending).iloc[0].to_dict()


def build_summary(
    document_scores: pd.DataFrame,
    bank_trends: pd.DataFrame,
    speaker_alerts: pd.DataFrame,
    meeting_analysis: pd.DataFrame,
    cross_bank: pd.DataFrame,
    generated_at: str,
) -> pd.DataFrame:
    scored = document_scores[document_scores.get("scoring_status", "").eq("scored")] if not document_scores.empty else pd.DataFrame()
    speakers = document_scores.get("speaker", pd.Series(dtype=str)).replace("", np.nan).dropna().nunique() if not document_scores.empty else 0
    most_hawkish_bank = cross_bank.iloc[0]["bank"] if not cross_bank.empty else ""
    most_dovish_bank = cross_bank.iloc[-1]["bank"] if not cross_bank.empty else ""
    rows = [
        {"metric": "total_documents_scored", "value": str(len(scored))},
        {"metric": "total_banks", "value": str(document_scores.get("bank", pd.Series(dtype=str)).nunique() if not document_scores.empty else 0)},
        {"metric": "total_speakers", "value": str(speakers)},
        {"metric": "total_meetings", "value": str(len(meeting_analysis))},
        {"metric": "most_hawkish_bank_latest", "value": str(most_hawkish_bank)},
        {"metric": "most_dovish_bank_latest", "value": str(most_dovish_bank)},
        {
            "metric": "strongest_hawkish_speaker_shift",
            "value": json.dumps(strongest_row(speaker_alerts, "shift_30_vs_prior_60", ascending=False), ensure_ascii=False, sort_keys=True),
        },
        {
            "metric": "strongest_dovish_speaker_shift",
            "value": json.dumps(strongest_row(speaker_alerts, "shift_30_vs_prior_60", ascending=True), ensure_ascii=False, sort_keys=True),
        },
        {
            "metric": "strongest_hawkish_meeting_shift",
            "value": json.dumps(strongest_row(meeting_analysis, "meeting_score_change", ascending=False), ensure_ascii=False, sort_keys=True),
        },
        {
            "metric": "strongest_dovish_meeting_shift",
            "value": json.dumps(strongest_row(meeting_analysis, "meeting_score_change", ascending=True), ensure_ascii=False, sort_keys=True),
        },
        {"metric": "generated_at", "value": generated_at},
    ]
    return pd.DataFrame(rows)


def write_dashboard_json(
    output_path: Path,
    generated_at: str,
    bank_trends: pd.DataFrame,
    speaker_alerts: pd.DataFrame,
    meeting_analysis: pd.DataFrame,
    cross_bank: pd.DataFrame,
    document_breakdown: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    payload = {
        "generated_at": generated_at,
        "bank_trends": frame_records(bank_trends),
        "speaker_shift_alerts": frame_records(speaker_alerts),
        "meeting_cycle_analysis": frame_records(meeting_analysis),
        "cross_bank_ranking": frame_records(cross_bank),
        "document_type_breakdown": frame_records(document_breakdown),
        "summary_statistics": {row["metric"]: row["value"] for _, row in summary.iterrows()},
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(output_dir: Path) -> None:
    text = """# Hawkometer Analytics Layer

Phase 3 turns Phase 2 document scores into research-ready analytics outputs. It does not use an LLM and does not alter raw scraped data.

## Bank Trends

Bank trends aggregate scored documents by bank and date. The script calculates daily average scores plus 30-day, 90-day, and 180-day rolling averages.

## Momentum

Momentum is `rolling_30d_score - rolling_90d_score`. Positive momentum means recent language is becoming more hawkish relative to the 90-day baseline. Negative momentum means recent language is becoming more dovish.

## Speaker Shift Alerts

Speaker alerts use the Phase 2 rolling speaker scores. A hawkish alert is created when `shift_30_vs_prior_60` is above the configured threshold. A dovish alert is created when it is below the negative threshold.

Alert strength:

- `mild`: absolute shift from 1.0 to below 2.0
- `strong`: absolute shift from 2.0 to below 3.5
- `extreme`: absolute shift 3.5 or above

## Meeting Cycle Analysis

Meeting analysis compares each meeting score with the prior meeting for the same bank. It also calculates 3-meeting and 6-meeting rolling averages to show meeting-cycle trend.

## Cross-Bank Ranking

The ranking uses each bank's latest 90-day score, latest meeting score, and latest 30-day score. The default weighted score is 50% latest 90-day, 30% latest meeting, and 20% latest 30-day, with weights re-normalised when a component is missing.

## Limitations

These analytics inherit Phase 2 limitations: phrase scoring is context-blind, weak on negation and hypotheticals, sensitive to missing text, and does not detect novel wording. Rankings are language indicators, not forecasts or causal estimates.

## Run

```powershell
python hawkometer_analytics.py --input processed/hawkometer --meeting-linkage processed/meeting_linkage --output processed/hawkometer_analytics
```

Optional:

```powershell
python hawkometer_analytics.py --start-date 2021-01-01 --end-date 2026-12-31 --shift-threshold 1.0 --latest-window-days 120
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run(
    input_dir: Path,
    meeting_linkage_dir: Path,
    output_dir: Path,
    start_date: str,
    end_date: str,
    shift_threshold: float,
    latest_window_days: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    document_scores = read_csv_safe(input_dir / "document_scores.csv", required=True)
    speaker_rolling = read_csv_safe(input_dir / "speaker_rolling_scores.csv")
    committee_scores = read_csv_safe(input_dir / "bank_committee_scores.csv")
    meeting_scores = read_csv_safe(input_dir / "meeting_hawkometer_scores.csv")
    meeting_documents = read_csv_safe(meeting_linkage_dir / "meeting_documents.csv")
    meeting_index = read_csv_safe(meeting_linkage_dir / "meeting_index.csv")

    document_scores = filter_by_date(document_scores, "date_au", start_date, end_date)
    speaker_rolling = filter_by_date(speaker_rolling, "date_au", start_date, end_date)
    committee_scores = filter_by_date(committee_scores, "date_au", start_date, end_date)
    meeting_scores = filter_by_date(meeting_scores, "meeting_date", start_date, end_date)
    meeting_index = filter_by_date(meeting_index, "meeting_date", start_date, end_date)

    bank_trends = build_bank_trends(document_scores, shift_threshold)
    speaker_alerts = build_speaker_shift_alerts(speaker_rolling, shift_threshold)
    meeting_analysis = build_meeting_cycle_analysis(meeting_scores, meeting_index, shift_threshold)
    cross_bank = build_cross_bank_ranking(bank_trends, meeting_analysis, latest_window_days)
    document_breakdown = build_document_type_breakdown(document_scores)
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = build_summary(document_scores, bank_trends, speaker_alerts, meeting_analysis, cross_bank, generated_at)

    bank_trends.to_csv(output_dir / "bank_trends.csv", index=False)
    speaker_alerts.to_csv(output_dir / "speaker_shift_alerts.csv", index=False)
    meeting_analysis.to_csv(output_dir / "meeting_cycle_analysis.csv", index=False)
    cross_bank.to_csv(output_dir / "cross_bank_ranking.csv", index=False)
    document_breakdown.to_csv(output_dir / "document_type_breakdown.csv", index=False)
    summary.to_csv(output_dir / "analytics_summary.csv", index=False)
    write_dashboard_json(
        output_dir / "hawkometer_dashboard_data.json",
        generated_at,
        bank_trends,
        speaker_alerts,
        meeting_analysis,
        cross_bank,
        document_breakdown,
        summary,
    )
    write_readme(output_dir)

    most_hawkish = cross_bank.iloc[0]["bank"] if not cross_bank.empty else "none"
    most_dovish = cross_bank.iloc[-1]["bank"] if not cross_bank.empty else "none"
    print(f"documents loaded: {len(document_scores)}")
    print(f"banks analysed: {document_scores.get('bank', pd.Series(dtype=str)).nunique() if not document_scores.empty else 0}")
    print(f"speakers analysed: {document_scores.get('speaker', pd.Series(dtype=str)).replace('', np.nan).dropna().nunique() if not document_scores.empty else 0}")
    print(f"meetings analysed: {len(meeting_analysis)}")
    print(f"speaker shift alerts found: {len(speaker_alerts)}")
    print(f"output folder path: {output_dir}")
    print(f"most hawkish current bank: {most_hawkish}")
    print(f"most dovish current bank: {most_dovish}")


def main() -> None:
    args = parse_args()
    run(
        input_dir=Path(args.input),
        meeting_linkage_dir=Path(args.meeting_linkage),
        output_dir=Path(args.output),
        start_date=args.start_date,
        end_date=args.end_date,
        shift_threshold=args.shift_threshold,
        latest_window_days=args.latest_window_days,
    )


if __name__ == "__main__":
    main()
