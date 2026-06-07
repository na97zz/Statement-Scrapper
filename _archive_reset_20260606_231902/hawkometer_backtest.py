from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHODOLOGY_VERSION = "hawkometer_backtest_v1"

MEETING_VALIDATION_COLUMNS = [
    "meeting_id",
    "bank",
    "meeting_date",
    "meeting_cycle_score",
    "meeting_cycle_lean",
    "previous_meeting_score",
    "meeting_score_change",
    "meeting_shift_classification",
    "rolling_3_meeting_score",
    "rolling_6_meeting_score",
    "documents_scored",
    "decision",
    "rate_change_bps",
    "policy_rate_after",
    "next_decision",
    "next_rate_change_bps",
    "hawkometer_signal",
    "hawkometer_signal_strength",
]

POLICY_BACKTEST_COLUMNS = [
    "meeting_id",
    "bank",
    "meeting_date",
    "meeting_cycle_score",
    "hawkometer_signal",
    "hawkometer_signal_strength",
    "decision",
    "rate_change_bps",
    "next_decision",
    "next_rate_change_bps",
    "same_meeting_direction_correct",
    "next_meeting_direction_correct",
    "absolute_error_bps",
    "signal_hit_rate_by_bank",
    "hit_rate_by_signal_type",
    "validation_note",
]

SCORE_CHANGE_BACKTEST_COLUMNS = [
    "meeting_id",
    "bank",
    "meeting_date",
    "meeting_cycle_score",
    "meeting_score_change",
    "rolling_3_meeting_score",
    "rolling_6_meeting_score",
    "rolling_3_score_change",
    "rolling_6_score_change",
    "shift_signal",
    "next_decision",
    "next_rate_change_bps",
    "next_decision_alignment",
    "next_rate_change_alignment",
    "score_change_bucket",
    "validation_note",
]

MARKET_BACKTEST_COLUMNS = [
    "bank",
    "meeting_id",
    "meeting_date",
    "asset",
    "window",
    "change",
    "change_bps",
    "change_pct",
    "meeting_cycle_score",
    "meeting_score_change",
    "hawkometer_signal",
    "meeting_cycle_lean",
]

MARKET_SUMMARY_COLUMNS = [
    "asset",
    "window",
    "group_type",
    "group_value",
    "observations",
    "average_market_change",
    "correlation_score_change",
    "correlation_score_level",
]

BANK_SUMMARY_COLUMNS = [
    "bank",
    "number_of_meetings",
    "average_meeting_score",
    "average_score_change",
    "hawkish_signal_count",
    "dovish_signal_count",
    "neutral_signal_count",
    "strong_hawkish_count",
    "strong_dovish_count",
    "policy_hit_rate",
    "next_meeting_hit_rate",
    "market_correlation_2y",
    "market_correlation_fx",
    "validation_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 Hawkometer validation and backtesting.")
    parser.add_argument("--hawkometer", default="processed/hawkometer", help="Phase 2 Hawkometer folder.")
    parser.add_argument("--analytics", default="processed/hawkometer_analytics", help="Phase 3 analytics folder.")
    parser.add_argument("--meeting-linkage", default="processed/meeting_linkage", help="Meeting linkage folder.")
    parser.add_argument("--output", default="processed/hawkometer_backtest", help="Backtest output folder.")
    parser.add_argument("--policy-decisions", default="data/external/policy_decisions.csv", help="Optional policy decisions CSV.")
    parser.add_argument("--market-reactions", default="data/external/market_reactions.csv", help="Optional market reactions CSV.")
    parser.add_argument("--start-date", default="2021-01-01", help="Start date filter.")
    parser.add_argument("--end-date", default="2026-12-31", help="End date filter.")
    parser.add_argument("--hawkish-threshold", type=float, default=1.0, help="Hawkish signal threshold.")
    parser.add_argument("--dovish-threshold", type=float, default=-1.0, help="Dovish signal threshold.")
    parser.add_argument("--strong-threshold", type=float, default=2.5, help="Strong hawkish/dovish threshold.")
    return parser.parse_args()


def read_csv_safe(path: Path, required: bool = False) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        kind = "required" if required else "optional"
        print(f"Warning: missing {kind} file: {path}")
        return pd.DataFrame(), False
    return pd.read_csv(path, dtype=str, keep_default_na=False), True


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def parse_dates(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    df = df.copy()
    if date_column not in df.columns:
        df[date_column] = ""
    df["_date"] = pd.to_datetime(df[date_column], errors="coerce")
    df["invalid_date"] = df["_date"].isna()
    df.loc[~df["invalid_date"], date_column] = df.loc[~df["invalid_date"], "_date"].dt.strftime("%Y-%m-%d")
    return df


def filter_dates(df: pd.DataFrame, date_column: str, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = parse_dates(df, date_column)
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


def hawkometer_signal(score: float, hawkish_threshold: float, dovish_threshold: float) -> str:
    if pd.isna(score):
        return "neutral"
    if score >= hawkish_threshold:
        return "hawkish"
    if score <= dovish_threshold:
        return "dovish"
    return "neutral"


def signal_strength(score: float, strong_threshold: float, hawkish_threshold: float, dovish_threshold: float) -> str:
    if pd.isna(score):
        return "neutral"
    if score >= strong_threshold:
        return "strong_hawkish"
    if score <= -abs(strong_threshold):
        return "strong_dovish"
    return hawkometer_signal(score, hawkish_threshold, dovish_threshold)


def decision_direction(decision: str, rate_change_bps: Any = None) -> str:
    rate = pd.to_numeric(pd.Series([rate_change_bps]), errors="coerce").iloc[0]
    if not pd.isna(rate):
        if rate > 0:
            return "hawkish"
        if rate < 0:
            return "dovish"
        return "neutral"
    decision_l = str(decision or "").strip().lower()
    if decision_l == "hike":
        return "hawkish"
    if decision_l == "cut":
        return "dovish"
    if decision_l == "hold":
        return "neutral"
    return ""


def direction_alignment(signal: str, decision: str, rate_change_bps: Any = None) -> str:
    direction = decision_direction(decision, rate_change_bps)
    if not direction:
        return ""
    return str(signal == direction)


def expected_rate_for_signal(signal: str) -> float:
    if signal == "hawkish":
        return 25.0
    if signal == "dovish":
        return -25.0
    return 0.0


def shift_signal(change: float, hawkish_threshold: float, dovish_threshold: float) -> str:
    if pd.isna(change):
        return "stable"
    if change >= hawkish_threshold:
        return "hawkish_shift"
    if change <= dovish_threshold:
        return "dovish_shift"
    return "stable"


def score_change_bucket(change: float) -> str:
    if pd.isna(change):
        return ""
    if change <= -2.5:
        return "<= -2.5"
    if change <= -1.0:
        return "-2.5 to -1.0"
    if change < 1.0:
        return "-1.0 to 1.0"
    if change < 2.5:
        return "1.0 to 2.5"
    return ">= 2.5"


def prepare_meeting_validation(
    meeting_scores: pd.DataFrame,
    meeting_cycle: pd.DataFrame,
    meeting_index: pd.DataFrame,
    policy_decisions: pd.DataFrame,
    policy_available: bool,
    hawkish_threshold: float,
    dovish_threshold: float,
    strong_threshold: float,
) -> pd.DataFrame:
    base = ensure_columns(
        meeting_scores,
        ["meeting_id", "bank", "meeting_date", "meeting_cycle_score", "meeting_cycle_lean", "documents_scored"],
    ).copy()
    cycle = ensure_columns(
        meeting_cycle,
        [
            "meeting_id",
            "previous_meeting_score",
            "meeting_score_change",
            "meeting_shift_classification",
            "rolling_3_meeting_score",
            "rolling_6_meeting_score",
        ],
    )
    index = ensure_columns(meeting_index, ["meeting_id", "meeting_date"])

    if not cycle.empty:
        base = base.merge(cycle.drop(columns=[c for c in ["bank", "meeting_date", "meeting_cycle_score", "meeting_cycle_lean", "documents_scored"] if c in cycle.columns]), on="meeting_id", how="left")
    if not index.empty:
        base = base.merge(index[["meeting_id", "meeting_date"]], on="meeting_id", how="left", suffixes=("", "_index"))
        base["meeting_date"] = base["meeting_date"].where(base["meeting_date"].astype(str).str.len() > 0, base["meeting_date_index"])
        base = base.drop(columns=["meeting_date_index"], errors="ignore")

    base = ensure_columns(base, MEETING_VALIDATION_COLUMNS)
    base = parse_dates(base, "meeting_date")
    base["meeting_cycle_score"] = numeric(base, "meeting_cycle_score")
    base["meeting_cycle_lean"] = base["meeting_cycle_score"].apply(classify_lean)
    base["hawkometer_signal"] = base["meeting_cycle_score"].apply(lambda score: hawkometer_signal(score, hawkish_threshold, dovish_threshold))
    base["hawkometer_signal_strength"] = base["meeting_cycle_score"].apply(
        lambda score: signal_strength(score, strong_threshold, hawkish_threshold, dovish_threshold)
    )

    if policy_available and not policy_decisions.empty:
        base = merge_policy_decisions(base, policy_decisions)
    else:
        for column in ["decision", "rate_change_bps", "policy_rate_after", "next_decision", "next_rate_change_bps"]:
            base[column] = ""

    return base[MEETING_VALIDATION_COLUMNS].sort_values(["bank", "meeting_date", "meeting_id"])


def merge_policy_decisions(meetings: pd.DataFrame, policy_decisions: pd.DataFrame) -> pd.DataFrame:
    policy = ensure_columns(
        policy_decisions,
        ["bank", "meeting_date", "meeting_id", "decision", "rate_change_bps", "policy_rate_after", "next_decision", "next_rate_change_bps"],
    )
    policy = parse_dates(policy, "meeting_date")
    direct = policy[policy["meeting_id"].astype(str).str.len() > 0].copy()
    no_id = policy[policy["meeting_id"].astype(str).str.len().eq(0)].copy()

    policy_columns = ["decision", "rate_change_bps", "policy_rate_after", "next_decision", "next_rate_change_bps"]
    merged = meetings.copy().drop(columns=policy_columns, errors="ignore")
    if not direct.empty:
        merged = merged.merge(
            direct[["meeting_id", *policy_columns]],
            on="meeting_id",
            how="left",
        )
    else:
        for column in policy_columns:
            merged[column] = ""

    if not no_id.empty:
        merged = parse_dates(merged, "meeting_date")
        for bank, bank_policy in no_id.groupby("bank"):
            bank_policy = bank_policy.dropna(subset=["_date"]).sort_values("_date")
            mask = merged["bank"].eq(bank) & merged["decision"].astype(str).str.len().eq(0)
            if not mask.any() or bank_policy.empty:
                continue
            bank_meetings = merged.loc[mask].sort_values("_date").copy()
            nearest = pd.merge_asof(
                bank_meetings.reset_index().sort_values("_date"),
                bank_policy[["bank", "_date", "decision", "rate_change_bps", "policy_rate_after", "next_decision", "next_rate_change_bps"]].sort_values("_date"),
                on="_date",
                by="bank",
                direction="nearest",
                tolerance=pd.Timedelta(days=7),
                suffixes=("", "_policy"),
            )
            for _, row in nearest.iterrows():
                if pd.isna(row.get("decision", np.nan)) or str(row.get("decision", "")).strip() == "":
                    continue
                idx = row["index"]
                for column in policy_columns:
                    merged.at[idx, column] = row[column]
    merged = ensure_columns(merged, policy_columns)
    return merged


def build_policy_signal_backtest(meeting_validation: pd.DataFrame, policy_available: bool) -> pd.DataFrame:
    if not policy_available:
        placeholder = pd.DataFrame(columns=POLICY_BACKTEST_COLUMNS)
        placeholder.loc[0, "validation_note"] = "policy_decisions.csv missing; add external policy data to enable this backtest."
        return placeholder

    df = ensure_columns(meeting_validation, POLICY_BACKTEST_COLUMNS[:10]).copy()
    df["same_meeting_direction_correct"] = df.apply(
        lambda row: direction_alignment(row["hawkometer_signal"], row["decision"], row["rate_change_bps"]), axis=1
    )
    df["next_meeting_direction_correct"] = df.apply(
        lambda row: direction_alignment(row["hawkometer_signal"], row["next_decision"], row["next_rate_change_bps"]), axis=1
    )
    df["rate_change_bps_num"] = pd.to_numeric(df["rate_change_bps"], errors="coerce")
    df["absolute_error_bps"] = df.apply(
        lambda row: abs(expected_rate_for_signal(row["hawkometer_signal"]) - row["rate_change_bps_num"])
        if not pd.isna(row["rate_change_bps_num"])
        else np.nan,
        axis=1,
    )
    same_bool = df["same_meeting_direction_correct"].map({"True": 1.0, "False": 0.0})
    df["signal_hit_rate_by_bank"] = df["bank"].map(same_bool.groupby(df["bank"]).mean())
    df["hit_rate_by_signal_type"] = df["hawkometer_signal"].map(same_bool.groupby(df["hawkometer_signal"]).mean())
    df["validation_note"] = ""
    return df[POLICY_BACKTEST_COLUMNS]


def build_score_change_backtest(
    meeting_validation: pd.DataFrame,
    policy_available: bool,
    hawkish_threshold: float,
    dovish_threshold: float,
) -> pd.DataFrame:
    df = ensure_columns(meeting_validation, SCORE_CHANGE_BACKTEST_COLUMNS[:7] + ["next_decision", "next_rate_change_bps"]).copy()
    df["meeting_score_change"] = numeric(df, "meeting_score_change")
    df["rolling_3_meeting_score"] = numeric(df, "rolling_3_meeting_score")
    df["rolling_6_meeting_score"] = numeric(df, "rolling_6_meeting_score")
    df = df.sort_values(["bank", "meeting_date", "meeting_id"])
    df["rolling_3_score_change"] = df.groupby("bank")["rolling_3_meeting_score"].diff()
    df["rolling_6_score_change"] = df.groupby("bank")["rolling_6_meeting_score"].diff()
    df["shift_signal"] = df["meeting_score_change"].apply(lambda value: shift_signal(value, hawkish_threshold, dovish_threshold))
    df["score_change_bucket"] = df["meeting_score_change"].apply(score_change_bucket)

    if policy_available:
        df["next_decision_alignment"] = df.apply(lambda row: shift_alignment(row["shift_signal"], row["next_decision"]), axis=1)
        df["next_rate_change_alignment"] = df.apply(
            lambda row: shift_rate_alignment(row["shift_signal"], row["next_rate_change_bps"]), axis=1
        )
        df["validation_note"] = ""
    else:
        df["next_decision_alignment"] = ""
        df["next_rate_change_alignment"] = ""
        df["validation_note"] = "policy_decisions.csv missing; add external policy data to enable shift validation."

    return df[SCORE_CHANGE_BACKTEST_COLUMNS]


def shift_alignment(signal: str, next_decision: str) -> str:
    expected = {"hawkish_shift": "hike", "dovish_shift": "cut", "stable": "hold"}.get(signal, "")
    if not expected or not str(next_decision).strip():
        return ""
    return str(str(next_decision).strip().lower() == expected)


def shift_rate_alignment(signal: str, next_rate_change_bps: Any) -> str:
    rate = pd.to_numeric(pd.Series([next_rate_change_bps]), errors="coerce").iloc[0]
    if pd.isna(rate):
        return ""
    if signal == "hawkish_shift":
        return str(rate > 0)
    if signal == "dovish_shift":
        return str(rate < 0)
    if signal == "stable":
        return str(rate == 0)
    return ""


def build_market_backtest(
    meeting_validation: pd.DataFrame,
    market_reactions: pd.DataFrame,
    market_available: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not market_available:
        backtest = pd.DataFrame(columns=MARKET_BACKTEST_COLUMNS)
        summary = pd.DataFrame(columns=MARKET_SUMMARY_COLUMNS)
        backtest.loc[0, "asset"] = "market_reactions.csv missing; add external market data to enable this backtest."
        return backtest, summary

    market = ensure_columns(
        market_reactions,
        ["bank", "meeting_date", "meeting_id", "asset", "window", "change", "change_bps", "change_pct"],
    )
    market = parse_dates(market, "meeting_date")
    validation = ensure_columns(
        meeting_validation,
        ["meeting_id", "bank", "meeting_date", "meeting_cycle_score", "meeting_score_change", "hawkometer_signal", "meeting_cycle_lean"],
    )
    merged = market.merge(
        validation[["meeting_id", "bank", "meeting_date", "meeting_cycle_score", "meeting_score_change", "hawkometer_signal", "meeting_cycle_lean"]],
        on=["meeting_id", "bank"],
        how="left",
        suffixes=("", "_meeting"),
    )
    merged["meeting_date"] = merged["meeting_date"].where(merged["meeting_date"].astype(str).str.len() > 0, merged["meeting_date_meeting"])
    merged = merged.drop(columns=["meeting_date_meeting"], errors="ignore")
    merged["change_num"] = pd.to_numeric(merged["change"], errors="coerce")
    merged["meeting_cycle_score"] = pd.to_numeric(merged["meeting_cycle_score"], errors="coerce")
    merged["meeting_score_change"] = pd.to_numeric(merged["meeting_score_change"], errors="coerce")

    summaries = []
    for keys, group_type in [(["asset", "window", "hawkometer_signal"], "hawkometer_signal"), (["asset", "window", "meeting_cycle_lean"], "meeting_cycle_lean")]:
        group_value_col = keys[-1]
        for values, group in merged.groupby(keys):
            asset, window, group_value = values
            summaries.append(
                {
                    "asset": asset,
                    "window": window,
                    "group_type": group_type,
                    "group_value": group_value,
                    "observations": len(group),
                    "average_market_change": group["change_num"].mean(),
                    "correlation_score_change": group["meeting_score_change"].corr(group["change_num"]),
                    "correlation_score_level": group["meeting_cycle_score"].corr(group["change_num"]),
                }
            )
    return merged[MARKET_BACKTEST_COLUMNS], pd.DataFrame(summaries, columns=MARKET_SUMMARY_COLUMNS)


def build_bank_validation_summary(
    meeting_validation: pd.DataFrame,
    policy_backtest: pd.DataFrame,
    market_summary: pd.DataFrame,
    policy_available: bool,
    market_available: bool,
) -> pd.DataFrame:
    rows = []
    validation_status = "complete" if policy_available and market_available else "policy_only" if policy_available else "score_only"
    for bank, group in meeting_validation.groupby("bank"):
        score_change = numeric(group, "meeting_score_change")
        policy_hit = np.nan
        next_hit = np.nan
        if policy_available and not policy_backtest.empty and "bank" in policy_backtest:
            bank_policy = policy_backtest[policy_backtest["bank"].eq(bank)]
            policy_hit = bool_mean(bank_policy.get("same_meeting_direction_correct", pd.Series(dtype=str)))
            next_hit = bool_mean(bank_policy.get("next_meeting_direction_correct", pd.Series(dtype=str)))
        rows.append(
            {
                "bank": bank,
                "number_of_meetings": len(group),
                "average_meeting_score": numeric(group, "meeting_cycle_score").mean(),
                "average_score_change": score_change.mean(),
                "hawkish_signal_count": int(group["hawkometer_signal"].eq("hawkish").sum()),
                "dovish_signal_count": int(group["hawkometer_signal"].eq("dovish").sum()),
                "neutral_signal_count": int(group["hawkometer_signal"].eq("neutral").sum()),
                "strong_hawkish_count": int(group["hawkometer_signal_strength"].eq("strong_hawkish").sum()),
                "strong_dovish_count": int(group["hawkometer_signal_strength"].eq("strong_dovish").sum()),
                "policy_hit_rate": policy_hit,
                "next_meeting_hit_rate": next_hit,
                "market_correlation_2y": market_correlation_for_bank(market_summary, bank, "2y") if market_available else np.nan,
                "market_correlation_fx": market_correlation_for_bank(market_summary, bank, "fx") if market_available else np.nan,
                "validation_status": validation_status,
            }
        )
    return pd.DataFrame(rows, columns=BANK_SUMMARY_COLUMNS).sort_values("bank")


def bool_mean(series: pd.Series) -> float:
    mapped = series.map({"True": 1.0, "False": 0.0})
    return mapped.mean()


def market_correlation_for_bank(market_summary: pd.DataFrame, bank: str, asset_hint: str) -> float:
    # Market summary is intentionally cross-bank in this phase; return NaN until bank-level market summaries are available.
    return np.nan


def build_backtest_summary(
    meeting_validation: pd.DataFrame,
    bank_summary: pd.DataFrame,
    policy_backtest: pd.DataFrame,
    policy_available: bool,
    market_available: bool,
    generated_at: str,
) -> pd.DataFrame:
    same_hit = np.nan
    next_hit = np.nan
    if policy_available and not policy_backtest.empty:
        same_hit = bool_mean(policy_backtest.get("same_meeting_direction_correct", pd.Series(dtype=str)))
        next_hit = bool_mean(policy_backtest.get("next_meeting_direction_correct", pd.Series(dtype=str)))
    strongest_hawkish = strongest_meeting(meeting_validation, "meeting_cycle_score", ascending=False)
    strongest_dovish = strongest_meeting(meeting_validation, "meeting_cycle_score", ascending=True)
    rows = [
        {"metric": "generated_at", "value": generated_at},
        {"metric": "total_meetings_tested", "value": str(len(meeting_validation))},
        {"metric": "total_banks", "value": str(meeting_validation["bank"].nunique() if not meeting_validation.empty else 0)},
        {"metric": "policy_decisions_available", "value": str(policy_available)},
        {"metric": "market_reactions_available", "value": str(market_available)},
        {"metric": "overall_policy_hit_rate", "value": "" if pd.isna(same_hit) else str(same_hit)},
        {"metric": "overall_next_meeting_hit_rate", "value": "" if pd.isna(next_hit) else str(next_hit)},
        {"metric": "best_validated_bank", "value": best_bank(bank_summary)},
        {"metric": "weakest_validated_bank", "value": weakest_bank(bank_summary)},
        {"metric": "strongest_hawkish_score", "value": json.dumps(strongest_hawkish, ensure_ascii=False, sort_keys=True)},
        {"metric": "strongest_dovish_score", "value": json.dumps(strongest_dovish, ensure_ascii=False, sort_keys=True)},
        {"metric": "average_meeting_score", "value": str(numeric(meeting_validation, "meeting_cycle_score").mean())},
        {"metric": "average_absolute_score_change", "value": str(numeric(meeting_validation, "meeting_score_change").abs().mean())},
    ]
    return pd.DataFrame(rows)


def strongest_meeting(df: pd.DataFrame, column: str, ascending: bool) -> dict[str, Any]:
    if df.empty:
        return {}
    temp = df.copy()
    temp[column] = numeric(temp, column)
    temp = temp.dropna(subset=[column])
    if temp.empty:
        return {}
    row = temp.sort_values(column, ascending=ascending).iloc[0]
    return {
        "meeting_id": row.get("meeting_id", ""),
        "bank": row.get("bank", ""),
        "meeting_date": row.get("meeting_date", ""),
        "meeting_cycle_score": float(row[column]),
        "meeting_cycle_lean": row.get("meeting_cycle_lean", ""),
    }


def best_bank(bank_summary: pd.DataFrame) -> str:
    if bank_summary.empty or "policy_hit_rate" not in bank_summary.columns:
        return ""
    temp = bank_summary.dropna(subset=["policy_hit_rate"])
    return "" if temp.empty else str(temp.sort_values("policy_hit_rate", ascending=False).iloc[0]["bank"])


def weakest_bank(bank_summary: pd.DataFrame) -> str:
    if bank_summary.empty or "policy_hit_rate" not in bank_summary.columns:
        return ""
    temp = bank_summary.dropna(subset=["policy_hit_rate"])
    return "" if temp.empty else str(temp.sort_values("policy_hit_rate", ascending=True).iloc[0]["bank"])


def frame_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


def write_json_output(
    path: Path,
    generated_at: str,
    policy_available: bool,
    market_available: bool,
    meeting_validation: pd.DataFrame,
    policy_backtest: pd.DataFrame,
    score_change_backtest: pd.DataFrame,
    bank_summary: pd.DataFrame,
    backtest_summary: pd.DataFrame,
) -> None:
    validation_status = "complete" if policy_available and market_available else "policy_only" if policy_available else "score_only"
    payload = {
        "generated_at": generated_at,
        "methodology_version": METHODOLOGY_VERSION,
        "validation_status": validation_status,
        "meeting_validation": frame_records(meeting_validation),
        "policy_signal_backtest": frame_records(policy_backtest),
        "score_change_backtest": frame_records(score_change_backtest),
        "bank_validation_summary": frame_records(bank_summary),
        "backtest_summary": {row["metric"]: row["value"] for _, row in backtest_summary.iterrows()},
        "methodology_notes": [
            "Phase 4 validates phrase-based Hawkometer communication tone against optional external policy and market datasets.",
            "If external files are missing, score-only placeholder outputs are produced.",
            "Hawkometer signals are based on meeting_cycle_score thresholds, not model predictions.",
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(output_dir: Path) -> None:
    text = """# Hawkometer Backtest

Phase 4 validates the phrase-based Hawkometer against optional policy decision and market reaction data. It does not use an LLM and does not alter raw archive files.

## Hawkometer Signals

Each meeting receives a directional signal from `meeting_cycle_score`:

- `hawkish` if the score is at or above the hawkish threshold
- `dovish` if the score is at or below the dovish threshold
- `neutral` otherwise

Strong signals use the configured strong threshold, defaulting to +/-2.5.

## Policy Decision Alignment

When `policy_decisions.csv` is available, the backtest compares Hawkometer signals to the same meeting decision and the next meeting decision. Hikes or positive rate changes are treated as hawkish, cuts or negative rate changes as dovish, and holds or zero changes as neutral.

Expected `policy_decisions.csv` columns:

- `bank`
- `meeting_date`
- `meeting_id` optional
- `decision`
- `rate_change_bps`
- `policy_rate_after`
- `next_decision`
- `next_rate_change_bps`

## Score Change Tests

The score-change backtest checks whether hawkish shifts precede hikes, dovish shifts precede cuts, and stable communication precedes holds. It also buckets meeting score changes into intuitive ranges.

## Market Reactions

When `market_reactions.csv` is available, meeting scores are merged to asset/window reactions and summarised by Hawkometer signal and meeting lean.

Expected `market_reactions.csv` columns:

- `bank`
- `meeting_date`
- `meeting_id` optional
- `asset`
- `window`
- `change`
- `change_bps` optional
- `change_pct` optional

## Limitations

- Hawkometer is a communication tone index, not a rate prediction model.
- Policy decisions depend on data, forecasts and committee voting, not speeches alone.
- A hawkish score may still occur before a hold.
- Market reactions depend on expectations, not just absolute tone.
- Phrase-based scores may misread negation or hypotheticals.
- Missing external validation data limits conclusions.

## Run

```powershell
python hawkometer_backtest.py --hawkometer processed/hawkometer --analytics processed/hawkometer_analytics --meeting-linkage processed/meeting_linkage --output processed/hawkometer_backtest
```

Optional:

```powershell
python hawkometer_backtest.py --policy-decisions data/external/policy_decisions.csv --market-reactions data/external/market_reactions.csv --start-date 2021-01-01 --end-date 2026-12-31 --hawkish-threshold 1.0 --dovish-threshold -1.0 --strong-threshold 2.5
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run(
    hawkometer_dir: Path,
    analytics_dir: Path,
    meeting_linkage_dir: Path,
    output_dir: Path,
    policy_decisions_path: Path,
    market_reactions_path: Path,
    start_date: str,
    end_date: str,
    hawkish_threshold: float,
    dovish_threshold: float,
    strong_threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    meeting_scores, _ = read_csv_safe(hawkometer_dir / "meeting_hawkometer_scores.csv", required=True)
    meeting_cycle, _ = read_csv_safe(analytics_dir / "meeting_cycle_analysis.csv", required=True)
    bank_trends, _ = read_csv_safe(analytics_dir / "bank_trends.csv")
    cross_bank, _ = read_csv_safe(analytics_dir / "cross_bank_ranking.csv")
    meeting_index, _ = read_csv_safe(meeting_linkage_dir / "meeting_index.csv", required=True)
    policy_decisions, policy_available = read_csv_safe(policy_decisions_path)
    market_reactions, market_available = read_csv_safe(market_reactions_path)

    if not policy_available:
        print("Warning: policy_decisions.csv not found; policy validation outputs will be placeholders.")
    if not market_available:
        print("Warning: market_reactions.csv not found; market validation outputs will be placeholders.")

    meeting_scores = filter_dates(meeting_scores, "meeting_date", start_date, end_date)
    meeting_cycle = filter_dates(meeting_cycle, "meeting_date", start_date, end_date)
    meeting_index = filter_dates(meeting_index, "meeting_date", start_date, end_date)
    if policy_available:
        policy_decisions = filter_dates(policy_decisions, "meeting_date", start_date, end_date)
    if market_available:
        market_reactions = filter_dates(market_reactions, "meeting_date", start_date, end_date)

    meeting_validation = prepare_meeting_validation(
        meeting_scores,
        meeting_cycle,
        meeting_index,
        policy_decisions,
        policy_available,
        hawkish_threshold,
        dovish_threshold,
        strong_threshold,
    )
    policy_backtest = build_policy_signal_backtest(meeting_validation, policy_available)
    score_change = build_score_change_backtest(meeting_validation, policy_available, hawkish_threshold, dovish_threshold)
    market_backtest, market_summary = build_market_backtest(meeting_validation, market_reactions, market_available)
    bank_summary = build_bank_validation_summary(meeting_validation, policy_backtest, market_summary, policy_available, market_available)
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = build_backtest_summary(
        meeting_validation,
        bank_summary,
        policy_backtest,
        policy_available,
        market_available,
        generated_at,
    )

    meeting_validation.to_csv(output_dir / "meeting_validation.csv", index=False)
    policy_backtest.to_csv(output_dir / "policy_signal_backtest.csv", index=False)
    score_change.to_csv(output_dir / "score_change_backtest.csv", index=False)
    market_backtest.to_csv(output_dir / "market_reaction_backtest.csv", index=False)
    market_summary.to_csv(output_dir / "market_reaction_summary.csv", index=False)
    bank_summary.to_csv(output_dir / "bank_validation_summary.csv", index=False)
    summary.to_csv(output_dir / "backtest_summary.csv", index=False)
    write_json_output(
        output_dir / "hawkometer_backtest.json",
        generated_at,
        policy_available,
        market_available,
        meeting_validation,
        policy_backtest,
        score_change,
        bank_summary,
        summary,
    )
    write_readme(output_dir)

    policy_validated = int(meeting_validation["decision"].astype(str).str.len().gt(0).sum()) if "decision" in meeting_validation else 0
    market_validated = 0 if not market_available else int(len(market_backtest))
    overall_hit = ""
    if policy_available and not policy_backtest.empty:
        overall_hit_value = bool_mean(policy_backtest.get("same_meeting_direction_correct", pd.Series(dtype=str)))
        overall_hit = "" if pd.isna(overall_hit_value) else f"{overall_hit_value:.3f}"

    print(f"meetings loaded: {len(meeting_validation)}")
    print(f"banks analysed: {meeting_validation['bank'].nunique() if not meeting_validation.empty else 0}")
    print(f"policy decision file found or missing: {'found' if policy_available else 'missing'}")
    print(f"market reaction file found or missing: {'found' if market_available else 'missing'}")
    print(f"meetings with policy validation: {policy_validated}")
    print(f"meetings with market validation: {market_validated}")
    print(f"output folder: {output_dir}")
    print(f"overall hit rate if available: {overall_hit if overall_hit else 'not available'}")


def main() -> None:
    args = parse_args()
    run(
        hawkometer_dir=Path(args.hawkometer),
        analytics_dir=Path(args.analytics),
        meeting_linkage_dir=Path(args.meeting_linkage),
        output_dir=Path(args.output),
        policy_decisions_path=Path(args.policy_decisions),
        market_reactions_path=Path(args.market_reactions),
        start_date=args.start_date,
        end_date=args.end_date,
        hawkish_threshold=args.hawkish_threshold,
        dovish_threshold=args.dovish_threshold,
        strong_threshold=args.strong_threshold,
    )


if __name__ == "__main__":
    main()
