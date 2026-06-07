from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:  # pragma: no cover - Streamlit fallback path
    px = None
    go = None
    HAS_PLOTLY = False


BASE_DIR = Path("processed/fed/hawkometer")
DOC_PATH = BASE_DIR / "fed_document_scores.csv"
SPEAKER_PATH = BASE_DIR / "fed_speaker_scores.csv"
MEETING_PATH = BASE_DIR / "fed_meeting_scores.csv"
SUMMARY_PATH = BASE_DIR / "fed_hawkometer_summary.csv"
POLICY_DECISIONS_PATH = Path("data/external/policy_decisions_final.csv")
BASELINE_YEAR = 2021
DEFAULT_START = pd.Timestamp("2021-01-01")
KEY_SPEAKERS = ["Powell", "Williams", "Jefferson", "Waller", "Bowman", "Daly", "Goolsbee"]
REGIME_LABELS = {
    2021: "Post-COVID accommodation",
    2022: "Aggressive tightening cycle",
    2023: "Banking stress / higher-for-longer",
    2024: "Pivot expectations",
    2025: "Easing cycle / normalization",
    2026: "Current regime",
}


st.set_page_config(
    page_title="FED Hawkometer",
    page_icon="FED",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #071016;
            --panel-bg: #0f1b24;
            --panel-soft: #111f2a;
            --panel-red: #251417;
            --panel-border: #263846;
            --panel-red-border: #713037;
            --ink: #edf6f8;
            --muted: #91a6b2;
            --hawk: #ff4d55;
            --dove: #4d8dff;
            --blue: #23d3b6;
        }
        .stApp { background: var(--page-bg); color: var(--ink); }
        [data-testid="stSidebar"] { background: #09151d; border-right: 1px solid var(--panel-border); }
        [data-testid="stHeader"] { background: rgba(7, 16, 22, 0.92); }
        h1, h2, h3 { letter-spacing: 0 !important; }
        h1, h2, h3, p, label, span, div { font-family: Inter, Segoe UI, Arial, sans-serif; }
        .terminal-title {
            font-family: Georgia, 'Times New Roman', serif;
            font-size: 1.9rem;
            font-weight: 700;
            color: var(--ink);
            padding: 0.25rem 0 0.35rem 0;
            margin-bottom: 0.45rem;
        }
        .section-copy { color: #c3d3da; margin: 0.1rem 0 1.15rem 0; }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
            margin: 0.75rem 0 1rem 0;
        }
        .kpi {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            min-height: 90px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
        }
        .kpi-label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; font-weight: 650; }
        .kpi-value { color: var(--ink); font-size: 1.35rem; font-weight: 750; padding-top: 0.25rem; }
        .kpi-help { color: var(--muted); font-size: 0.78rem; padding-top: 0.2rem; }
        .status-line {
            border: 1px solid var(--panel-border);
            background: #0d1820;
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
            color: #dce8eb;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
        }
        .small-muted { color: var(--muted); font-size: 0.84rem; }
        .research-panel {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-left: 3px solid var(--blue);
            border-radius: 10px;
            padding: 1.35rem 1.45rem;
            margin: 1rem 0 1.25rem 0;
            box-shadow: 0 2px 14px rgba(0, 0, 0, 0.28);
        }
        .panel-title {
            font-family: Georgia, 'Times New Roman', serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.4rem;
        }
        .bank-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(285px, 1fr));
            gap: 0.85rem;
            margin-top: 1rem;
        }
        .bank-card {
            background: var(--panel-soft);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 1rem;
            min-height: 82px;
        }
        .bank-card.hawkish { background: var(--panel-red); border-color: var(--panel-red-border); }
        .bank-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            font-weight: 750;
            color: var(--ink);
        }
        .bank-name { display: flex; align-items: center; gap: 0.55rem; }
        .flag-badge {
            width: 24px;
            height: 16px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #162838;
            border: 1px solid #2d4356;
            border-radius: 2px;
            color: #dce8eb;
            font-size: 0.62rem;
            font-weight: 800;
        }
        .score-red { color: var(--hawk); font-weight: 800; }
        .score-blue { color: var(--dove); font-weight: 800; }
        .lean-pill {
            border: 1px solid #2d4356;
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.72rem;
            color: #c3d3da;
            background: #0d1820;
            white-space: nowrap;
        }
        .lean-pill.hawkish { color: var(--hawk); border-color: var(--panel-red-border); }
        .score-track {
            position: relative;
            height: 9px;
            border-radius: 999px;
            background: linear-gradient(90deg, #3169df 0%, #5d6670 50%, #e5242a 100%);
            margin-top: 0.75rem;
        }
        .score-marker {
            position: absolute;
            top: -4px;
            width: 3px;
            height: 17px;
            background: #f5fbfc;
            border-radius: 2px;
        }
        .score-scale {
            display: flex;
            justify-content: space-between;
            color: #a9bbc5;
            font-size: 0.72rem;
            margin-top: 0.35rem;
        }
        .shift-line {
            margin-top: 0.75rem;
            color: #c3d3da;
            font-size: 0.84rem;
        }
        div[data-testid="stDataFrame"] {
            background: #0f1b24;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.3rem;
        }
        div[data-testid="stTabs"] button {
            background: #0f1b24;
            border: 1px solid var(--panel-border);
            color: var(--ink);
            border-radius: 8px 8px 0 0;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            border-bottom-color: var(--blue);
            color: #ffffff;
        }
        div[data-testid="stAlert"] {
            background: #1e2630;
            color: var(--ink);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def required_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def as_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.warning(f"Missing file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        st.warning(f"Could not load {path}: {exc}")
        return pd.DataFrame()


def parse_match_blob(value: Any) -> list[dict[str, Any]]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    try:
        parsed = json.loads(str(value))
    except Exception:
        try:
            parsed = ast.literal_eval(str(value))
        except Exception:
            return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    docs = safe_read_csv(DOC_PATH)
    meetings = safe_read_csv(MEETING_PATH)
    speakers = safe_read_csv(SPEAKER_PATH)
    summary = safe_read_csv(SUMMARY_PATH)

    docs = required_columns(
        docs,
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
            "source_url",
            "score",
            "classification",
            "confidence",
            "lean",
            "hawkish_score",
            "dovish_score",
            "total_phrase_matches",
            "matched_phrases_json",
        ],
    )
    docs = as_numeric(
        docs,
        ["score", "confidence", "hawkish_score", "dovish_score", "total_phrase_matches", "matched_phrase_count"],
    )
    docs["date"] = pd.to_datetime(docs["date_au"], errors="coerce")
    docs["year"] = docs["date"].dt.year
    docs["clipped_score"] = docs["score"]
    docs["speaker"] = docs["speaker"].replace("", "Unknown speaker")

    meetings = required_columns(
        meetings,
        [
            "meeting_id",
            "bank",
            "meeting_date",
            "meeting_score",
            "classification",
            "lean",
            "statement_score",
            "minutes_score",
            "press_conference_score",
            "speech_avg_score",
            "documents_scored",
            "zero_match_documents",
        ],
    )
    meetings = as_numeric(
        meetings,
        [
            "meeting_score",
            "average_document_score",
            "average_confidence",
            "statement_score",
            "minutes_score",
            "press_conference_score",
            "sep_score",
            "speech_avg_score",
            "documents_scored",
            "zero_match_documents",
        ],
    )
    meetings["date"] = pd.to_datetime(meetings["meeting_date"], errors="coerce")
    meetings["year"] = meetings["date"].dt.year
    meetings["meeting_cycle_score"] = meetings.apply(dashboard_meeting_score, axis=1)
    classified = meetings["meeting_cycle_score"].apply(classify_score)
    meetings["meeting_cycle_classification"] = classified.apply(lambda item: item[0])
    meetings["meeting_cycle_lean"] = classified.apply(lambda item: item[1])
    meetings["rolling_3"] = meetings.sort_values("date")["meeting_cycle_score"].rolling(3, min_periods=1).mean()
    meetings["rolling_6"] = meetings.sort_values("date")["meeting_cycle_score"].rolling(6, min_periods=1).mean()

    speakers = required_columns(
        speakers,
        [
            "speaker",
            "documents_scored",
            "average_score",
            "average_confidence",
            "latest_date",
            "latest_score",
            "most_hawkish_score",
            "most_dovish_score",
        ],
    )
    speakers = as_numeric(
        speakers,
        ["documents_scored", "average_score", "average_confidence", "latest_score", "most_hawkish_score", "most_dovish_score"],
    )

    phrase_rows: list[dict[str, Any]] = []
    for _, row in docs.iterrows():
        for match in parse_match_blob(row.get("matched_phrases_json", "")):
            count = int(match.get("count", 1) or 1)
            phrase_rows.append(
                {
                    "doc_id": row["doc_id"],
                    "date": row["date"],
                    "date_au": row["date_au"],
                    "year": row["year"],
                    "speaker": row["speaker"],
                    "title": row["title"],
                    "document_subtype": row["document_subtype"],
                    "score": row["score"],
                    "phrase": match.get("phrase", ""),
                    "direction": match.get("direction", ""),
                    "theme": match.get("theme", ""),
                    "count": count,
                    "weight": float(match.get("weight", 0) or 0),
                    "scaled_contribution": float(match.get("scaled_contribution", 0) or 0),
                }
            )
    phrases = pd.DataFrame(phrase_rows)
    if phrases.empty:
        phrases = pd.DataFrame(
            columns=[
                "doc_id",
                "date",
                "date_au",
                "year",
                "speaker",
                "title",
                "document_subtype",
                "score",
                "phrase",
                "direction",
                "theme",
                "count",
                "weight",
                "scaled_contribution",
            ]
        )
    return docs, meetings, speakers, summary, phrases


def print_startup_summary(docs: pd.DataFrame, meetings: pd.DataFrame, speakers: pd.DataFrame) -> None:
    date_min = docs["date"].min() if not docs.empty else None
    date_max = docs["date"].max() if not docs.empty else None
    print("FED Hawkometer dashboard ready")
    print(f"documents loaded: {len(docs)}")
    print(f"meetings loaded: {len(meetings)}")
    print(f"speakers loaded: {docs['speaker'].nunique() if not docs.empty else 0}")
    print(f"date range: {date_min.date() if pd.notna(date_min) else 'n/a'} to {date_max.date() if pd.notna(date_max) else 'n/a'}")


def csv_download(df: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")


def kpi_card(label: str, value: Any, help_text: str = "") -> str:
    return f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-help">{help_text}</div>
    </div>
    """


def render_kpis(cards: list[tuple[str, Any, str]]) -> None:
    for start in range(0, len(cards), 4):
        row = cards[start : start + 4]
        columns = st.columns(len(row))
        for column, card in zip(columns, row):
            with column:
                st.markdown(kpi_card(*card), unsafe_allow_html=True)


def score_colour_class(score: float) -> str:
    return "score-red" if pd.notna(score) and score > 0 else "score-blue" if pd.notna(score) and score < 0 else ""


def score_text(score: float) -> str:
    if pd.isna(score):
        return "n/a"
    return f"{score:+.2f}"


def score_position(score: float) -> float:
    if pd.isna(score):
        return 50.0
    return max(0.0, min(100.0, (float(score) + 10.0) / 20.0 * 100.0))


def lean_label(score: float) -> str:
    if pd.isna(score):
        return "Unavailable"
    if score >= 3:
        return "Hawkish"
    if score >= 0.5:
        return "Leaning Hawkish"
    if score <= -3:
        return "Dovish"
    if score <= -0.5:
        return "Leaning Dovish"
    return "Neutral"


def fed_bank_card(score: float, shift: float | None = None) -> str:
    lean = lean_label(score)
    hawkish = pd.notna(score) and score >= 0.5
    shift_text = "n/a" if shift is None or pd.isna(shift) else score_text(float(shift))
    return f"""
    <div class="bank-card {'hawkish' if hawkish else ''}">
      <div class="bank-top">
        <div class="bank-name"><span class="flag-badge">US</span><span>Federal Reserve</span></div>
        <span class="{score_colour_class(score)}">{score_text(score)}</span>
      </div>
      <div class="small-muted" style="margin-top:0.55rem;">{lean}</div>
      <div class="score-track"><div class="score-marker" style="left:{score_position(score)}%;"></div></div>
      <div class="score-scale"><span>Dovish -10</span><span>+10 Hawkish</span></div>
      <div class="shift-line">30d vs prior 60d shift: <span class="{score_colour_class(shift if shift is not None else np.nan)}">{shift_text}</span></div>
    </div>
    """


def fed_compact_bank_card(score: float) -> str:
    lean = lean_label(score)
    hawkish = pd.notna(score) and score >= 0.5
    return f"""
    <div class="bank-card {'hawkish' if hawkish else ''}">
      <div class="bank-top">
        <div class="bank-name"><span class="flag-badge">US</span><span>Federal Reserve</span></div>
        <span class="{score_colour_class(score)}">{score_text(score)}</span>
      </div>
      <div class="small-muted" style="margin-top:0.7rem;">{lean}</div>
    </div>
    """


def recent_meeting_shift(meetings: pd.DataFrame) -> float:
    if meetings.empty:
        return np.nan
    data = meetings.sort_values("date").copy()
    latest_date = data["date"].max()
    recent = data[data["date"] >= latest_date - pd.Timedelta(days=30)]["meeting_cycle_score"].mean()
    prior = data[
        (data["date"] < latest_date - pd.Timedelta(days=30))
        & (data["date"] >= latest_date - pd.Timedelta(days=90))
    ]["meeting_cycle_score"].mean()
    return recent - prior if pd.notna(recent) and pd.notna(prior) else np.nan


def current_fed_score(meetings: pd.DataFrame) -> float:
    if meetings.empty:
        return np.nan
    latest = meetings.sort_values("date").tail(1)
    return latest["meeting_cycle_score"].iloc[0] if not latest.empty else np.nan


def section_global_dashboard(meetings: pd.DataFrame) -> None:
    score = current_fed_score(meetings)
    st.markdown('<div class="terminal-title">The global dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">The bar chart below ranks each committee by its current 90-day, voter-weighted Hawkometer score.</div>',
        unsafe_allow_html=True,
    )
    chart_data = pd.DataFrame(
        [{"committee": "Federal Reserve", "score": score if pd.notna(score) else 0.0}]
    )
    if HAS_PLOTLY:
        colours = ["#e5242a" if value >= 0 else "#3169df" for value in chart_data["score"]]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=chart_data["score"],
                y=chart_data["committee"],
                orientation="h",
                marker_color=colours,
                text=[score_text(value) for value in chart_data["score"]],
                textposition="outside",
                textfont={"color": "#edf6f8", "size": 12},
            )
        )
        fig.update_layout(
            template="plotly_dark",
            title={"text": "Which committee is most hawkish right now?", "x": 0.5, "font": {"size": 15, "color": "#edf6f8"}},
            height=250,
            xaxis={"range": [-10, 10], "title": "Weighted 90-day committee score", "zeroline": True},
            yaxis={"title": ""},
            margin={"l": 135, "r": 40, "t": 55, "b": 45},
            paper_bgcolor="#0f1b24",
            plot_bgcolor="#0f1b24",
            font={"color": "#c3d3da"},
        )
        fig.update_xaxes(gridcolor="#223541", zerolinecolor="#edf6f8")
        fig.update_yaxes(gridcolor="#223541")
        fig.add_vline(x=0, line_width=1, line_color="#edf6f8")
        st.plotly_chart(fig, width="stretch", theme=None)
    else:
        st.bar_chart(chart_data.set_index("committee")["score"])


def section_per_bank(meetings: pd.DataFrame) -> None:
    score = current_fed_score(meetings)
    st.markdown('<div class="terminal-title">Per-bank Hawkometers</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Each central bank has a dedicated Hawkometer page covering committee composition, voting bloc lean, recent speech sentiment and upcoming appearances.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="bank-grid">{fed_compact_bank_card(score)}</div>', unsafe_allow_html=True)


def section_committee_glance(meetings: pd.DataFrame) -> None:
    score = current_fed_score(meetings)
    shift = recent_meeting_shift(meetings)
    st.markdown(
        f"""
        <div class="research-panel">
          <div class="panel-title">Committee scores at a glance</div>
          <div class="section-copy">Voter-weighted 90-day rolling Hawkometer score for each committee. Hover any card for the underlying speaker breakdown.</div>
          <div class="bank-grid">{fed_bank_card(score, shift)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_latest_appearances(docs: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="research-panel">
          <div class="panel-title">Latest scored appearances</div>
        """,
        unsafe_allow_html=True,
    )
    latest = docs.sort_values("date", ascending=False).head(20).copy()
    latest["Bank"] = "US"
    latest["Date"] = latest["date"].apply(
        lambda value: f"{value.strftime('%b')} {value.day}, {value.year}" if pd.notna(value) else ""
    )
    latest["Speaker"] = latest["speaker"]
    latest["Title"] = latest["title"]
    latest["Type"] = latest["document_subtype"].str.replace("_", " ", regex=False).str.title()
    latest["Score"] = latest["score"].map(lambda value: f"{value:+.2f}" if pd.notna(value) else "n/a")
    table = latest[["Date", "Bank", "Speaker", "Title", "Type", "Score"]]
    st.dataframe(table, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_four_section_dashboard(docs: pd.DataFrame, meetings: pd.DataFrame) -> None:
    section_global_dashboard(meetings)
    section_per_bank(meetings)
    section_committee_glance(meetings)
    section_latest_appearances(docs)


def parse_rate_token(token: str) -> float | None:
    text = str(token).lower().strip().replace("–", "-").replace("—", "-").replace("‑", "-")
    text = text.replace("−", "-").replace("‐", "-").replace("‑", "-").replace("‒", "-")
    text = text.replace("zero", "0").replace("¼", "1/4").replace("½", "1/2").replace("¾", "3/4")
    text = re.sub(r"\s+", "", text)
    try:
        if "-" in text and "/" in text:
            whole, frac = text.split("-", 1)
            numerator, denominator = frac.split("/", 1)
            return float(whole) + float(numerator) / float(denominator)
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return float(numerator) / float(denominator)
        return float(text)
    except Exception:
        return None


def infer_target_midpoint(text_file: str) -> float | None:
    path = Path(str(text_file))
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    text = text.replace("â€‘", "-").replace("â€™", "'").replace("â€“", "-")
    text = text.replace("−", "-").replace("‐", "-").replace("‑", "-").replace("‒", "-").replace("–", "-").replace("—", "-")
    text = text.replace("¼", "1/4").replace("½", "1/2").replace("¾", "3/4")
    text = re.sub(r"\s+", " ", text)
    patterns = [
        r"target range for the federal funds rate.*?(?:at|to)\s+(zero|\d+(?:[- ]\d+/\d+|/\d+|\.\d+)?)\s+to\s+(zero|\d+(?:[- ]\d+/\d+|/\d+|\.\d+)?)\s+percent",
        r"federal funds rate.*?target range.*?(?:at|to)\s+(zero|\d+(?:[- ]\d+/\d+|/\d+|\.\d+)?)\s+to\s+(zero|\d+(?:[- ]\d+/\d+|/\d+|\.\d+)?)\s+percent",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        low = parse_rate_token(match.group(1))
        high = parse_rate_token(match.group(2))
        if low is not None and high is not None:
            return (low + high) / 2.0
    return None


def decision_from_bps(change_bps: float | None) -> str:
    if change_bps is None or pd.isna(change_bps):
        return "unknown"
    if change_bps > 1:
        return "hike"
    if change_bps < -1:
        return "cut"
    return "hold"


def load_policy_decisions() -> pd.DataFrame:
    if not POLICY_DECISIONS_PATH.exists():
        return pd.DataFrame()
    decisions = safe_read_csv(POLICY_DECISIONS_PATH)
    decisions = required_columns(decisions, ["meeting_id", "decision", "rate_change_bps"])
    decisions["rate_change_bps"] = pd.to_numeric(decisions["rate_change_bps"], errors="coerce")
    decisions["decision"] = decisions["decision"].astype(str).str.lower().str.strip()
    return decisions[["meeting_id", "decision", "rate_change_bps"]].drop_duplicates("meeting_id")


def build_decision_table(meetings: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    base = meetings.sort_values("date").copy()
    base = base[["meeting_id", "meeting_date", "date", "meeting_cycle_score", "statement_score", "minutes_score", "press_conference_score", "speech_avg_score"]]
    policy = load_policy_decisions()
    if not policy.empty:
        base = base.merge(policy.rename(columns={"decision": "current_decision", "rate_change_bps": "current_rate_change_bps"}), on="meeting_id", how="left")
        base["decision_source"] = "policy_decisions_final.csv"
        base["target_midpoint"] = np.nan
    else:
        statements = docs[docs["document_subtype"].eq("statement")].sort_values("date").drop_duplicates("meeting_id")
        statement_paths = statements.set_index("meeting_id")["text_file"].to_dict()
        base["target_midpoint"] = base["meeting_id"].map(lambda meeting_id: infer_target_midpoint(statement_paths.get(meeting_id, "")))
        base["current_rate_change_bps"] = base["target_midpoint"].diff() * 100.0
        if not base.empty:
            base.loc[base.index[0], "current_rate_change_bps"] = 0.0
        base["current_decision"] = base["current_rate_change_bps"].apply(decision_from_bps)
        base["decision_source"] = "inferred from statement target range"

    base["next_decision"] = base["current_decision"].shift(-1)
    base["next_rate_change_bps"] = base["current_rate_change_bps"].shift(-1)
    base["next_meeting_score"] = base["meeting_cycle_score"].shift(-1)
    base["next_meeting_id"] = base["meeting_id"].shift(-1)
    base["next_target_midpoint"] = base["target_midpoint"].shift(-1)
    return base


def build_research_cycles(docs: pd.DataFrame, meetings: pd.DataFrame) -> pd.DataFrame:
    cycles = build_decision_table(meetings, docs)
    speech_types = {"speech", "interview", "testimony"}
    intermeeting = docs[docs["document_subtype"].isin(speech_types)].copy()
    speech_summary = intermeeting.groupby("meeting_id", as_index=False).agg(
        speech_signal_score=("score", "mean"),
        speech_documents=("doc_id", "count"),
        speech_score_change=("score", lambda series: series.sort_index().iloc[-1] - series.sort_index().iloc[0] if len(series) > 1 else 0.0),
    )
    cycles = cycles.merge(speech_summary, on="meeting_id", how="left")
    cycles["speech_signal_score"] = cycles["speech_signal_score"].fillna(0.0)
    cycles["speech_documents"] = cycles["speech_documents"].fillna(0).astype(int)
    cycles["speech_score_change"] = cycles["speech_score_change"].fillna(0.0)
    cycles["statement_score_change"] = cycles["statement_score"].diff()
    cycles["speech_statement_gap"] = cycles["speech_signal_score"] - cycles["statement_score"]
    cycles["fgi"] = cycles.apply(forward_guidance_indicator, axis=1)
    cycles["fgi_classification"] = cycles["fgi"].apply(fgi_classification)
    return cycles


def forward_guidance_indicator(row: pd.Series) -> float:
    weights = {
        "statement_score": 0.40,
        "minutes_score": 0.30,
        "press_conference_score": 0.20,
        "speech_signal_score": 0.10,
    }
    numerator = 0.0
    denominator = 0.0
    for column, weight in weights.items():
        value = row.get(column)
        if pd.notna(value):
            numerator += float(value) * weight
            denominator += weight
    return numerator / denominator if denominator else np.nan


def fgi_classification(score: float) -> str:
    if pd.isna(score):
        return "Unavailable"
    if score <= -6:
        return "Strongly Dovish"
    if score <= -2:
        return "Dovish"
    if score < 2:
        return "Neutral"
    if score < 6:
        return "Hawkish"
    return "Strongly Hawkish"


def apply_lab_filters(docs: pd.DataFrame, cycles: pd.DataFrame, phrases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    st.sidebar.header("Research Lab Filters")
    min_date = pd.Timestamp("2021-01-01")
    max_date = docs["date"].max() if not docs.empty else min_date
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date.date(), max_date.date() if pd.notna(max_date) else min_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date() if pd.notna(max_date) else min_date.date(),
        key="lab_date_range",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start, end = min_date, max_date
    filtered_docs = docs[(docs["date"] >= start) & (docs["date"] <= end)].copy()
    filtered_cycles = cycles[(cycles["date"] >= start) & (cycles["date"] <= end)].copy()

    speaker_options = sorted(filtered_docs["speaker"].dropna().unique().tolist())
    type_options = sorted(filtered_docs["document_subtype"].dropna().unique().tolist())
    decision_options = ["cut", "hold", "hike", "unknown"]
    phrase_options = sorted(phrases["phrase"].dropna().unique().tolist()) if not phrases.empty else []
    selected_speakers = st.sidebar.multiselect("Speaker", speaker_options, key="lab_speaker")
    selected_types = st.sidebar.multiselect("Document type", type_options, key="lab_type")
    selected_phrase = st.sidebar.selectbox("Phrase", ["All phrases"] + phrase_options, key="lab_phrase")
    selected_decisions = st.sidebar.multiselect("Next decision", decision_options, key="lab_next_decision")

    if selected_speakers:
        filtered_docs = filtered_docs[filtered_docs["speaker"].isin(selected_speakers)]
    if selected_types:
        filtered_docs = filtered_docs[filtered_docs["document_subtype"].isin(selected_types)]
    if selected_decisions:
        filtered_cycles = filtered_cycles[filtered_cycles["next_decision"].isin(selected_decisions)]
    filtered_phrases = phrases[phrases["doc_id"].isin(filtered_docs["doc_id"])] if not phrases.empty else phrases
    if selected_phrase != "All phrases" and not filtered_phrases.empty:
        matching_docs = filtered_phrases[filtered_phrases["phrase"].eq(selected_phrase)]["doc_id"].unique()
        filtered_docs = filtered_docs[filtered_docs["doc_id"].isin(matching_docs)]
        filtered_phrases = filtered_phrases[filtered_phrases["phrase"].eq(selected_phrase)]
    return filtered_docs, filtered_cycles, filtered_phrases


def lab_plot_scatter(cycles: pd.DataFrame) -> None:
    data = cycles.dropna(subset=["speech_signal_score", "next_rate_change_bps"])
    if data.empty:
        st.info("No next-decision data available for the scatter plot.")
        return
    fig = px.scatter(
        data,
        x="speech_signal_score",
        y="next_rate_change_bps",
        color="next_decision",
        hover_data=["meeting_id", "meeting_date", "next_meeting_id"],
        title="Speech Signal Score vs Next Rate Change",
        color_discrete_map={"hike": "#e5242a", "hold": "#7b8794", "cut": "#3169df", "unknown": "#9aa5b1"},
    )
    fig.update_layout(template="plotly_dark", height=420)
    fig.add_hline(y=0, line_dash="dot", line_color="#52677a")
    fig.add_vline(x=0, line_dash="dot", line_color="#52677a")
    st.plotly_chart(fig, width="stretch", theme=None)


def research_metric_table(cycles: pd.DataFrame, score_column: str, label: str) -> pd.DataFrame:
    return (
        cycles.dropna(subset=[score_column])
        .groupby("next_decision", as_index=False)
        .agg(
            average_score=(score_column, "mean"),
            median_score=(score_column, "median"),
            meetings=("meeting_id", "count"),
            average_next_rate_change_bps=("next_rate_change_bps", "mean"),
        )
        .rename(columns={"average_score": f"average_{label}"})
        .sort_values("next_decision")
    )


def page_lab_speech_signal(cycles: pd.DataFrame) -> None:
    st.markdown('<div class="research-panel"><div class="panel-title">1. Speech Signal Analysis</div>', unsafe_allow_html=True)
    st.markdown("Do speeches, interviews, and testimony between one FOMC meeting and the next contain information about the next decision?")
    lab_plot_scatter(cycles)
    c1, c2 = st.columns(2)
    with c1:
        box_data = cycles[cycles["next_decision"].isin(["cut", "hold", "hike"])]
        if box_data.empty:
            st.info("No cut/hold/hike decision groups available.")
        else:
            fig = px.box(
                box_data,
                x="next_decision",
                y="speech_signal_score",
                color="next_decision",
                title="Speech Signal Score by Next Decision",
                color_discrete_map={"hike": "#e5242a", "hold": "#7b8794", "cut": "#3169df"},
            )
            fig.update_layout(template="plotly_dark", height=390)
            st.plotly_chart(fig, width="stretch", theme=None)
    with c2:
        st.dataframe(research_metric_table(cycles, "speech_signal_score", "speech_score"), width="stretch", hide_index=True)
        corr_data = cycles[["speech_signal_score", "next_rate_change_bps"]].dropna()
        corr = corr_data["speech_signal_score"].corr(corr_data["next_rate_change_bps"]) if len(corr_data) > 1 else np.nan
        render_kpis([("Correlation", f"{corr:.3f}" if pd.notna(corr) else "n/a", "Speech score vs next rate change")])
    st.markdown("</div>", unsafe_allow_html=True)


def phrase_summary_for_docs(doc_ids: list[str], phrases: pd.DataFrame, directions: set[str] | None = None) -> pd.DataFrame:
    if phrases.empty or not doc_ids:
        return pd.DataFrame(columns=["phrase", "direction", "occurrences", "average_contribution"])
    data = phrases[phrases["doc_id"].isin(doc_ids)].copy()
    if directions:
        data = data[data["direction"].isin(directions)]
    if data.empty:
        return pd.DataFrame(columns=["phrase", "direction", "occurrences", "average_contribution"])
    return (
        data.groupby(["phrase", "direction"], as_index=False)
        .agg(occurrences=("count", "sum"), average_contribution=("scaled_contribution", "mean"))
        .sort_values(["occurrences", "average_contribution"], ascending=[False, False])
    )


def page_lab_meeting_cycle_bridge(cycles: pd.DataFrame, docs: pd.DataFrame, phrases: pd.DataFrame) -> None:
    st.markdown(
        '<div class="research-panel"><div class="panel-title">Meeting Cycle Forward Guidance Bridge</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Select a current FOMC meeting to compare its policy decision and forward-guidance language with all intermeeting speeches before the next FOMC decision."
    )
    if cycles.empty:
        st.info("No meeting-cycle data available.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    timeline = cycles[cycles["date"].dt.year.between(2021, 2026, inclusive="both")].sort_values("date").copy()
    st.subheader("2021-2026 Meeting-to-Next-Meeting Timeline")
    st.markdown(
        "Each point is a current FOMC meeting. The line shows the Forward Guidance Indicator, bars show intermeeting speech signal, and marker colour shows the next policy decision."
    )
    if timeline.empty:
        st.info("No 2021-2026 timeline data available.")
    elif HAS_PLOTLY:
        colour_map = {"hike": "#ff4d55", "hold": "#91a6b2", "cut": "#4d8dff", "unknown": "#f0b429"}
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=timeline["date"],
                y=timeline["speech_signal_score"],
                name="Intermeeting speech signal",
                marker_color="rgba(77, 141, 255, 0.42)",
                hovertemplate="%{x|%Y-%m-%d}<br>Speech signal: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timeline["date"],
                y=timeline["fgi"],
                mode="lines+markers+text",
                name="Forward Guidance Indicator",
                marker={
                    "size": 11,
                    "color": [colour_map.get(str(item), "#f0b429") for item in timeline["next_decision"].fillna("unknown")],
                    "line": {"width": 1, "color": "#edf6f8"},
                },
                text=timeline["next_decision"].fillna("n/a"),
                textposition="top center",
                customdata=timeline[
                    [
                        "meeting_id",
                        "current_decision",
                        "current_rate_change_bps",
                        "next_meeting_id",
                        "next_decision",
                        "next_rate_change_bps",
                        "statement_score",
                        "speech_signal_score",
                    ]
                ].fillna("n/a"),
                hovertemplate=(
                    "Meeting: %{customdata[0]}<br>"
                    "Current decision: %{customdata[1]} (%{customdata[2]} bps)<br>"
                    "FGI: %{y:.2f}<br>"
                    "Statement score: %{customdata[6]}<br>"
                    "Speech signal: %{customdata[7]}<br>"
                    "Next meeting: %{customdata[3]}<br>"
                    "Next decision: %{customdata[4]} (%{customdata[5]} bps)<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timeline["date"],
                y=timeline["statement_score"],
                mode="lines",
                name="Statement score",
                line={"dash": "dot", "color": "#23d3b6"},
                hovertemplate="%{x|%Y-%m-%d}<br>Statement score: %{y:.2f}<extra></extra>",
            )
        )
        if "target_midpoint" in timeline.columns and timeline["target_midpoint"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=timeline["date"],
                    y=timeline["target_midpoint"],
                    mode="lines+markers",
                    name="Target rate midpoint",
                    yaxis="y2",
                    line={"color": "#f0b429", "width": 2},
                    marker={"size": 7},
                    customdata=timeline[["meeting_id", "target_midpoint", "next_target_midpoint"]].fillna("n/a"),
                    hovertemplate=(
                        "Meeting: %{customdata[0]}<br>"
                        "Current target midpoint: %{customdata[1]}%<br>"
                        "Next target midpoint: %{customdata[2]}%<extra></extra>"
                    ),
                )
            )
        for year in range(2021, 2027):
            fig.add_vline(x=pd.Timestamp(f"{year}-01-01"), line_dash="dot", line_color="#263846")
        fig.add_hline(y=0, line_dash="dot", line_color="#91a6b2")
        fig.update_layout(
            template="plotly_dark",
            height=520,
            paper_bgcolor="#0f1b24",
            plot_bgcolor="#0f1b24",
            font={"color": "#c3d3da"},
            title="Forward Guidance Timeline: Current Meeting to Next Decision",
            xaxis_title="Meeting date",
            yaxis_title="Score",
            yaxis2={
                "title": {"text": "Fed funds target midpoint (%)", "font": {"color": "#f0b429"}},
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "tickfont": {"color": "#f0b429"},
            },
            legend={"orientation": "h", "y": -0.22},
        )
        fig.update_xaxes(range=[pd.Timestamp("2021-01-01"), pd.Timestamp("2026-12-31")], gridcolor="#223541")
        fig.update_yaxes(gridcolor="#223541")
        st.plotly_chart(fig, width="stretch", theme=None)
    else:
        st.line_chart(timeline.set_index("date")[["fgi", "statement_score", "speech_signal_score"]])

    timeline_table = timeline[
        [
            "meeting_id",
            "meeting_date",
            "current_decision",
            "current_rate_change_bps",
            "target_midpoint",
            "statement_score",
            "speech_signal_score",
            "fgi",
            "fgi_classification",
            "next_meeting_id",
            "next_decision",
            "next_rate_change_bps",
            "next_target_midpoint",
            "next_meeting_score",
        ]
    ].rename(
        columns={
            "meeting_date": "current_meeting_date",
            "current_rate_change_bps": "current_change_bps",
            "target_midpoint": "current_target_midpoint_pct",
            "next_rate_change_bps": "next_change_bps",
            "next_target_midpoint": "next_target_midpoint_pct",
        }
    )
    st.dataframe(timeline_table, width="stretch", hide_index=True)

    options = cycles.sort_values("date", ascending=False)["meeting_id"].tolist()
    default_index = 1 if len(options) > 1 else 0
    selected_id = st.selectbox("Current meeting", options, index=default_index, key="cycle_bridge_meeting")
    row = cycles[cycles["meeting_id"].eq(selected_id)].iloc[0]
    current_docs = docs[docs["meeting_id"].eq(selected_id)].copy()
    speech_types = {"speech", "interview", "testimony"}
    speech_docs = current_docs[current_docs["document_subtype"].isin(speech_types)].sort_values("date")
    statement_docs = current_docs[current_docs["document_subtype"].eq("statement")]
    statement_doc_ids = statement_docs["doc_id"].dropna().tolist()
    speech_doc_ids = speech_docs["doc_id"].dropna().tolist()

    render_kpis(
        [
            ("Current Meeting", row["meeting_id"], str(row["meeting_date"])),
            ("Current Decision", row.get("current_decision", "unknown"), f"{row.get('current_rate_change_bps', np.nan):+.0f} bps" if pd.notna(row.get("current_rate_change_bps")) else "bps n/a"),
            ("Current Target", f"{row.get('target_midpoint', np.nan):.3f}%" if pd.notna(row.get("target_midpoint")) else "n/a", "Fed funds midpoint"),
            ("Statement Score", f"{row['statement_score']:.2f}" if pd.notna(row["statement_score"]) else "n/a", "Current FOMC wording"),
            ("Speech Signal", f"{row['speech_signal_score']:.2f}" if pd.notna(row["speech_signal_score"]) else "n/a", f"{int(row.get('speech_documents', 0))} speeches/interviews/testimony"),
            ("FGI", f"{row['fgi']:.2f}" if pd.notna(row["fgi"]) else "n/a", row.get("fgi_classification", "")),
            ("Next Meeting", row.get("next_meeting_id", "n/a"), row.get("next_decision", "n/a") if pd.notna(row.get("next_decision")) else "no next meeting"),
            ("Next Target", f"{row.get('next_target_midpoint', np.nan):.3f}%" if pd.notna(row.get("next_target_midpoint")) else "n/a", "Next fed funds midpoint"),
        ]
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Current Policy Decision and Forward-Guidance Language")
        current_table = pd.DataFrame(
            [
                {
                    "meeting_id": row["meeting_id"],
                    "meeting_date": row["meeting_date"],
                    "current_decision": row.get("current_decision", ""),
                    "current_rate_change_bps": row.get("current_rate_change_bps", np.nan),
                    "target_midpoint": row.get("target_midpoint", np.nan),
                    "meeting_cycle_score": row.get("meeting_cycle_score", np.nan),
                    "statement_score": row.get("statement_score", np.nan),
                    "minutes_score": row.get("minutes_score", np.nan),
                    "press_conference_score": row.get("press_conference_score", np.nan),
                    "speech_signal_score": row.get("speech_signal_score", np.nan),
                    "fgi": row.get("fgi", np.nan),
                    "fgi_classification": row.get("fgi_classification", ""),
                }
            ]
        )
        st.dataframe(current_table, width="stretch", hide_index=True)
        st.subheader("Core Statement Phrase Matches")
        st.dataframe(phrase_summary_for_docs(statement_doc_ids, phrases, {"hawkish", "dovish"}).head(15), width="stretch", hide_index=True)
    with c2:
        st.subheader("Next Meeting Policy Decision")
        next_table = pd.DataFrame(
            [
                {
                    "next_meeting_id": row.get("next_meeting_id", ""),
                    "next_decision": row.get("next_decision", ""),
                    "next_rate_change_bps": row.get("next_rate_change_bps", np.nan),
                    "next_target_midpoint": row.get("next_target_midpoint", np.nan),
                    "next_meeting_score": row.get("next_meeting_score", np.nan),
                }
            ]
        )
        st.dataframe(next_table, width="stretch", hide_index=True)
        st.subheader("Intermeeting Speech Phrase Matches")
        st.dataframe(phrase_summary_for_docs(speech_doc_ids, phrases, {"hawkish", "dovish"}).head(15), width="stretch", hide_index=True)

    st.subheader("Speeches, Interviews, and Testimony Before the Next Meeting")
    if speech_docs.empty:
        st.info("No speeches, interviews, or testimony linked to this meeting window.")
    else:
        speech_table = speech_docs[
            [
                "date_au",
                "speaker",
                "title",
                "document_subtype",
                "score",
                "classification",
                "confidence",
                "total_phrase_matches",
                "source_url",
            ]
        ].rename(
            columns={
                "date_au": "date",
                "document_subtype": "type",
                "total_phrase_matches": "phrase_matches",
            }
        )
        st.dataframe(speech_table, width="stretch", hide_index=True)

        if HAS_PLOTLY:
            fig = px.bar(
                speech_docs,
                x="date",
                y="score",
                color="speaker",
                hover_data=["title", "document_subtype", "classification", "total_phrase_matches"],
                title="Intermeeting Speech Scores",
            )
            fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="#0f1b24", plot_bgcolor="#0f1b24")
            fig.add_hline(y=0, line_dash="dot", line_color="#91a6b2")
            st.plotly_chart(fig, width="stretch", theme=None)
    st.markdown("</div>", unsafe_allow_html=True)


def page_lab_forward_guidance(cycles: pd.DataFrame) -> None:
    st.markdown('<div class="research-panel"><div class="panel-title">2. Meeting-to-Meeting Forward Guidance</div>', unsafe_allow_html=True)
    st.markdown("Can wording inside the current FOMC statement provide forward guidance about the next meeting?")
    table = cycles[
        [
            "meeting_id",
            "meeting_date",
            "meeting_cycle_score",
            "statement_score",
            "next_decision",
            "next_rate_change_bps",
            "next_meeting_score",
        ]
    ].copy()
    st.dataframe(table, width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        hist_data = cycles[cycles["next_decision"].isin(["cut", "hold", "hike"])]
        if hist_data.empty:
            st.info("No statement-score decision groups available.")
        else:
            fig = px.histogram(
                hist_data,
                x="statement_score",
                color="next_decision",
                nbins=18,
                barmode="overlay",
                title="Statement Scores Before Cuts, Holds, and Hikes",
                color_discrete_map={"hike": "#e5242a", "hold": "#7b8794", "cut": "#3169df"},
            )
            fig.update_layout(template="plotly_dark", height=390)
            st.plotly_chart(fig, width="stretch", theme=None)
    with c2:
        st.dataframe(research_metric_table(cycles, "statement_score", "statement_score"), width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def phrase_decision_table(phrases: pd.DataFrame, docs: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    if phrases.empty:
        return pd.DataFrame()
    doc_meetings = docs[["doc_id", "meeting_id"]].drop_duplicates()
    decision_map = cycles[["meeting_id", "next_decision", "next_rate_change_bps"]]
    data = phrases.merge(doc_meetings, on="doc_id", how="left").merge(decision_map, on="meeting_id", how="left")
    return data


def page_lab_phrase_evolution(phrases: pd.DataFrame, docs: pd.DataFrame, cycles: pd.DataFrame) -> None:
    st.markdown('<div class="research-panel"><div class="panel-title">3. Phrase Evolution</div>', unsafe_allow_html=True)
    st.markdown("Which phrases appear before hikes, holds, and cuts?")
    phrase_decisions = phrase_decision_table(phrases, docs, cycles)
    if phrase_decisions.empty:
        st.info("No phrase-match data available.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    tabs = st.tabs(["Before hikes", "Before holds", "Before cuts", "Year heatmap"])
    for tab, decision in zip(tabs[:3], ["hike", "hold", "cut"]):
        with tab:
            subset = phrase_decisions[phrase_decisions["next_decision"].eq(decision)]
            summary = (
                subset.groupby(["phrase", "direction"], as_index=False)
                .agg(
                    occurrences=("count", "sum"),
                    average_contribution=("scaled_contribution", "mean"),
                    average_next_rate_change=("next_rate_change_bps", "mean"),
                )
                .sort_values("occurrences", ascending=False)
                .head(25)
            )
            st.dataframe(summary, width="stretch", hide_index=True)
    with tabs[3]:
        heat = (
            phrase_decisions.groupby(["phrase", "year"], as_index=False)["count"].sum()
            .sort_values("count", ascending=False)
        )
        top_phrases = heat.groupby("phrase")["count"].sum().sort_values(ascending=False).head(20).index
        pivot = heat[heat["phrase"].isin(top_phrases)].pivot_table(index="phrase", columns="year", values="count", aggfunc="sum", fill_value=0)
        if pivot.empty:
            st.info("No phrase-year frequency data available.")
        else:
            fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", title="Phrase Frequency by Year")
            fig.update_layout(template="plotly_dark", height=520)
            st.plotly_chart(fig, width="stretch", theme=None)
    st.markdown("</div>", unsafe_allow_html=True)


def page_lab_transition_matrix(cycles: pd.DataFrame) -> None:
    st.markdown('<div class="research-panel"><div class="panel-title">4. Decision Transition Matrix</div>', unsafe_allow_html=True)
    matrix_data = cycles[cycles["current_decision"].isin(["cut", "hold", "hike"]) & cycles["next_decision"].isin(["cut", "hold", "hike"])]
    if matrix_data.empty:
        st.info("No decision transition data available.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    counts = pd.crosstab(matrix_data["current_decision"], matrix_data["next_decision"]).reindex(index=["cut", "hold", "hike"], columns=["cut", "hold", "hike"], fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Transition Counts")
        st.dataframe(counts, width="stretch")
    with c2:
        st.subheader("Transition Probabilities")
        st.dataframe((probs * 100).round(1).astype(str) + "%", width="stretch")
    fig = px.imshow(probs, text_auto=".0%", color_continuous_scale="Blues", title="Current Decision to Next Decision")
    fig.update_layout(template="plotly_dark", height=430)
    st.plotly_chart(fig, width="stretch", theme=None)
    st.markdown("</div>", unsafe_allow_html=True)


def top_meeting_phrases(meeting_id: str, docs: pd.DataFrame, phrases: pd.DataFrame) -> pd.DataFrame:
    if phrases.empty:
        return pd.DataFrame()
    doc_ids = docs[docs["meeting_id"].eq(meeting_id)]["doc_id"].unique()
    return (
        phrases[phrases["doc_id"].isin(doc_ids)]
        .groupby(["phrase", "direction"], as_index=False)
        .agg(occurrences=("count", "sum"), average_contribution=("scaled_contribution", "mean"))
        .sort_values("occurrences", ascending=False)
        .head(8)
    )


def page_lab_case_studies(cycles: pd.DataFrame, docs: pd.DataFrame, phrases: pd.DataFrame) -> None:
    st.markdown('<div class="research-panel"><div class="panel-title">5. Case Studies</div>', unsafe_allow_html=True)
    if cycles.empty:
        st.info("No meeting cycles available.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    cases = {
        "Most hawkish meeting": cycles.sort_values("meeting_cycle_score", ascending=False).head(1),
        "Most dovish meeting": cycles.sort_values("meeting_cycle_score", ascending=True).head(1),
        "Largest shift in score": cycles.assign(abs_shift=cycles["meeting_cycle_score"].diff().abs()).sort_values("abs_shift", ascending=False).head(1),
        "Largest speech score change": cycles.assign(abs_speech_shift=cycles["speech_score_change"].abs()).sort_values("abs_speech_shift", ascending=False).head(1),
        "Largest statement score change": cycles.assign(abs_statement_shift=cycles["statement_score_change"].abs()).sort_values("abs_statement_shift", ascending=False).head(1),
        "Largest speech/statement disagreement": cycles.assign(abs_gap=cycles["speech_statement_gap"].abs()).sort_values("abs_gap", ascending=False).head(1),
    }
    for label, frame in cases.items():
        if frame.empty:
            continue
        row = frame.iloc[0]
        with st.expander(label, expanded=label in {"Most hawkish meeting", "Most dovish meeting"}):
            render_kpis(
                [
                    ("Meeting", row["meeting_id"], str(row["meeting_date"])),
                    ("Decision", row.get("current_decision", "unknown"), f"Next: {row.get('next_decision', 'unknown')}"),
                    ("Meeting Score", f"{row['meeting_cycle_score']:.2f}" if pd.notna(row["meeting_cycle_score"]) else "n/a", ""),
                    ("Speech Signal", f"{row['speech_signal_score']:.2f}" if pd.notna(row["speech_signal_score"]) else "n/a", ""),
                ]
            )
            meeting_docs = docs[docs["meeting_id"].eq(row["meeting_id"])][["date_au", "speaker", "title", "document_subtype", "score", "classification"]].sort_values("date_au")
            st.dataframe(meeting_docs, width="stretch", hide_index=True)
            st.subheader("Top Phrase Matches")
            st.dataframe(top_meeting_phrases(row["meeting_id"], docs, phrases), width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def conclusion_sentence(label: str, values: pd.Series) -> str:
    available = values.dropna()
    if available.empty:
        return f"{label}: not enough data."
    most_hawkish = available.idxmax()
    most_dovish = available.idxmin()
    return f"{label}: average tone is most hawkish before {most_hawkish}s and most dovish before {most_dovish}s."


def page_lab_conclusions(cycles: pd.DataFrame, docs: pd.DataFrame) -> None:
    st.markdown('<div class="research-panel"><div class="panel-title">6. Research Conclusions</div>', unsafe_allow_html=True)
    speech_means = cycles.groupby("next_decision")["speech_signal_score"].mean()
    statement_means = cycles.groupby("next_decision")["statement_score"].mean()
    summary = pd.DataFrame(
        {
            "average_speech_score": speech_means,
            "average_statement_score": statement_means,
        }
    ).reset_index()
    st.dataframe(summary, width="stretch", hide_index=True)
    observations = [
        conclusion_sentence("Speech scores", speech_means),
        conclusion_sentence("Statement scores", statement_means),
    ]
    if {"cut", "hold", "hike"}.intersection(statement_means.index):
        spread_statement = statement_means.max() - statement_means.min()
        spread_speech = speech_means.max() - speech_means.min()
        if pd.notna(spread_statement) and pd.notna(spread_speech):
            stronger = "Statements contain stronger forward-guidance separation than speeches." if spread_statement > spread_speech else "Speech scores show stronger separation than statements in this sample."
            observations.append(stronger)
    powell = docs[docs["speaker"].str.contains("Powell", case=False, na=False)]
    observations.append(
        "Powell speeches are present in the intermeeting communication set and can be reviewed case-by-case."
        if not powell.empty
        else "Powell speech data was not found in the filtered sample."
    )
    observations.append("These are descriptive observations only, not predictions.")
    for item in observations:
        st.markdown(f"- {item}")
    st.markdown("</div>", unsafe_allow_html=True)


def page_lab_fgi(cycles: pd.DataFrame) -> None:
    st.markdown('<div class="research-panel"><div class="panel-title">Bonus. Forward Guidance Indicator</div>', unsafe_allow_html=True)
    st.markdown("FGI = 40% statement, 30% minutes, 20% press conference, 10% intermeeting speeches. Missing components are reweighted.")
    table = cycles[
        [
            "meeting_id",
            "meeting_date",
            "fgi",
            "fgi_classification",
            "statement_score",
            "minutes_score",
            "press_conference_score",
            "speech_signal_score",
            "next_decision",
            "next_rate_change_bps",
        ]
    ].copy()
    st.dataframe(table, width="stretch", hide_index=True)
    data = cycles.dropna(subset=["fgi", "next_rate_change_bps"])
    if not data.empty:
        fig = px.scatter(
            data,
            x="fgi",
            y="next_rate_change_bps",
            color="next_decision",
            hover_data=["meeting_id", "meeting_date"],
            title="FGI vs Next Rate Change",
            color_discrete_map={"hike": "#e5242a", "hold": "#7b8794", "cut": "#3169df", "unknown": "#9aa5b1"},
        )
        fig.update_layout(template="plotly_dark", height=420)
        fig.add_hline(y=0, line_dash="dot", line_color="#52677a")
        fig.add_vline(x=0, line_dash="dot", line_color="#52677a")
        st.plotly_chart(fig, width="stretch", theme=None)
    corr = data["fgi"].corr(data["next_rate_change_bps"]) if len(data) > 1 else np.nan
    render_kpis([("FGI Correlation", f"{corr:.3f}" if pd.notna(corr) else "n/a", "FGI vs next meeting rate change")])
    st.markdown("</div>", unsafe_allow_html=True)


def render_forward_guidance_lab(docs: pd.DataFrame, meetings: pd.DataFrame, phrases: pd.DataFrame) -> None:
    st.markdown('<div class="terminal-title">FED Forward Guidance Research Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Research and pattern discovery for whether Federal Reserve communication from 2021 onward contains information about the next FOMC meeting. This page does not make predictions.</div>',
        unsafe_allow_html=True,
    )
    cycles = build_research_cycles(docs, meetings)
    filtered_docs, filtered_cycles, filtered_phrases = apply_lab_filters(docs, cycles, phrases)
    if load_policy_decisions().empty:
        st.warning("Policy decision file not found. Decisions and rate changes are inferred from statement target-range text where possible.")
    page_lab_meeting_cycle_bridge(filtered_cycles, filtered_docs, filtered_phrases)
    page_lab_speech_signal(filtered_cycles)
    page_lab_forward_guidance(filtered_cycles)
    page_lab_phrase_evolution(filtered_phrases, filtered_docs, filtered_cycles)
    page_lab_transition_matrix(filtered_cycles)
    page_lab_case_studies(filtered_cycles, filtered_docs, filtered_phrases)
    page_lab_conclusions(filtered_cycles, filtered_docs)
    page_lab_fgi(filtered_cycles)


def plot_line(df: pd.DataFrame, x: str, y: list[str] | str, title: str, y_title: str = "Score") -> None:
    if df.empty:
        st.info("No data available for this chart.")
        return
    if HAS_PLOTLY:
        fig = px.line(df, x=x, y=y, markers=True, title=title)
        fig.update_layout(template="plotly_dark", height=420, yaxis_title=y_title, xaxis_title="")
        fig.add_hline(y=0, line_dash="dot", line_color="#8ea3ad")
        st.plotly_chart(fig, width="stretch", theme=None)
    else:
        chart_df = df.set_index(x)[y] if isinstance(y, list) else df.set_index(x)[[y]]
        st.line_chart(chart_df)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    if df.empty:
        st.info("No data available for this chart.")
        return
    if HAS_PLOTLY:
        fig = px.bar(df, x=x, y=y, color=color, title=title)
        fig.update_layout(template="plotly_dark", height=390, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, width="stretch", theme=None)
    else:
        st.bar_chart(df.set_index(x)[y])


def plot_hist(df: pd.DataFrame, column: str, title: str) -> None:
    if df.empty:
        st.info("No data available for this chart.")
        return
    if HAS_PLOTLY:
        fig = px.histogram(df, x=column, nbins=30, title=title)
        fig.update_layout(template="plotly_dark", height=360, xaxis_title="Score", yaxis_title="Documents")
        fig.add_vline(x=0, line_dash="dot", line_color="#8ea3ad")
        st.plotly_chart(fig, width="stretch", theme=None)
    else:
        st.bar_chart(np.histogram(df[column].dropna(), bins=30)[0])


def regime_classification(delta: float) -> str:
    if pd.isna(delta):
        return "Unavailable"
    if delta >= 2.5:
        return "Much More Hawkish than 2021"
    if delta >= 1.0:
        return "More Hawkish than 2021"
    if delta > -1.0:
        return "Similar to 2021"
    if delta > -2.5:
        return "More Dovish than 2021"
    return "Much More Dovish than 2021"


def classify_score(score: float) -> tuple[str, str]:
    if pd.isna(score):
        return "Unavailable", "none"
    if score >= 8:
        classification = "Strongly Hawkish"
    elif score >= 3:
        classification = "Hawkish"
    elif score >= 0.5:
        classification = "Neutral-Hawkish"
    elif score >= -0.5:
        classification = "Neutral"
    elif score > -3:
        classification = "Neutral-Dovish"
    elif score > -8:
        classification = "Dovish"
    else:
        classification = "Strongly Dovish"
    lean = "hawkish" if score > 0.5 else "dovish" if score < -0.5 else "none"
    return classification, lean


def dashboard_meeting_score(row: pd.Series) -> float:
    weights = {
        "statement_score": 0.45,
        "minutes_score": 0.25,
        "press_conference_score": 0.20,
        "speech_avg_score": 0.10,
    }
    numerator = 0.0
    denominator = 0.0
    for column, weight in weights.items():
        value = row.get(column)
        if pd.notna(value):
            numerator += float(value) * weight
            denominator += weight
    if denominator:
        return numerator / denominator
    fallback = row.get("meeting_score")
    return float(fallback) if pd.notna(fallback) else np.nan


def filter_data(docs: pd.DataFrame, meetings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = docs["date"].max() if not docs.empty else DEFAULT_START
    min_date = DEFAULT_START
    st.sidebar.header("FED Filters")
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date.date(), max_date.date() if pd.notna(max_date) else min_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date() if pd.notna(max_date) else min_date.date(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start, end = min_date, max_date

    speaker_options = sorted(docs["speaker"].dropna().unique().tolist())
    subtype_options = sorted(docs["document_subtype"].dropna().unique().tolist())
    class_options = sorted(docs["classification"].dropna().unique().tolist())
    selected_speakers = st.sidebar.multiselect("Speaker", speaker_options)
    selected_subtypes = st.sidebar.multiselect("Document subtype", subtype_options)
    selected_classes = st.sidebar.multiselect("Classification", class_options)

    filtered_docs = docs[(docs["date"] >= start) & (docs["date"] <= end)].copy()
    if selected_speakers:
        filtered_docs = filtered_docs[filtered_docs["speaker"].isin(selected_speakers)]
    if selected_subtypes:
        filtered_docs = filtered_docs[filtered_docs["document_subtype"].isin(selected_subtypes)]
    if selected_classes:
        filtered_docs = filtered_docs[filtered_docs["classification"].isin(selected_classes)]

    filtered_meetings = meetings[(meetings["date"] >= start) & (meetings["date"] <= end)].copy()
    return filtered_docs, filtered_meetings


def page_overview(docs: pd.DataFrame, meetings: pd.DataFrame, speakers: pd.DataFrame) -> None:
    st.markdown('<div class="terminal-title">Per-bank Hawkometers</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Federal Reserve communication dashboard covering meeting-cycle tone, speaker sentiment, phrase evidence, and latest scored appearances.</div>',
        unsafe_allow_html=True,
    )
    latest_meeting = meetings.sort_values("date").tail(1)
    baseline = meetings[meetings["year"].eq(BASELINE_YEAR)]["meeting_cycle_score"].mean()
    current = latest_meeting["meeting_cycle_score"].iloc[0] if not latest_meeting.empty else np.nan
    delta = current - baseline if pd.notna(current) and pd.notna(baseline) else np.nan
    shift = recent_meeting_shift(meetings)
    hawkish_doc = docs.sort_values("score", ascending=False).head(1)
    dovish_doc = docs.sort_values("score", ascending=True).head(1)

    st.markdown(f'<div class="bank-grid">{fed_bank_card(current, shift)}</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="research-panel">
          <div class="panel-title">Committee scores at a glance</div>
          <div class="section-copy">Dashboard meeting score using Statement 45%, Minutes 25%, Press conference 20%, Speeches 10%. Hover chart points and inspect pages for the source document breakdown.</div>
          <div class="bank-grid">{fed_bank_card(current, shift)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="terminal-title">Overview</div>', unsafe_allow_html=True)
    render_kpis(
        [
            ("Documents Scored", f"{len(docs):,}", "FED-only corpus"),
            ("Meetings Scored", f"{len(meetings):,}", "FOMC meeting cycles"),
            ("Speakers", f"{docs['speaker'].nunique():,}", "Unique speaker labels"),
            ("Average FED Score", f"{docs['score'].mean():.2f}" if not docs.empty else "n/a", "-10 to +10 scale"),
            ("Latest Meeting", f"{current:.2f}" if pd.notna(current) else "n/a", latest_meeting["lean"].iloc[0] if not latest_meeting.empty else ""),
            ("Most Hawkish Doc", f"{hawkish_doc['score'].iloc[0]:.2f}" if not hawkish_doc.empty else "n/a", hawkish_doc["title"].iloc[0][:54] if not hawkish_doc.empty else ""),
            ("Most Dovish Doc", f"{dovish_doc['score'].iloc[0]:.2f}" if not dovish_doc.empty else "n/a", dovish_doc["title"].iloc[0][:54] if not dovish_doc.empty else ""),
            ("Regime vs 2021", regime_classification(delta), f"Delta: {delta:.2f}" if pd.notna(delta) else "Delta unavailable"),
        ]
    )

    st.subheader("Current Regime vs 2021 Baseline")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "baseline_score_2021": baseline,
                    "current_score": current,
                    "delta_from_2021": delta,
                    "regime_classification": regime_classification(delta),
                }
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        plot_hist(docs, "score", "Score Distribution")
    with c2:
        breakdown = docs.groupby("document_subtype", dropna=False).size().reset_index(name="documents")
        plot_bar(breakdown, "document_subtype", "documents", "Document Type Breakdown")

    st.subheader("Latest scored appearances")
    st.dataframe(
        docs.sort_values("date", ascending=False)[
            ["date_au", "speaker", "title", "document_subtype", "score", "classification", "total_phrase_matches"]
        ].head(10),
        width="stretch",
        hide_index=True,
    )


def page_meeting_timeline(meetings: pd.DataFrame) -> None:
    st.markdown('<div class="terminal-title">Meeting Timeline</div>', unsafe_allow_html=True)
    timeline = meetings.sort_values("date").copy()
    plot_line(timeline, "date", ["meeting_cycle_score", "rolling_3", "rolling_6"], "FED Meeting Cycle Score")
    if HAS_PLOTLY and not timeline.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timeline["date"], y=timeline["meeting_cycle_score"], mode="lines+markers", name="Meeting"))
        fig.add_trace(go.Scatter(x=timeline["date"], y=timeline["rolling_3"], mode="lines", name="Rolling 3"))
        fig.add_trace(go.Scatter(x=timeline["date"], y=timeline["rolling_6"], mode="lines", name="Rolling 6"))
        for year in range(2021, 2027):
            fig.add_vline(x=pd.Timestamp(f"{year}-01-01"), line_dash="dot", line_color="#41515a")
        fig.add_hline(y=0, line_dash="dot", line_color="#8ea3ad")
        fig.update_layout(template="plotly_dark", height=440, title="Meeting Timeline with Year Markers", yaxis_title="Score")
        st.plotly_chart(fig, width="stretch", theme=None)

    table = timeline[
        [
            "meeting_id",
            "meeting_date",
            "statement_score",
            "minutes_score",
            "press_conference_score",
            "speech_avg_score",
            "meeting_cycle_score",
            "meeting_cycle_lean",
            "documents_scored",
        ]
    ].rename(columns={"speech_avg_score": "speech_score"})
    st.dataframe(table, width="stretch", hide_index=True)
    csv_download(table, "Export meeting table", "fed_meeting_timeline.csv")


def speaker_rollups(docs: pd.DataFrame, speaker: str) -> pd.DataFrame:
    data = docs[docs["speaker"].eq(speaker)].sort_values("date").copy()
    if data.empty:
        return data
    data = data.set_index("date")
    data["rolling_90d_score"] = data["score"].rolling("90D", min_periods=1).mean()
    data["rolling_30d_score"] = data["score"].rolling("30D", min_periods=1).mean()
    data["prior_60d"] = data["score"].shift(1).rolling("60D", min_periods=1).mean()
    data["shift_30_vs_prior_60"] = data["rolling_30d_score"] - data["prior_60d"]
    return data.reset_index()


def page_speaker_dashboard(docs: pd.DataFrame, speaker_scores: pd.DataFrame) -> None:
    st.markdown('<div class="terminal-title">Speaker Dashboard</div>', unsafe_allow_html=True)
    speaker_options = sorted(docs["speaker"].dropna().unique().tolist())
    default_speaker = next((s for s in speaker_options if "Powell" in s), speaker_options[0] if speaker_options else "")
    selected = st.selectbox("Speaker", speaker_options, index=speaker_options.index(default_speaker) if default_speaker in speaker_options else 0)
    roll = speaker_rollups(docs, selected)
    plot_line(roll, "date", ["rolling_90d_score", "rolling_30d_score", "shift_30_vs_prior_60"], f"{selected} Rolling Score")

    st.subheader("Speaker Ranking")
    ranking = speaker_scores.sort_values("average_score", ascending=False)
    st.dataframe(ranking, width="stretch", hide_index=True)
    csv_download(ranking, "Export speaker table", "fed_speaker_scores_filtered.csv")

    st.subheader("Speaker Shift Alerts")
    alert_rows = []
    for speaker in speaker_options:
        speaker_data = speaker_rollups(docs, speaker)
        if speaker_data.empty:
            continue
        latest = speaker_data.sort_values("date").tail(1).iloc[0]
        shift = latest.get("shift_30_vs_prior_60")
        if pd.notna(shift) and abs(float(shift)) >= 2:
            alert_rows.append({"speaker": speaker, "latest_date": latest["date_au"], "shift_30_vs_prior_60": shift})
    alerts = pd.DataFrame(alert_rows).sort_values("shift_30_vs_prior_60", ascending=False) if alert_rows else pd.DataFrame()
    st.dataframe(alerts, width="stretch", hide_index=True)

    st.subheader("Key FED Speakers")
    key = ranking[ranking["speaker"].str.contains("|".join(KEY_SPEAKERS), case=False, na=False)]
    st.dataframe(key, width="stretch", hide_index=True)


def page_meeting_explorer(docs: pd.DataFrame, meetings: pd.DataFrame) -> None:
    st.markdown('<div class="terminal-title">Meeting Explorer</div>', unsafe_allow_html=True)
    ids = meetings["meeting_id"].dropna().tolist()
    if not ids:
        st.info("No meetings available.")
        return
    selected = st.selectbox("Meeting ID", ids, index=len(ids) - 1)
    meeting = meetings[meetings["meeting_id"].eq(selected)].iloc[0]
    render_kpis(
        [
            ("Meeting Score", f"{meeting['meeting_cycle_score']:.2f}", meeting.get("classification", "")),
            ("Meeting Lean", meeting.get("lean", ""), meeting.get("meeting_date", "")),
            ("Statement", f"{meeting['statement_score']:.2f}" if pd.notna(meeting["statement_score"]) else "n/a", ""),
            ("Minutes", f"{meeting['minutes_score']:.2f}" if pd.notna(meeting["minutes_score"]) else "n/a", ""),
            ("Press Conference", f"{meeting['press_conference_score']:.2f}" if pd.notna(meeting["press_conference_score"]) else "n/a", ""),
            ("Speech", f"{meeting['speech_avg_score']:.2f}" if pd.notna(meeting["speech_avg_score"]) else "n/a", ""),
        ]
    )
    table = docs[docs["meeting_id"].eq(selected)][
        ["date_au", "speaker", "title", "document_subtype", "clipped_score", "classification", "total_phrase_matches"]
    ].sort_values("date_au")
    st.dataframe(table, width="stretch", hide_index=True)


def matched_phrases_for_doc(row: pd.Series, direction: str) -> str:
    phrases = [
        f"{m.get('phrase', '')} ({m.get('count', 1)})"
        for m in parse_match_blob(row.get("matched_phrases_json", ""))
        if m.get("direction") == direction
    ]
    return "; ".join(phrases)


def page_document_explorer(docs: pd.DataFrame) -> None:
    st.markdown('<div class="terminal-title">Document Explorer</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    min_matches = c1.number_input("Minimum phrase matches", min_value=0, value=0)
    keyword = c2.text_input("Title keyword")
    max_rows = c3.number_input("Rows", min_value=10, max_value=500, value=100, step=10)
    table = docs[docs["total_phrase_matches"].fillna(0) >= min_matches].copy()
    if keyword:
        table = table[table["title"].str.contains(keyword, case=False, na=False)]
    table["matched_hawkish_phrases"] = table.apply(lambda row: matched_phrases_for_doc(row, "hawkish"), axis=1)
    table["matched_dovish_phrases"] = table.apply(lambda row: matched_phrases_for_doc(row, "dovish"), axis=1)
    display_cols = [
        "date_au",
        "speaker",
        "title",
        "document_subtype",
        "clipped_score",
        "classification",
        "matched_hawkish_phrases",
        "matched_dovish_phrases",
    ]
    st.dataframe(table.sort_values("date", ascending=False)[display_cols].head(max_rows), width="stretch", hide_index=True)
    csv_download(table[display_cols], "Export filtered documents", "fed_documents_filtered.csv")

    options = (table["doc_id"] + " | " + table["date_au"] + " | " + table["title"]).tolist()
    if not options:
        st.info("No document selected.")
        return
    selected = st.selectbox("Inspect document", options)
    doc_id = selected.split(" | ", 1)[0]
    row = table[table["doc_id"].eq(doc_id)].iloc[0]
    render_kpis(
        [
            ("Score", f"{row['score']:.2f}", row["classification"]),
            ("Confidence", f"{row['confidence']:.2f}", row["lean"]),
            ("Phrase Matches", int(row["total_phrase_matches"]), row["document_subtype"]),
        ]
    )
    st.json(
        {
            "doc_id": row["doc_id"],
            "date": row["date_au"],
            "speaker": row["speaker"],
            "title": row["title"],
            "source_url": row["source_url"],
            "text_file": row["text_file"],
        }
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Matched Hawkish Phrases")
        st.write(row["matched_hawkish_phrases"] or "None")
    with c2:
        st.subheader("Matched Dovish Phrases")
        st.write(row["matched_dovish_phrases"] or "None")
    with st.expander("Document Text"):
        path = Path(str(row["text_file"]))
        if path.exists():
            st.text(path.read_text(encoding="utf-8", errors="replace")[:30000])
        else:
            st.warning(f"Text file not found: {path}")


def page_phrase_explorer(phrases: pd.DataFrame) -> None:
    st.markdown('<div class="terminal-title">Phrase Match Explorer</div>', unsafe_allow_html=True)
    if phrases.empty:
        st.info("No phrase matches available.")
        return
    hawkish = phrases[phrases["direction"].eq("hawkish")].groupby("phrase", as_index=False).agg(
        occurrences=("count", "sum"), average_contribution=("scaled_contribution", "mean")
    )
    dovish = phrases[phrases["direction"].eq("dovish")].groupby("phrase", as_index=False).agg(
        occurrences=("count", "sum"), average_contribution=("scaled_contribution", "mean")
    )
    c1, c2 = st.columns(2)
    with c1:
        plot_bar(hawkish.sort_values("occurrences", ascending=False).head(15), "phrase", "occurrences", "Top Hawkish Phrases")
    with c2:
        plot_bar(dovish.sort_values("occurrences", ascending=False).head(15), "phrase", "occurrences", "Top Dovish Phrases")

    freq = phrases.groupby(["phrase", "direction", "theme"], as_index=False).agg(
        occurrences=("count", "sum"),
        documents=("doc_id", "nunique"),
        average_contribution=("scaled_contribution", "mean"),
    ).sort_values("occurrences", ascending=False)
    st.dataframe(freq, width="stretch", hide_index=True)

    selected = st.selectbox("Phrase", freq["phrase"].tolist())
    selected_docs = phrases[phrases["phrase"].eq(selected)].sort_values("date", ascending=False)
    plot_line(selected_docs.groupby("date", as_index=False)["count"].sum(), "date", "count", f"Phrase Usage: {selected}")
    st.dataframe(
        selected_docs[["speaker", "title", "date_au", "score", "direction", "count"]].drop_duplicates(),
        width="stretch",
        hide_index=True,
    )


def yearly_summary(meetings: pd.DataFrame, docs: pd.DataFrame, phrases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in range(2021, 2027):
        m = meetings[meetings["year"].eq(year)]
        d = docs[docs["year"].eq(year)]
        p = phrases[phrases["year"].eq(year)]
        rows.append(
            {
                "year": year,
                "regime": REGIME_LABELS.get(year, ""),
                "average_meeting_score": m["meeting_cycle_score"].mean(),
                "average_statement_score": m["statement_score"].mean(),
                "average_minutes_score": m["minutes_score"].mean(),
                "average_speech_score": m["speech_avg_score"].mean(),
                "meetings": len(m),
                "documents": len(d),
                "average_phrase_matches": d["total_phrase_matches"].mean(),
                "average_hawkish_phrases": p[p["direction"].eq("hawkish")]["count"].mean(),
                "average_dovish_phrases": p[p["direction"].eq("dovish")]["count"].mean(),
                "most_hawkish_meeting": m.sort_values("meeting_cycle_score", ascending=False)["meeting_id"].head(1).iloc[0] if not m.empty else "",
                "most_dovish_meeting": m.sort_values("meeting_cycle_score", ascending=True)["meeting_id"].head(1).iloc[0] if not m.empty else "",
            }
        )
    result = pd.DataFrame(rows)
    baseline = result[result["year"].eq(2021)]["average_meeting_score"].iloc[0]
    result["change_from_previous_year"] = result["average_meeting_score"].diff()
    result["change_from_2021_baseline"] = result["average_meeting_score"] - baseline
    result["regime_classification"] = result["change_from_2021_baseline"].apply(regime_classification)
    return result


def page_regime_view(docs: pd.DataFrame, meetings: pd.DataFrame, phrases: pd.DataFrame) -> None:
    st.markdown('<div class="terminal-title">Regime View</div>', unsafe_allow_html=True)
    timeline = meetings.sort_values("date")
    if HAS_PLOTLY and not timeline.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timeline["date"], y=timeline["meeting_cycle_score"], mode="lines+markers", name="Meeting"))
        fig.add_trace(go.Scatter(x=timeline["date"], y=timeline["rolling_3"], mode="lines", name="Rolling 3"))
        fig.add_trace(go.Scatter(x=timeline["date"], y=timeline["rolling_6"], mode="lines", name="Rolling 6"))
        colors = ["rgba(35,211,182,0.08)", "rgba(47,208,125,0.10)", "rgba(238,188,80,0.10)", "rgba(92,167,255,0.10)", "rgba(240,93,94,0.09)", "rgba(170,130,255,0.10)"]
        for idx, year in enumerate(range(2021, 2027)):
            fig.add_vrect(
                x0=pd.Timestamp(f"{year}-01-01"),
                x1=pd.Timestamp(f"{year}-12-31"),
                fillcolor=colors[idx % len(colors)],
                line_width=0,
                annotation_text=f"{year}: {REGIME_LABELS[year]}",
                annotation_position="top left",
            )
        fig.add_hline(y=0, line_dash="dot", line_color="#8ea3ad")
        fig.update_layout(template="plotly_dark", height=470, title="Regime Timeline", yaxis_title="Score")
        st.plotly_chart(fig, width="stretch", theme=None)
    else:
        plot_line(timeline, "date", ["meeting_cycle_score", "rolling_3", "rolling_6"], "Regime Timeline")

    yearly = yearly_summary(meetings, docs, phrases)
    st.subheader("Regime Summary Cards")
    for _, row in yearly.iterrows():
        render_kpis(
            [
                (f"{int(row['year'])}", row["regime"], f"Meetings: {int(row['meetings'])} | Docs: {int(row['documents'])}"),
                ("Avg Meeting", f"{row['average_meeting_score']:.2f}" if pd.notna(row["average_meeting_score"]) else "n/a", ""),
                ("Avg Statement", f"{row['average_statement_score']:.2f}" if pd.notna(row["average_statement_score"]) else "n/a", ""),
                ("Avg Speech", f"{row['average_speech_score']:.2f}" if pd.notna(row["average_speech_score"]) else "n/a", ""),
                ("Hawkish Meeting", row["most_hawkish_meeting"], ""),
                ("Dovish Meeting", row["most_dovish_meeting"], ""),
            ]
        )

    st.subheader("Year Comparison Heatmap")
    heat_cols = [
        "average_meeting_score",
        "average_statement_score",
        "average_speech_score",
        "average_phrase_matches",
        "average_hawkish_phrases",
        "average_dovish_phrases",
    ]
    heat = yearly.set_index("year")[heat_cols]
    if HAS_PLOTLY:
        fig = px.imshow(heat, color_continuous_scale="RdYlGn", aspect="auto", title="Year Comparison Heatmap")
        fig.update_layout(template="plotly_dark", height=430)
        st.plotly_chart(fig, width="stretch", theme=None)
    else:
        st.dataframe(heat.style.background_gradient(cmap="RdYlGn"), width="stretch")

    st.subheader("Regime Change Table")
    st.dataframe(
        yearly[
            [
                "year",
                "average_meeting_score",
                "change_from_previous_year",
                "change_from_2021_baseline",
                "regime_classification",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Phrase Evolution")
    year_choice = st.selectbox("Phrase year", ["All years", 2021, 2022, 2023, 2024, 2025, 2026])
    phrase_subset = phrases if year_choice == "All years" else phrases[phrases["year"].eq(int(year_choice))]
    phrase_summary = phrase_subset.groupby(["phrase", "direction"], as_index=False).agg(
        occurrences=("count", "sum"),
        average_contribution=("scaled_contribution", "mean"),
        speakers_using_phrase=("speaker", "nunique"),
    )
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(
            phrase_summary[phrase_summary["direction"].eq("hawkish")].sort_values("occurrences", ascending=False).head(10),
            width="stretch",
            hide_index=True,
        )
    with c2:
        st.dataframe(
            phrase_summary[phrase_summary["direction"].eq("dovish")].sort_values("occurrences", ascending=False).head(10),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Speaker Regime Changes")
    key_docs = docs[docs["speaker"].str.contains("|".join(KEY_SPEAKERS), case=False, na=False)]
    speaker_year = key_docs.groupby(["speaker", "year"], as_index=False)["score"].mean()
    if HAS_PLOTLY and not speaker_year.empty:
        fig = px.line(speaker_year, x="year", y="score", color="speaker", markers=True, title="Key Speaker Yearly Score")
        fig.update_layout(template="plotly_dark", height=430)
        st.plotly_chart(fig, width="stretch", theme=None)
        pivot = speaker_year.pivot(index="speaker", columns="year", values="score")
        fig = px.imshow(pivot, color_continuous_scale="RdYlGn", aspect="auto", title="Key Speaker Heatmap")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, width="stretch", theme=None)
    else:
        st.dataframe(speaker_year, width="stretch", hide_index=True)

    st.subheader("Meeting Distribution by Year")
    if HAS_PLOTLY and not meetings.empty:
        fig = px.histogram(meetings, x="meeting_cycle_score", color="year", nbins=18, barmode="overlay")
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, width="stretch", theme=None)
    else:
        st.dataframe(meetings.groupby("year")["meeting_cycle_score"].describe(), width="stretch")

    st.subheader("Regime Comparison Tool")
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    a, b = st.columns(2)
    year_a = a.selectbox("Year A", years, index=0)
    year_b = b.selectbox("Year B", years, index=len(years) - 1)
    comparison = []
    for year in [year_a, year_b]:
        m = meetings[meetings["year"].eq(year)]
        d = docs[docs["year"].eq(year)]
        p = phrases[phrases["year"].eq(year)]
        comparison.append(
            {
                "year": year,
                "average_meeting_score": m["meeting_cycle_score"].mean(),
                "statement_score": m["statement_score"].mean(),
                "speech_score": m["speech_avg_score"].mean(),
                "top_hawkish_phrase": p[p["direction"].eq("hawkish")].groupby("phrase")["count"].sum().sort_values(ascending=False).head(1).index.tolist(),
                "top_dovish_phrase": p[p["direction"].eq("dovish")].groupby("phrase")["count"].sum().sort_values(ascending=False).head(1).index.tolist(),
                "most_hawkish_speaker": d.groupby("speaker")["score"].mean().sort_values(ascending=False).head(1).index.tolist(),
                "most_dovish_speaker": d.groupby("speaker")["score"].mean().sort_values(ascending=True).head(1).index.tolist(),
                "most_hawkish_meeting": m.sort_values("meeting_cycle_score", ascending=False)["meeting_id"].head(1).tolist(),
                "most_dovish_meeting": m.sort_values("meeting_cycle_score", ascending=True)["meeting_id"].head(1).tolist(),
            }
        )
    st.dataframe(pd.DataFrame(comparison), width="stretch", hide_index=True)


def page_methodology(docs: pd.DataFrame, meetings: pd.DataFrame, speakers: pd.DataFrame) -> None:
    st.markdown('<div class="terminal-title">Summary / Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **FED-only scope.** This dashboard reads the FED Phase 1 linkage and Hawkometer outputs under
        `processed/fed/` and `processed/fed/hawkometer/`.

        **Phrase scoring.** Documents are scored by deterministic phrase matching. Hawkish evidence adds
        positive weight, dovish evidence adds negative weight, repeated phrases use sub-linear scaling, and
        document scores are clipped to the -10 to +10 range.

        **Classification thresholds.**
        - +8 to +10: Strongly Hawkish
        - +3 to +8: Hawkish
        - +0.5 to +3: Neutral-Hawkish
        - -0.5 to +0.5: Neutral
        - -3 to -0.5: Neutral-Dovish
        - -8 to -3: Dovish
        - -10 to -8: Strongly Dovish

        **Speaker rolling scores.** Speaker charts use document-level scores with 30-day and 90-day rolling averages.
        Shift alerts compare recent 30-day tone against the prior 60-day window.

        **Meeting weights used for research interpretation.**
        Statement = 45%, Minutes = 25%, Press conference = 20%, Speeches = 10%.

        **Known limitations.** The method is phrase based and can miss context, negation, irony, conditional language,
        hypothetical statements, and speaker intent. The phrase library is deliberately conservative and is not a
        causal prediction engine.
        """
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        csv_download(docs, "Export document table", "fed_document_scores_export.csv")
    with c2:
        csv_download(meetings, "Export meeting table", "fed_meeting_scores_export.csv")
    with c3:
        csv_download(speakers, "Export speaker table", "fed_speaker_scores_export.csv")


def main() -> None:
    inject_style()
    docs, meetings, speakers, summary, phrases = load_data()
    print_startup_summary(docs, meetings, speakers)

    if docs.empty:
        st.warning("No FED document score data loaded. Run FED scoring first.")
        return
    if not docs["bank"].astype(str).str.upper().eq("FED").all():
        st.warning("Non-FED rows detected in the document score file. They have been filtered out.")
        docs = docs[docs["bank"].astype(str).str.upper().eq("FED")]

    dashboard_tab, research_tab = st.tabs(["Global Dashboard", "Research Lab"])
    with dashboard_tab:
        render_four_section_dashboard(docs, meetings)
    with research_tab:
        render_forward_guidance_lab(docs, meetings, phrases)


if __name__ == "__main__":
    main()
