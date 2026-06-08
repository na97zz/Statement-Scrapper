from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pandas as pd
import streamlit as st

from src import state
from src.config import BANK_CATEGORIES, DATA_DIR, LOG_DIR, MASTER_INDEX_CSV, PID_FILE, PROJECT_ROOT
from src.utils.files import clear_data_dir, ensure_data_dirs


st.set_page_config(page_title="Central Bank Scraper", layout="wide")
ensure_data_dirs()


def is_process_running(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, check=False)
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_scraper(selected_banks: list[str]) -> None:
    if is_process_running(state.read_pid()):
        st.warning("Scraper is already running.")
        return
    state.clear_stop()
    command = [sys.executable, "-m", "src.runner"]
    if selected_banks:
        command.extend(["--banks", *selected_banks])
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    state.write_pid(process.pid)
    state.write_state(
        {
            "status": "starting",
            "phase": "launching background process",
            "current_bank": "",
            "current_category": "",
            "current_title": "",
            "current_url": "",
            "last_message": "Launching scraper process",
            "selected_banks": selected_banks,
        },
        merge=False,
    )


def load_index() -> pd.DataFrame:
    if MASTER_INDEX_CSV.exists():
        return pd.read_csv(MASTER_INDEX_CSV)
    return pd.DataFrame(
        columns=[
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
        ]
    )


def logs_tail(lines: int = 200) -> str:
    log_file = LOG_DIR / "scraper.log"
    if not log_file.exists():
        return "No logs yet."
    content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


st.title("Central Bank Monetary Policy Scraper")

runtime_state = state.read_state()
pid = state.read_pid()
running = is_process_running(pid)
if not running and pid and PID_FILE.exists():
    state.clear_pid()

if running:
    st.markdown("<meta http-equiv='refresh' content='2'>", unsafe_allow_html=True)

left, right = st.columns([1, 2])

with left:
    st.subheader("Controls")
    selected = st.multiselect("Banks", options=list(BANK_CATEGORIES.keys()), default=list(BANK_CATEGORIES.keys()))
    start_col, stop_col = st.columns(2)
    with start_col:
        if st.button("Start scraping", type="primary", disabled=running):
            start_scraper(selected)
            st.rerun()
    with stop_col:
        if st.button("Stop scraping", disabled=not running):
            state.request_stop()
            st.info("Stop requested. The current download will finish, then the runner will stop.")

    if st.button("Clear all data", disabled=running):
        clear_data_dir()
        if LOG_DIR.exists():
            shutil.rmtree(LOG_DIR)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        st.success("Cleared data and logs.")
        st.rerun()

    if st.button("Refresh status"):
        st.rerun()

    status_value = "running" if running else runtime_state.get("status", "idle")
    st.metric("Runner status", status_value)
    st.caption("The page auto-refreshes every 2 seconds while the scraper process is running.")

    status_rows = {
        "PID": pid or "none",
        "Phase": runtime_state.get("phase", "idle"),
        "Current bank": runtime_state.get("current_bank", "") or "none",
        "Current category": runtime_state.get("current_category", "") or "none",
        "Current document": runtime_state.get("current_title", "") or "none",
        "Current URL": runtime_state.get("current_url", "") or "none",
        "Last update": runtime_state.get("updated_at", "") or "none",
        "Last result": runtime_state.get("last_result", "") or "none",
    }
    st.table(pd.DataFrame(status_rows.items(), columns=["Field", "Value"]))

    candidate_position = int(runtime_state.get("candidate_position") or 0)
    candidate_total = int(runtime_state.get("candidate_total") or 0)
    category_position = int(runtime_state.get("category_position") or 0)
    category_total = int(runtime_state.get("category_total") or 0)
    bank_position = int(runtime_state.get("bank_position") or 0)
    bank_total = int(runtime_state.get("total_banks") or len(runtime_state.get("selected_banks", [])) or len(BANK_CATEGORIES))

    st.progress(
        candidate_position / candidate_total if candidate_total else 0,
        text=f"Documents: {candidate_position}/{candidate_total}" if candidate_total else "Documents: discovering",
    )
    st.progress(
        category_position / category_total if category_total else 0,
        text=f"Categories: {category_position}/{category_total}" if category_total else "Categories: waiting",
    )
    st.progress(
        bank_position / bank_total if bank_total else 0,
        text=f"Banks: {bank_position}/{bank_total}",
    )

    count_cols = st.columns(3)
    count_cols[0].metric("Saved this bank", int(runtime_state.get("bank_saved") or 0))
    count_cols[1].metric("Skipped this bank", int(runtime_state.get("bank_skipped") or 0))
    count_cols[2].metric("Failed this bank", int(runtime_state.get("bank_failed") or 0))

    st.info(runtime_state.get("last_message", "No scraper activity yet."))

with right:
    st.subheader("Progress by central bank")
    index_df = load_index()
    if index_df.empty:
        progress_df = pd.DataFrame({"bank": list(BANK_CATEGORIES), "saved_files": 0})
    else:
        progress_df = (
            index_df[index_df["status"].eq("success")]
            .groupby("bank", as_index=False)
            .size()
            .rename(columns={"size": "saved_files"})
        )
        progress_df = pd.DataFrame({"bank": list(BANK_CATEGORIES)}).merge(progress_df, on="bank", how="left").fillna(0)
    st.dataframe(progress_df, use_container_width=True, hide_index=True)

st.subheader("Summary table of scraped files")
st.dataframe(index_df.sort_values("scraped_at", ascending=False) if not index_df.empty else index_df, use_container_width=True)

st.subheader("Logs")
st.code(logs_tail(), language="text")

st.caption(f"Files are saved under {DATA_DIR}")
