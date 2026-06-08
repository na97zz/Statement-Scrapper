from __future__ import annotations

import argparse
from datetime import datetime
import os

from src import state
from src.scrapers import SCRAPER_CLASSES
from src.utils.files import ensure_data_dirs
from src.utils.logging_utils import setup_logging


def run_scrapers(selected_banks: list[str] | None = None) -> list[dict]:
    logger = setup_logging()
    ensure_data_dirs()
    state.clear_stop()
    selected = selected_banks or [scraper_cls.bank for scraper_cls in SCRAPER_CLASSES]
    state.write_pid(os.getpid())
    state.write_state(
        {
            "status": "starting",
            "phase": "initialising",
            "selected_banks": selected,
            "total_banks": len(selected),
            "bank_position": 0,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "summaries": [],
            "error": "",
        },
        merge=False,
    )
    summaries: list[dict] = []
    try:
        for scraper_cls in SCRAPER_CLASSES:
            if selected_banks and scraper_cls.bank not in selected_banks:
                continue
            if state.stop_requested():
                break
            state.write_state(
                {
                    "status": "running",
                    "phase": "starting bank",
                    "current_bank": scraper_cls.bank,
                    "current_category": "",
                    "current_url": "",
                    "current_title": "",
                    "bank_position": len(summaries) + 1,
                    "category_position": 0,
                    "candidate_position": 0,
                    "candidate_total": 0,
                    "last_message": f"Starting {scraper_cls.bank}",
                }
            )
            scraper = scraper_cls(logger=logger)
            summaries.append(scraper.run())
            state.write_state({"summaries": summaries, "last_message": f"Finished {scraper_cls.bank}"})
        final_status = "stopped" if state.stop_requested() else "finished"
        state.write_state(
            {
                "status": final_status,
                "phase": final_status,
                "current_bank": "",
                "current_category": "",
                "current_url": "",
                "current_title": "",
                "summaries": summaries,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "last_message": f"Scraper runner {final_status}",
            }
        )
        logger.info("Scraper runner %s", final_status)
    except Exception as exc:
        logger.exception("Runner failed: %s", exc)
        state.write_state({"status": "failed", "phase": "failed", "error": str(exc), "summaries": summaries})
    finally:
        state.clear_pid()
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run central bank scraper pipeline.")
    parser.add_argument("--banks", nargs="*", help="Optional bank codes to scrape, e.g. FED ECB RBA")
    args = parser.parse_args()
    run_scrapers(args.banks)


if __name__ == "__main__":
    main()
