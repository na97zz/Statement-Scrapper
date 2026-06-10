from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
CONTROL_FILE = DATA_DIR / "control.json"
PID_FILE = DATA_DIR / "scraper.pid"
STATE_FILE = DATA_DIR / "scraper_state.json"
MASTER_INDEX_CSV = DATA_DIR / "master_index.csv"
MASTER_INDEX_JSONL = DATA_DIR / "master_index.jsonl"
FAILED_SCRAPES_CSV = DATA_DIR / "failed_scrapes.csv"

START_YEAR = 2000
CURRENT_YEAR = datetime.now().year
YEARS = list(range(CURRENT_YEAR, START_YEAR - 1, -1))

USER_AGENT = (
    "central-bank-policy-research-scraper/1.0 "
    "(polite academic use; contact: local-runner)"
)
REQUEST_TIMEOUT_SECONDS = 30
RATE_LIMIT_SECONDS = 1.5
MAX_FILENAME_LENGTH = 150

INDEX_COLUMNS = [
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

FAILED_COLUMNS = ["bank", "category", "url", "error_message", "timestamp"]


@dataclass(frozen=True)
class ScrapeTarget:
    bank: str
    category: str
    url: str
    include_keywords: tuple[str, ...] = ()
    exclude_keywords: tuple[str, ...] = ()
    include_url_patterns: tuple[str, ...] = ()
    exclude_url_patterns: tuple[str, ...] = ()
    years: tuple[int, ...] = tuple(YEARS)
    prefer_pdf: bool = False
    max_pages: int = 8


BANK_CATEGORIES = {
    "FED": ["Monetary Policy", "Minutes", "Speeches"],
    "ECB": ["Monetary Policy", "Press Conference Q&A", "Speeches"],
    "RBA": ["Monetary Policy", "Minutes", "Speeches"],
    "BOJ": ["Monetary Policy", "Speeches"],
    "SNB": ["Monetary Policy", "Speeches"],
    "RBNZ": ["Monetary Policy", "Minutes", "Speeches"],
    "BOE": ["Monetary Policy", "Speeches"],
    "BOC": ["Monetary Policy", "Speeches"],
    "RIKSBANK": ["Monetary Policy", "Speeches"],
    "NORGES_BANK": ["Monetary Policy", "Speeches"],
}
