from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

from src.config import BANK_CATEGORIES, DATA_DIR, MAX_FILENAME_LENGTH


INVALID_FILENAME_CHARS = r'<>:"/\|?*'


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for bank, categories in BANK_CATEGORIES.items():
        for category in categories:
            (DATA_DIR / bank / category).mkdir(parents=True, exist_ok=True)


def clean_filename(value: str, max_length: int = MAX_FILENAME_LENGTH) -> str:
    value = value or "Untitled"
    value = "".join("_" if c in INVALID_FILENAME_CHARS else c for c in value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    value = re.sub(r"_+", "_", value)
    return (value[:max_length].rstrip(" .") or "Untitled")


def document_stem(date_for_file: str, title: str, speaker: str = "", is_speech: bool = False) -> str:
    title_part = clean_filename(title)
    if is_speech:
        if speaker:
            speaker_part = clean_filename(speaker)
            return clean_filename(f"{date_for_file} - {speaker_part} - {title_part}")
        return clean_filename(f"{date_for_file} - {title_part}")
    return clean_filename(f"{date_for_file} - {title_part}")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clear_data_dir() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    ensure_data_dirs()
