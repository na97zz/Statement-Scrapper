from __future__ import annotations

import csv
import json
from pathlib import Path

from src.config import INDEX_COLUMNS, MASTER_INDEX_CSV, MASTER_INDEX_JSONL
from src.scraper_base import BaseScraper, DocumentCandidate
from src.utils.dates import au_date
from src.utils.files import sha256_file, write_json


def document_rank(metadata: dict) -> tuple[int, str]:
    document_type = str(metadata.get("document_type", "")).lower()
    source_url = str(metadata.get("source_url", "")).lower()
    pdf_score = 0 if document_type == "pdf" or source_url.endswith(".pdf") else 1
    return pdf_score, str(metadata.get("scraped_at", ""))


def read_metadata(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not payload.get("bank") or not payload.get("category") or not payload.get("source_url"):
        return None
    return payload


def rewrite_text_payload(text_path: Path, metadata: dict, scraper: BaseScraper) -> None:
    if not text_path.exists():
        return
    text = scraper.repair_text_encoding(text_path.read_text(encoding="utf-8", errors="replace"))
    lines = text.splitlines()
    replacements = {
        "BANK:": f"BANK: {metadata.get('bank', '')}",
        "CATEGORY:": f"CATEGORY: {metadata.get('category', '')}",
        "TITLE:": f"TITLE: {metadata.get('title', '')}",
        "DATE:": f"DATE: {metadata.get('date_au', '')}",
        "SPEAKER:": f"SPEAKER: {metadata.get('speaker', '')}",
        "SPEAKER ROLE:": f"SPEAKER ROLE: {metadata.get('speaker_role', '')}",
        "SOURCE URL:": f"SOURCE URL: {metadata.get('source_url', '')}",
        "DOCUMENT TYPE:": f"DOCUMENT TYPE: {metadata.get('document_type', '')}",
    }
    seen = set()
    for index, line in enumerate(lines):
        for prefix, replacement in replacements.items():
            if line.startswith(prefix):
                lines[index] = replacement
                seen.add(prefix)
                break
        if line.startswith("==== EXTRACTED TEXT START ===="):
            break
    if "==== EXTRACTED TEXT START ====" not in text:
        body = "\n".join(lines)
        lines = [replacement for prefix, replacement in replacements.items()]
        lines.extend(["", "==== EXTRACTED TEXT START ====", "", body])
    text_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def move_if_needed(old_path: Path, new_path: Path) -> Path:
    if old_path == new_path:
        return old_path
    if not old_path.exists():
        return new_path if new_path.exists() else old_path
    new_path.parent.mkdir(parents=True, exist_ok=True)
    if new_path.exists():
        old_path.unlink()
        return new_path
    old_path.rename(new_path)
    return new_path


def cleanup() -> None:
    scraper = BaseScraper()
    metadata_paths = [path for path in Path("data").glob("*/*/*.json")]
    grouped: dict[tuple[str, str, str, str], list[tuple[Path, dict]]] = {}

    for metadata_path in metadata_paths:
        metadata = read_metadata(metadata_path)
        if not metadata or metadata.get("status") not in {"success", "skipped"}:
            continue
        source_url = str(metadata.get("source_url", ""))
        if "snb.ch" in source_url and "/news-publications/speeches~page-" in source_url:
            for field in ("raw_file_path", "text_file_path"):
                path_value = metadata.get(field)
                if path_value:
                    path = Path(path_value)
                    if path.exists():
                        path.unlink()
            metadata_path.unlink()
            continue
        metadata["title"] = scraper.normalize_document_title(
            str(metadata.get("title", "")),
            str(metadata.get("bank", "")),
            str(metadata.get("category", "")),
        )
        url_date = scraper.extract_date_text(source_url)
        if not __import__("re").fullmatch(r"\d{4}-\d{2}-\d{2}", url_date or ""):
            url_date = ""
        if url_date:
            metadata["date_original"] = url_date
            metadata["date_au"] = au_date(url_date)
        if metadata.get("category") == "Speeches":
            text_path = Path(str(metadata.get("text_file_path", "")))
            extracted_text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""
            current_speaker = str(metadata.get("speaker", "")).strip()
            title_lower = str(metadata.get("title", "")).lower()
            suspicious_speaker = (
                current_speaker.lower() in {"", "nan", "unknown speaker"}
                or (len(current_speaker.split()) > 3 and current_speaker.lower() in title_lower)
            )
            if suspicious_speaker:
                speaker = scraper.infer_speaker(
                    str(metadata.get("title", "")),
                    extracted_text,
                    str(metadata.get("bank", "")),
                    str(metadata.get("source_url", "")),
                )
                metadata["speaker"] = speaker or ""
        candidate = DocumentCandidate(
            bank=str(metadata.get("bank", "")),
            category=str(metadata.get("category", "")),
            title=str(metadata.get("title", "")),
            date_original=str(metadata.get("date_original") or metadata.get("date_au") or ""),
            source_url=str(metadata.get("source_url", "")),
            speaker=str(metadata.get("speaker", "")),
            speaker_role=str(metadata.get("speaker_role", "")),
        )
        key = scraper.logical_document_key(candidate)
        grouped.setdefault(key, []).append((metadata_path, metadata))

    kept_metadata: list[dict] = []
    removed = 0
    renamed = 0

    for rows in grouped.values():
        rows.sort(key=lambda item: document_rank(item[1]))
        keep_path, metadata = rows[0]
        for drop_path, drop_metadata in rows[1:]:
            for field in ("raw_file_path", "text_file_path"):
                path_value = drop_metadata.get(field)
                if path_value:
                    path = Path(path_value)
                    if path.exists():
                        path.unlink()
            if drop_path.exists():
                drop_path.unlink()
            removed += 1

        candidate = DocumentCandidate(
            bank=str(metadata.get("bank", "")),
            category=str(metadata.get("category", "")),
            title=str(metadata.get("title", "")),
            date_original=str(metadata.get("date_original") or metadata.get("date_au") or ""),
            source_url=str(metadata.get("source_url", "")),
            speaker=str(metadata.get("speaker", "")),
            speaker_role=str(metadata.get("speaker_role", "")),
        )
        document_type = str(metadata.get("document_type", "html"))
        target_paths = scraper.paths_for(candidate, document_type)
        raw_path = Path(str(metadata.get("raw_file_path", "")))
        text_path = Path(str(metadata.get("text_file_path", "")))

        new_raw = move_if_needed(raw_path, target_paths["raw"])
        new_text = move_if_needed(text_path, target_paths["text"])
        new_metadata_path = move_if_needed(keep_path, target_paths["metadata"])
        if new_metadata_path != keep_path or new_raw != raw_path or new_text != text_path:
            renamed += 1

        metadata["raw_file_path"] = str(new_raw)
        metadata["text_file_path"] = str(new_text)
        metadata["date_au"] = au_date(candidate.date_original)
        metadata["status"] = "success"
        if new_raw.exists():
            metadata["content_sha256"] = sha256_file(new_raw)
        rewrite_text_payload(new_text, metadata, scraper)
        write_json(new_metadata_path, metadata)
        kept_metadata.append(metadata)

    kept_metadata.sort(key=lambda row: (row.get("bank", ""), row.get("category", ""), row.get("date_au", ""), row.get("title", "")))
    MASTER_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MASTER_INDEX_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for metadata in kept_metadata:
            writer.writerow({column: metadata.get(column, "") for column in INDEX_COLUMNS})
    with MASTER_INDEX_JSONL.open("w", encoding="utf-8") as handle:
        for metadata in kept_metadata:
            row = {column: metadata.get(column, "") for column in INDEX_COLUMNS}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Kept {len(kept_metadata)} logical documents")
    print(f"Removed {removed} duplicate metadata groups")
    print(f"Renamed or refreshed {renamed} documents")


if __name__ == "__main__":
    cleanup()
