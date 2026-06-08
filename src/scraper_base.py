from __future__ import annotations

import csv
import logging
import mimetypes
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src import state
from src.config import (
    FAILED_COLUMNS,
    FAILED_SCRAPES_CSV,
    INDEX_COLUMNS,
    MASTER_INDEX_CSV,
    MASTER_INDEX_JSONL,
    RATE_LIMIT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    ScrapeTarget,
    USER_AGENT,
)
from src.utils.dates import au_date, filename_date, is_in_scope, parse_date
from src.utils.files import (
    append_jsonl,
    clean_filename,
    document_stem,
    ensure_data_dirs,
    sha256_bytes,
    sha256_file,
    write_json,
)
from src.utils.pdf import extract_pdf_text


@dataclass
class DocumentCandidate:
    bank: str
    category: str
    title: str
    date_original: str
    source_url: str
    speaker: str = ""
    speaker_role: str = ""
    datetime_original: str = ""
    notes: str = ""


class BaseScraper:
    bank: str = ""
    targets: list[ScrapeTarget] = []

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("central_bank_scraper")
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1.2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._robots: dict[str, RobotFileParser] = {}
        self._seen_urls = self._load_seen_urls()
        self._seen_hashes = self._load_seen_hashes()
        self._seen_logical_keys = self._load_seen_logical_keys()

    def run(self) -> dict:
        ensure_data_dirs()
        summary = {"bank": self.bank, "saved": 0, "skipped": 0, "failed": 0}
        self.logger.info("Starting %s", self.bank)
        for category_index, target in enumerate(self.targets, start=1):
            if state.stop_requested():
                self.logger.info("Stop requested before %s %s", self.bank, target.category)
                break
            state.write_state(
                {
                    "current_bank": self.bank,
                    "current_category": target.category,
                    "current_url": target.url,
                    "current_title": "",
                    "status": "running",
                    "phase": "discovering links",
                    "category_position": category_index,
                    "category_total": len(self.targets),
                    "candidate_position": 0,
                    "candidate_total": 0,
                    "bank_saved": summary["saved"],
                    "bank_skipped": summary["skipped"],
                    "bank_failed": summary["failed"],
                    "last_message": f"Discovering {self.bank} {target.category}",
                }
            )
            try:
                candidates = self.dedupe_candidates(list(self.discover(target)))
                state.write_state(
                    {
                        "phase": "processing documents",
                        "candidate_total": len(candidates),
                        "candidate_position": 0,
                        "last_message": f"{self.bank} {target.category}: discovered {len(candidates)} candidates",
                    }
                )
                self.logger.info("%s %s discovered %s candidates", self.bank, target.category, len(candidates))
                for candidate_index, candidate in enumerate(candidates, start=1):
                    if state.stop_requested():
                        self.logger.info("Stop requested during %s", self.bank)
                        break
                    state.write_state(
                        {
                            "phase": "downloading/extracting",
                            "candidate_position": candidate_index,
                            "candidate_total": len(candidates),
                            "current_title": candidate.title,
                            "current_url": candidate.source_url,
                            "last_message": f"{self.bank} {target.category}: {candidate_index}/{len(candidates)} {candidate.title}",
                        }
                    )
                    result = self.save_candidate(candidate)
                    summary[result] += 1
                    state.write_state(
                        {
                            "bank_saved": summary["saved"],
                            "bank_skipped": summary["skipped"],
                            "bank_failed": summary["failed"],
                            "last_result": result,
                            "last_message": f"{self.bank} {target.category}: {result} {candidate.title}",
                        }
                    )
            except Exception as exc:
                summary["failed"] += 1
                state.write_state(
                    {
                        "phase": "category failed",
                        "bank_failed": summary["failed"],
                        "last_result": "failed",
                        "last_message": f"{self.bank} {target.category} failed: {exc}",
                    }
                )
                self.log_failure(target.bank, target.category, target.url, exc)
        state.write_state(
            {
                "phase": "bank finished",
                "current_bank": self.bank,
                "current_category": "",
                "current_title": "",
                "current_url": "",
                "bank_saved": summary["saved"],
                "bank_skipped": summary["skipped"],
                "bank_failed": summary["failed"],
                "last_message": f"Finished {self.bank}: {summary['saved']} saved, {summary['skipped']} skipped, {summary['failed']} failed",
            }
        )
        self.logger.info("Finished %s: %s", self.bank, summary)
        return summary

    def discover(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        yield from self.discover_links(target)

    def discover_links(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        pages = self.year_urls(target) if "{year}" in target.url else [target.url]
        visited: set[str] = set()
        yielded: set[str] = set()
        for page_index, page_url in enumerate(pages, start=1):
            state.write_state(
                {
                    "phase": "reading listing page",
                    "current_url": page_url,
                    "listing_page_position": page_index,
                    "listing_page_total": len(pages),
                    "last_message": f"Reading listing page {page_index}/{len(pages)} for {target.bank} {target.category}",
                }
            )
            for listing_url in self.follow_listing_pages(page_url, target.max_pages):
                if listing_url in visited:
                    continue
                visited.add(listing_url)
                state.write_state(
                    {
                        "phase": "parsing listing page",
                        "current_url": listing_url,
                        "discovered_candidates": len(yielded),
                        "last_message": f"Parsing {listing_url}",
                    }
                )
                soup = self.get_soup(listing_url)
                if not soup:
                    continue
                for link in soup.find_all("a", href=True):
                    title = self.clean_text(link.get_text(" ", strip=True))
                    href = urljoin(listing_url, link["href"])
                    if href.rstrip("/") == listing_url.rstrip("/"):
                        continue
                    if not self.link_matches(href, title, target):
                        continue
                    if href in yielded:
                        continue
                    yielded.add(href)
                    surrounding = self.clean_text(link.parent.get_text(" ", strip=True) if link.parent else title)
                    surrounding_date = self.extract_date_text(surrounding)
                    url_date = self.extract_date_text(href)
                    date_text = url_date if self.is_more_specific_date(url_date, surrounding_date) else surrounding_date or url_date
                    if date_text and not is_in_scope(date_text):
                        continue
                    candidate_title = self.normalize_document_title(title or self.title_from_url(href), target.bank, target.category)
                    state.write_state(
                        {
                            "phase": "found candidate",
                            "current_title": candidate_title,
                            "current_url": href,
                            "discovered_candidates": len(yielded),
                            "last_message": f"Found candidate {len(yielded)}: {candidate_title}",
                        }
                    )
                    yield DocumentCandidate(
                        bank=target.bank,
                        category=target.category,
                        title=candidate_title,
                        date_original=date_text,
                        source_url=href,
                        speaker=self.extract_speaker(surrounding, title, target.category),
                    )

    def year_urls(self, target: ScrapeTarget) -> list[str]:
        if "{year}" not in target.url:
            return []
        return [target.url.format(year=year) for year in target.years]

    def dedupe_candidates(self, candidates: list[DocumentCandidate]) -> list[DocumentCandidate]:
        grouped: dict[tuple[str, str, str, str], DocumentCandidate] = {}
        order: list[tuple[str, str, str, str]] = []
        for candidate in candidates:
            candidate.title = self.normalize_document_title(candidate.title, candidate.bank, candidate.category)
            key = self.logical_document_key(candidate)
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = candidate
                order.append(key)
                continue
            if self.candidate_preference(candidate) < self.candidate_preference(existing):
                grouped[key] = candidate
        return [grouped[key] for key in order]

    def candidate_preference(self, candidate: DocumentCandidate) -> tuple[int, str]:
        source_url = candidate.source_url.lower()
        return (0 if source_url.endswith(".pdf") or ".pdf" in source_url else 1, source_url)

    def follow_listing_pages(self, start_url: str, max_pages: int) -> Iterable[str]:
        yield start_url
        if "{year}" in start_url:
            return
        current = start_url
        visited = {current}
        for _ in range(max_pages - 1):
            soup = self.get_soup(current)
            if not soup:
                return
            next_link = soup.find("a", string=re.compile(r"next|older|more", re.I))
            if not next_link or not next_link.get("href"):
                return
            next_url = urljoin(current, next_link["href"])
            if next_url in visited:
                return
            visited.add(next_url)
            current = next_url
            yield current

    def link_matches(self, url: str, title: str, target: ScrapeTarget) -> bool:
        haystack = f"{url} {title}".lower()
        if target.category == "Speeches" and target.bank != "ECB":
            if any(token in haystack for token in ("q&a", "q-and-a", "qanda", "transcript")):
                return False
        if target.prefer_pdf and ".pdf" not in url.lower():
            return False
        if target.include_url_patterns and not any(re.search(pattern, url, re.I) for pattern in target.include_url_patterns):
            return False
        if target.exclude_url_patterns and any(re.search(pattern, url, re.I) for pattern in target.exclude_url_patterns):
            return False
        if target.include_keywords and not any(k.lower() in haystack for k in target.include_keywords):
            return False
        if target.exclude_keywords and any(k.lower() in haystack for k in target.exclude_keywords):
            return False
        if any(bad in haystack for bad in ("javascript:", "mailto:", "#")):
            return False
        return True

    def save_candidate(self, candidate: DocumentCandidate) -> str:
        if candidate.source_url in self._seen_urls:
            state.write_state({"phase": "skipping duplicate url", "last_result": "skipped"})
            self.write_metadata(candidate, "skipped", notes="Duplicate source_url in master index")
            return "skipped"
        if not self.allowed(candidate.source_url):
            state.write_state({"phase": "skipping robots.txt", "last_result": "skipped"})
            self.write_metadata(candidate, "skipped", notes="Blocked by robots.txt")
            return "skipped"
        try:
            state.write_state({"phase": "downloading", "current_url": candidate.source_url})
            response = self.fetch(candidate.source_url)
            content = response.content
            content_hash = sha256_bytes(content)
            if content_hash in self._seen_hashes:
                state.write_state({"phase": "skipping duplicate file", "last_result": "skipped"})
                self.write_metadata(candidate, "skipped", notes="Duplicate content hash")
                return "skipped"
            document_type = self.document_type(candidate.source_url, response)
            candidate.title = self.normalize_document_title(candidate.title, candidate.bank, candidate.category)
            self.ensure_candidate_date(candidate, content, document_type)
            logical_key = self.logical_document_key(candidate)
            if logical_key in self._seen_logical_keys:
                state.write_state({"phase": "skipping duplicate logical document", "last_result": "skipped"})
                self.write_metadata(candidate, "skipped", notes="Duplicate logical document in master index")
                return "skipped"
            paths = self.paths_for(candidate, document_type)
            state.write_state({"phase": f"saving raw {document_type}", "raw_file_path": str(paths["raw"])})
            paths["raw"].parent.mkdir(parents=True, exist_ok=True)
            paths["raw"].write_bytes(content)
            state.write_state({"phase": "extracting text", "text_file_path": str(paths["text"])})
            text = self.extract_text(paths["raw"], content, document_type, candidate.source_url)
            text = self.repair_text_encoding(text)
            if candidate.category == "Speeches":
                self.ensure_candidate_speaker(candidate, text)
                final_paths = self.paths_for(candidate, document_type)
                if final_paths["raw"] != paths["raw"]:
                    final_paths["raw"].parent.mkdir(parents=True, exist_ok=True)
                    if final_paths["raw"].exists():
                        paths["raw"].unlink(missing_ok=True)
                    else:
                        paths["raw"].rename(final_paths["raw"])
                    paths = final_paths
            if candidate.category == "Press Conference Q&A":
                text = self.format_q_and_a(text)
            state.write_state({"phase": "writing text and metadata", "text_file_path": str(paths["text"])})
            paths["text"].write_text(self.text_payload(candidate, document_type, text), encoding="utf-8")
            metadata = self.metadata(candidate, document_type, paths, "success")
            metadata["content_sha256"] = sha256_file(paths["raw"])
            write_json(paths["metadata"], metadata)
            self.append_index(metadata)
            self._seen_urls.add(candidate.source_url)
            self._seen_hashes.add(metadata["content_sha256"])
            self._seen_logical_keys.add(logical_key)
            self.logger.info("Saved %s %s: %s", candidate.bank, candidate.category, candidate.title)
            return "saved"
        except Exception as exc:
            self.log_failure(candidate.bank, candidate.category, candidate.source_url, exc)
            self.write_metadata(candidate, "failed", notes=str(exc))
            return "failed"

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            robot = RobotFileParser()
            robots_url = urljoin(base, "/robots.txt")
            robot.set_url(robots_url)
            try:
                response = self.session.get(robots_url, timeout=REQUEST_TIMEOUT_SECONDS)
                if response.status_code >= 400:
                    self.logger.warning(
                        "Could not verify robots.txt for %s; got HTTP %s, continuing politely",
                        base,
                        response.status_code,
                    )
                    robot.allow_all = True
                else:
                    robot.parse(response.text.splitlines())
            except Exception:
                self.logger.warning("Could not read robots.txt for %s; continuing politely", base)
                robot.allow_all = True
            self._robots[base] = robot
        return self._robots[base].can_fetch(USER_AGENT, url) or self._robots[base].can_fetch("*", url)

    def fetch(self, url: str) -> requests.Response:
        time.sleep(RATE_LIMIT_SECONDS)
        response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response

    def get_soup(self, url: str) -> BeautifulSoup | None:
        if not self.allowed(url):
            self.logger.info("Skipping %s due to robots.txt", url)
            return None
        try:
            response = self.fetch(url)
            return BeautifulSoup(response.text, "html.parser")
        except Exception as exc:
            self.log_failure(self.bank, "Listing", url, exc)
            return None

    def document_type(self, url: str, response: requests.Response) -> str:
        content_type = response.headers.get("content-type", "").lower()
        if ".pdf" in url.lower() or "application/pdf" in content_type:
            return "pdf"
        return "html"

    def paths_for(self, candidate: DocumentCandidate, document_type: str) -> dict[str, Path]:
        date_part = filename_date(candidate.date_original)
        is_speech = candidate.category == "Speeches"
        stem = document_stem(date_part, candidate.title, candidate.speaker, is_speech)
        base = Path("data") / candidate.bank / candidate.category
        extension = ".pdf" if document_type == "pdf" else ".html"
        return {
            "raw": base / f"{stem}{extension}",
            "text": base / f"{stem}.txt",
            "metadata": base / f"{stem}.json",
        }

    def extract_text(self, raw_path: Path, content: bytes, document_type: str, url: str) -> str:
        if document_type == "pdf":
            return extract_pdf_text(raw_path)
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        article = soup.find("main") or soup.find("article") or soup.body or soup
        return self.clean_text(article.get_text("\n", strip=True))

    def text_payload(self, candidate: DocumentCandidate, document_type: str, extracted_text: str) -> str:
        return "\n".join(
            [
                f"BANK: {candidate.bank}",
                f"CATEGORY: {candidate.category}",
                f"TITLE: {candidate.title}",
                f"DATE: {au_date(candidate.date_original)}",
                f"SPEAKER: {candidate.speaker or ''}",
                f"SPEAKER ROLE: {candidate.speaker_role or ''}",
                f"SOURCE URL: {candidate.source_url}",
                f"DOCUMENT TYPE: {document_type}",
                "",
                "==== EXTRACTED TEXT START ====",
                "",
                extracted_text.strip(),
                "",
            ]
        )

    def ensure_candidate_speaker(self, candidate: DocumentCandidate, text: str) -> None:
        if candidate.speaker and candidate.speaker.lower() != "unknown speaker":
            return
        candidate.speaker = self.infer_speaker(candidate.title, text, candidate.bank, candidate.source_url)

    def infer_speaker(self, title: str, text: str, bank: str = "", source_url: str = "") -> str:
        candidates = [
            self.speaker_from_title(title),
            self.speaker_from_lines(text),
        ]
        for candidate in candidates:
            candidate = self.clean_speaker_name(candidate)
            if candidate:
                return candidate
        return ""

    def speaker_from_title(self, title: str) -> str:
        patterns = [
            r"\bStatement by\s+(?:Vice Chair(?: for Supervision)?|Chair|Governor|Deputy Governor|Governor)\s+(.+)$",
            r"\b(?:speech|remarks|opening remarks|fireside chat)\s+by\s+(.+)$",
            r"\b(?:remarks|statement|panel contribution|introductory statement)\s+by\s+(.+?)(?:,|\s+at\s+|$)",
            r"\bby\s+(.+?)(?:\s*[-–]\s*speech|\s*[-–]\s*remarks|$)",
            r"[-–]\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ .'-]+)$",
            r"^(?:Speech|Presentation)\s+([^:]+):",
        ]
        for pattern in patterns:
            match = re.search(pattern, title, re.I)
            if match:
                return match.group(1)
        return ""

    def speaker_from_lines(self, text: str) -> str:
        lines = [self.clean_text(line) for line in text.splitlines()]
        lines = [line for line in lines if line]
        role_prefix = (
            r"(?:Chair|Vice Chair(?: for Supervision)?|Governor|Deputy Governor|Senior Deputy Governor|"
            r"External Deputy Governor|First Deputy Governor|Deputy Chair|"
            r"Chairman of the Governing Board|Vice Chairman of the Governing Board|"
            r"Member of the Governing Board|Executive Director|Chief Economist|"
            r"President|Board Member|Member of the Executive Board)"
        )
        role_suffix = (
            r"(?:Chair|Vice Chair|Governor|Deputy Governor|Senior Deputy Governor|"
            r"External Deputy Governor|First Deputy Governor|Deputy Chair|"
            r"Chairman|Vice Chairman|Member|Executive Director|Chief Economist|"
            r"President|Board Member|Head|Manager)"
        )
        role_line = (
            r"(?:Governor|Deputy Governor|Senior Deputy Governor|External Deputy Governor|"
            r"First Deputy Governor|Assistant Governor|Head of|Manager|Chair|Vice Chair(?: for Supervision)?|"
            r"Member of|Executive Director|Chief Economist|President|Board Member)"
        )
        for index, line in enumerate(lines[:120]):
            match = re.search(rf"\b(?:speech|remarks|address)\s+by\s+(.+?),\s+{role_line}\b", line, re.I)
            if match:
                return match.group(1)
            if len(line) > 180:
                continue
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            next_role_line = lines[index + 2] if next_line in {"[*]", "[ * ]", "*"} and index + 2 < len(lines) else next_line
            if re.search(rf"\b{role_line}\b", next_role_line):
                if re.fullmatch(r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ .'-]{2,80}", line) and len(line.split()) <= 5:
                    return line
            match = re.search(rf"\b{role_prefix}\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ .'-]+)$", line)
            if match:
                return match.group(1)
            match = re.search(rf"^([A-Z][A-Za-zÀ-ÖØ-öø-ÿ .'-]+),\s+{role_suffix}\b", line)
            if match:
                return match.group(1)
            match = re.search(r"\bSpeaker:\s*(.+?)(?:\s+(?:Governor|Deputy|Chair|Member|Place:)|$)", line, re.I)
            if match:
                return match.group(1)
            match = re.search(r"\b(?:speech|remarks|opening remarks)\s+by\s+(.+)$", line, re.I)
            if match:
                return match.group(1)
            match = re.search(r"^By\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ .'-]+)$", line)
            if match:
                return match.group(1)
        return ""

    def clean_speaker_name(self, value: str) -> str:
        value = self.clean_text(value)
        value = re.sub(r"\s*[-–—]\s*(?:speech|remarks).*$", "", value, flags=re.I)
        value = re.sub(r"\s+at\s+.+$", "", value, flags=re.I)
        value = re.sub(r"\s*,?\s*(?:Chair|Vice Chair|Governor|Deputy Governor|Senior Deputy Governor|External Deputy Governor|First Deputy Governor|Chairman|Vice Chairman|Member|Executive Director|Chief Economist|President|Board Member).*$", "", value)
        value = value.strip(" :-–—,")
        bad = {
            "",
            "speech",
            "remarks",
            "pdf",
            "html",
            "unknown speaker",
            "bank of canada",
            "federal reserve",
            "nikkei",
        }
        if value.lower() in bad or len(value) > 80 or len(value.split()) > 6:
            return ""
        if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", value):
            return ""
        return value

    def metadata(self, candidate: DocumentCandidate, document_type: str, paths: dict[str, Path], status: str, notes: str = "") -> dict:
        return {
            "bank": candidate.bank,
            "category": candidate.category,
            "title": candidate.title,
            "date_original": candidate.date_original,
            "date_au": au_date(candidate.date_original),
            "datetime_original": candidate.datetime_original,
            "speaker": candidate.speaker,
            "speaker_role": candidate.speaker_role,
            "source_url": candidate.source_url,
            "raw_file_path": str(paths.get("raw", "")),
            "text_file_path": str(paths.get("text", "")),
            "document_type": document_type,
            "language": "English",
            "scraped_at": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "notes": notes or candidate.notes,
        }

    def write_metadata(self, candidate: DocumentCandidate, status: str, notes: str = "") -> None:
        document_type = mimetypes.guess_type(candidate.source_url)[0] or "html"
        doc_type = "pdf" if "pdf" in document_type or candidate.source_url.lower().endswith(".pdf") else "html"
        paths = self.paths_for(candidate, doc_type)
        write_json(paths["metadata"], self.metadata(candidate, doc_type, paths, status, notes))

    def append_index(self, metadata: dict) -> None:
        row = {column: metadata.get(column, "") for column in INDEX_COLUMNS}
        MASTER_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
        exists = MASTER_INDEX_CSV.exists()
        with MASTER_INDEX_CSV.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        append_jsonl(MASTER_INDEX_JSONL, [row])

    def log_failure(self, bank: str, category: str, url: str, exc: Exception) -> None:
        self.logger.exception("Failed %s %s %s: %s", bank, category, url, exc)
        FAILED_SCRAPES_CSV.parent.mkdir(parents=True, exist_ok=True)
        exists = FAILED_SCRAPES_CSV.exists()
        with FAILED_SCRAPES_CSV.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FAILED_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "bank": bank,
                    "category": category,
                    "url": url,
                    "error_message": str(exc),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )

    def _load_seen_urls(self) -> set[str]:
        if not MASTER_INDEX_CSV.exists():
            return set()
        try:
            return set(pd.read_csv(MASTER_INDEX_CSV)["source_url"].dropna().astype(str))
        except Exception:
            return set()

    def _load_seen_hashes(self) -> set[str]:
        hashes: set[str] = set()
        for path in Path("data").rglob("*.json"):
            try:
                import json

                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("content_sha256"):
                    hashes.add(payload["content_sha256"])
            except Exception:
                continue
        return hashes

    def _load_seen_logical_keys(self) -> set[tuple[str, str, str, str]]:
        if not MASTER_INDEX_CSV.exists():
            return set()
        try:
            df = pd.read_csv(MASTER_INDEX_CSV).fillna("")
            keys: set[tuple[str, str, str, str]] = set()
            for _, row in df.iterrows():
                candidate = DocumentCandidate(
                    bank=str(row.get("bank", "")),
                    category=str(row.get("category", "")),
                    title=self.normalize_document_title(str(row.get("title", "")), str(row.get("bank", "")), str(row.get("category", ""))),
                    date_original=str(row.get("date_au", "")),
                    source_url=str(row.get("source_url", "")),
                )
                keys.add(self.logical_document_key(candidate))
            return keys
        except Exception:
            return set()

    def logical_document_key(self, candidate: DocumentCandidate) -> tuple[str, str, str, str]:
        parsed = parse_date(candidate.date_original)
        date_key = parsed.strftime("%Y-%m-%d") if parsed else self.extract_date_text(candidate.source_url)
        title_key = self.normalize_document_title(candidate.title, candidate.bank, candidate.category).lower()
        title_key = re.sub(r"\s+", " ", title_key).strip()
        return (candidate.bank.upper(), candidate.category.lower(), date_key, title_key)

    def extract_date_text(self, text: str) -> str:
        patterns = [
            r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b",
            r"\b[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match and parse_date(match.group(0)):
                return match.group(0)
        compact_match = re.search(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", text)
        if compact_match:
            compact_date = f"{compact_match.group(1)}-{compact_match.group(2)}-{compact_match.group(3)}"
            if parse_date(compact_date):
                return compact_date
        short_compact_match = re.search(r"(?<!\d)(2[1-9])(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", text)
        if short_compact_match:
            compact_date = f"20{short_compact_match.group(1)}-{short_compact_match.group(2)}-{short_compact_match.group(3)}"
            if parse_date(compact_date):
                return compact_date
        year_match = re.search(r"\b20(2[1-9])\b", text)
        return year_match.group(0) if year_match else ""

    def ensure_candidate_date(self, candidate: DocumentCandidate, content: bytes, document_type: str) -> None:
        if parse_date(candidate.date_original):
            return
        date_text = (
            self.extract_date_text(candidate.date_original)
            or self.extract_date_text(candidate.source_url)
            or self.extract_date_text(candidate.title)
        )
        if not date_text:
            if document_type == "html":
                date_text = self.extract_date_text_from_html(content)
            elif document_type == "pdf":
                date_text = self.extract_date_text_from_pdf(content)
        if date_text and parse_date(date_text):
            candidate.date_original = date_text

    def is_more_specific_date(self, candidate_date: str, current_date: str) -> bool:
        if not candidate_date:
            return False
        if not current_date:
            return True
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate_date) and re.fullmatch(r"20\d{2}", current_date))

    def extract_date_text_from_html(self, content: bytes) -> str:
        soup = BeautifulSoup(content, "html.parser")
        selectors = [
            ("meta", {"property": "article:published_time"}),
            ("meta", {"name": "dcterms.date"}),
            ("meta", {"name": "DC.Date"}),
            ("meta", {"name": "date"}),
        ]
        for name, attrs in selectors:
            tag = soup.find(name, attrs=attrs)
            value = tag.get("content", "") if tag else ""
            date_text = self.extract_date_text(value)
            if date_text:
                return date_text
        text = soup.get_text(" ", strip=True)
        return self.extract_date_text(text[:5000])

    def extract_date_text_from_pdf(self, content: bytes) -> str:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
                handle.write(content)
                temp_path = Path(handle.name)
            text = extract_pdf_text(temp_path)
            return self.extract_date_text(text[:5000])
        except Exception:
            return ""
        finally:
            if temp_path:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def normalize_document_title(self, title: str, bank: str, category: str) -> str:
        clean_title = self.clean_text(title)
        if clean_title.lower() not in {"", "pdf", "html"}:
            return clean_title
        if bank == "FED" and category == "Monetary Policy":
            return "FOMC statement"
        if bank == "FED" and category == "Minutes":
            return "FOMC minutes"
        return f"{bank} {category}"

    def repair_text_encoding(self, text: str) -> str:
        replacements = {
            "â€™": "'",
            "â€˜": "'",
            "â€œ": '"',
            "â€": '"',
            "â€“": "-",
            "â€”": "-",
            "â€¦": "...",
            "Â ": " ",
            "Â": "",
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text

    def extract_speaker(self, surrounding_text: str, title: str, category: str) -> str:
        if category != "Speeches":
            return ""
        text = surrounding_text.replace(title, "")
        match = re.search(r"\b(?:by|speaker:?|remarks by)\s+([^|,;\n]+)", text, re.I)
        if match:
            return clean_filename(match.group(1))
        return ""

    def title_from_url(self, url: str) -> str:
        slug = Path(urlparse(url).path).stem
        return clean_filename(slug.replace("-", " ").replace("_", " ").title())

    def format_q_and_a(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        formatted: list[str] = []
        for line in lines:
            lower = line.lower()
            if lower.startswith(("question:", "q:")):
                formatted.append(f"QUESTION: {line.split(':', 1)[-1].strip()}")
                formatted.append("ASKED BY: Unknown journalist")
            elif lower.startswith(("answer:", "a:")):
                formatted.append("ANSWERED BY: Unknown respondent")
                formatted.append(f"ANSWER: {line.split(':', 1)[-1].strip()}")
            else:
                formatted.append(line)
        return "\n".join(formatted)

    def clean_text(self, value: str) -> str:
        value = (value or "").replace("\xa0", " ")
        return re.sub(r"[ \t]+", " ", value).strip()
