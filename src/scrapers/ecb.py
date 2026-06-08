from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src import state
from src.config import START_YEAR, ScrapeTarget
from src.scraper_base import BaseScraper, DocumentCandidate
from src.utils.dates import is_in_scope


class ECBScraper(BaseScraper):
    bank = "ECB"
    targets = [
        ScrapeTarget(
            bank=bank,
            category="Monetary Policy",
            url="https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html",
            include_keywords=("monetary policy decisions",),
            include_url_patterns=(r"/press/pr/date/20\d{2}/html/ecb\.mp\d+~.*\.en\.html$",),
        ),
        ScrapeTarget(
            bank=bank,
            category="Press Conference Q&A",
            url="https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/html/index.en.html",
            include_keywords=("monetary policy statement", "q&a"),
            include_url_patterns=(r"/press/press_conference/monetary-policy-statement/20\d{2}/html/ecb\.is\d+~.*\.en\.html$",),
        ),
        ScrapeTarget(
            bank=bank,
            category="Speeches",
            url="https://www.ecb.europa.eu/press/pubbydate/html/index.en.html?name_of_publication=Speech",
            include_keywords=(),
            include_url_patterns=(r"/press/key/date/20\d{2}/html/ecb\.sp\d+~.*\.en\.html$",),
        ),
    ]

    def discover(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        if target.category in {"Monetary Policy", "Press Conference Q&A"}:
            yield from self.discover_year_snippets(target)
            return
        if target.category == "Speeches":
            yield from self.discover_speeches_from_addsearch(target)
            return
        yield from super().discover(target)

    def discover_year_snippets(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        soup = self.get_soup(target.url)
        if not soup:
            return
        container = soup.find(id="lazyload-container")
        snippets = (container.get("data-snippets") if container else "") or ""
        seen: set[str] = set()
        for snippet in [part.strip() for part in snippets.split(",") if part.strip()]:
            year_match = re.search(r"/(20\d{2})/", snippet)
            if not year_match:
                continue
            year = int(year_match.group(1))
            if year < START_YEAR or year > datetime.now().year:
                continue
            snippet_url = urljoin(target.url, snippet)
            state.write_state(
                {
                    "phase": "reading ECB snippet",
                    "current_url": snippet_url,
                    "last_message": f"Reading ECB {target.category} snippet for {year}",
                }
            )
            snippet_soup = self.get_soup(snippet_url)
            if not snippet_soup:
                continue
            current_date = ""
            for node in snippet_soup.find_all(["dt", "dd"]):
                if node.name == "dt":
                    current_date = node.get("isoDate") or node.get_text(" ", strip=True)
                    continue
                for link in node.find_all("a", href=True):
                    title = self.clean_text(link.get_text(" ", strip=True))
                    href = urljoin(target.url, link["href"])
                    if href in seen or not self.link_matches(href, title, target):
                        continue
                    if current_date and not is_in_scope(current_date):
                        continue
                    seen.add(href)
                    text = self.clean_text(node.get_text(" ", strip=True))
                    speaker, role = self.extract_ecb_speaker_details(text, title)
                    yield DocumentCandidate(
                        bank=target.bank,
                        category=target.category,
                        title=title,
                        date_original=current_date,
                        source_url=href,
                        speaker=speaker,
                        speaker_role=role,
                    )
                    break

    def discover_speeches_from_addsearch(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        api_url = "https://api.addsearch.com/v1/search/61893af990d2673c4a92b492dd7f6631"
        seen: set[str] = set()
        for page in range(1, 80):
            state.write_state(
                {
                    "phase": "reading ECB speech API",
                    "current_url": target.url,
                    "last_message": f"Reading ECB speech search page {page}",
                }
            )
            response = self.session.get(
                api_url,
                params={"term": "Speech", "page": page, "limit": 50, "sort": "date", "order": "desc"},
                timeout=30,
            )
            response.raise_for_status()
            hits = response.json().get("hits", [])
            if not hits:
                break
            older_than_scope = False
            for hit in hits:
                href = (hit.get("url") or "").replace("https://www.ecb.europa.eu//", "https://www.ecb.europa.eu/")
                title = self.clean_text(hit.get("title") or "")
                date_text = hit.get("ts") or ""
                if date_text and not is_in_scope(date_text):
                    if re.search(r"20\d{2}", date_text) and int(date_text[:4]) < START_YEAR:
                        older_than_scope = True
                    continue
                if href in seen or not self.link_matches(href, title, target):
                    continue
                seen.add(href)
                yield DocumentCandidate(
                    bank=target.bank,
                    category=target.category,
                    title=title,
                    date_original=date_text,
                    source_url=href,
                    speaker=self.extract_speaker_from_ecb_url_or_title(href, title),
                )
            if older_than_scope:
                break

    def extract_ecb_speaker_details(self, text: str, title: str) -> tuple[str, str]:
        if ":" in title:
            speaker = title.split(":", 1)[0].strip()
        else:
            speaker = ""
        role = ""
        if speaker:
            remainder = text.replace(title, "", 1).strip()
            role = re.sub(r"\s+", " ", remainder)
        return speaker, role

    def extract_speaker_from_ecb_url_or_title(self, url: str, title: str) -> str:
        if ":" in title:
            return title.split(":", 1)[0].strip()
        match = re.search(r"ecb\.sp\d+~", url)
        return "" if match else ""
