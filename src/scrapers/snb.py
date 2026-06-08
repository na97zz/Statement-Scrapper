from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.config import START_YEAR, ScrapeTarget
from src.scraper_base import BaseScraper, DocumentCandidate
from src.utils.dates import is_in_scope


class SNBScraper(BaseScraper):
    bank = "SNB"
    decisions_url = "https://www.snb.ch/en/the-snb/mandates-goals/monetary-policy/decisions"
    speeches_url = "https://www.snb.ch/en/news-publications/speeches"
    targets = [
        ScrapeTarget(
            bank=bank,
            category="Monetary Policy",
            url=decisions_url,
            include_url_patterns=(r"/publications/communication/press-releases(?:-restricted)?/(?:20\d{2}/)?pre_20\d{6}",),
        ),
        ScrapeTarget(
            bank=bank,
            category="Speeches",
            url=speeches_url,
            include_url_patterns=(r"/publications/communication/speeches(?:-restricted)?/(?:20\d{2}/)?ref_20\d{6}",),
        ),
    ]

    def discover(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        if target.category == "Monetary Policy":
            yield from self.discover_monetary_policy(target)
            return
        if target.category == "Speeches":
            yield from self.discover_speeches(target)
            return
        yield from super().discover(target)

    def discover_monetary_policy(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        soup = self.get_soup(self.decisions_url)
        if not soup:
            return
        seen: set[str] = set()
        for link in soup.find_all("a", href=True):
            url = urljoin(self.decisions_url, link["href"])
            if url in seen or not re.search(r"/press-releases(?:-restricted)?/(?:20\d{2}/)?pre_20\d{6}", url):
                continue
            date_text = self.extract_date_text(url)
            if not is_in_scope(date_text):
                continue
            seen.add(url)
            title = self.clean_text(link.get_text(" ", strip=True)) or f"Monetary policy assessment {date_text}"
            yield DocumentCandidate(
                bank=target.bank,
                category=target.category,
                title=title,
                date_original=date_text,
                source_url=url,
            )

    def discover_speeches(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        seen: set[str] = set()
        for page_url in self.speech_page_urls():
            soup = self.get_soup(page_url)
            if not soup:
                continue
            page_candidates = 0
            older_than_scope = 0
            for link in soup.find_all("a", href=True):
                url = urljoin(page_url, link["href"])
                if url in seen or not re.search(r"/speeches(?:-restricted)?/(?:20\d{2}/)?ref_20\d{6}", url):
                    continue
                date_text = self.extract_date_text(url)
                if not is_in_scope(date_text):
                    older_than_scope += 1
                    continue
                seen.add(url)
                page_candidates += 1
                title = self.clean_speech_title(link.get_text(" ", strip=True))
                yield DocumentCandidate(
                    bank=target.bank,
                    category=target.category,
                    title=title or self.title_from_url(url),
                    date_original=date_text,
                    source_url=url,
                    speaker=self.extract_speaker_from_title(title),
                )
            if page_candidates == 0 and older_than_scope > 0:
                break

    def speech_page_urls(self) -> Iterable[str]:
        yield self.speeches_url
        for page in range(2, 40):
            yield f"{self.speeches_url}~page-0={page}~"

    def clean_speech_title(self, title: str) -> str:
        title = self.clean_text(title)
        title = re.sub(r"^\d{1,2}\.\d{1,2}\.20\d{2}\s+", "", title)
        return title

    def extract_speaker_from_title(self, title: str) -> str:
        match = re.match(r"(.+?),\s+(?:Chairman|Vice Chairman|Member|Alternate Member|President)\b", title)
        if match:
            return self.clean_text(match.group(1))
        return ""
