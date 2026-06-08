from __future__ import annotations

import re
from typing import Iterable

from bs4 import BeautifulSoup

from src.config import START_YEAR, ScrapeTarget
from src.scraper_base import BaseScraper, DocumentCandidate
from src.utils.dates import is_in_scope


class BOCScraper(BaseScraper):
    bank = "BOC"
    sitemap_index_url = "https://www.bankofcanada.ca/wp-sitemap.xml"
    targets = [
        ScrapeTarget(
            bank=bank,
            category="Monetary Policy",
            url="https://www.bankofcanada.ca/publications/mpr/",
            include_url_patterns=(
                r"/publications/mpr/mpr-20\d{2}",
                r"/20\d{2}/\d{2}/bank-(of-)?canada-interest-rate-announcement",
                r"/20\d{2}/\d{2}/interest-rate-announcement",
            ),
        ),
        ScrapeTarget(
            bank=bank,
            category="Speeches",
            url="https://www.bankofcanada.ca/press/speeches/",
            include_url_patterns=(r"/20\d{2}/\d{2}/speech-", r"/20\d{2}/\d{2}/speech-by-"),
            exclude_url_patterns=(r"/wp-content/", r"\.pdf$"),
        ),
    ]

    def discover(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        urls = self.sitemap_urls()
        if target.category == "Monetary Policy":
            yield from self.discover_monetary_policy(target, urls)
            return
        if target.category == "Speeches":
            yield from self.discover_speeches(target, urls)
            return
        yield from super().discover(target)

    def sitemap_urls(self) -> list[str]:
        response = self.fetch(self.sitemap_index_url)
        sitemap_urls = re.findall(r"<loc>(https://www\.bankofcanada\.ca/wp-sitemap-posts-post-\d+\.xml)</loc>", response.text)
        urls: list[str] = []
        for sitemap_url in sitemap_urls:
            page = self.fetch(sitemap_url)
            urls.extend(re.findall(r"<loc>(.*?)</loc>", page.text))
        return urls

    def discover_monetary_policy(self, target: ScrapeTarget, urls: list[str]) -> Iterable[DocumentCandidate]:
        patterns = (
            re.compile(r"https://www\.bankofcanada\.ca/publications/mpr/mpr-(20\d{2})(?:-\d{2})?(?:-\d{2})?/?$"),
            re.compile(r"https://www\.bankofcanada\.ca/(20\d{2})/\d{2}/(?:bank-(?:of-)?canada-)?interest-rate-announcement.*?/?$"),
        )
        yielded: set[str] = set()
        for page_url in self.sort_urls(urls):
            if page_url in yielded:
                continue
            match = next((pattern.match(page_url) for pattern in patterns if pattern.match(page_url)), None)
            if not match or int(match.group(1)) < START_YEAR:
                continue
            soup = self.get_soup(page_url)
            if not soup:
                continue
            title = self.extract_page_title(soup) or self.title_from_url(page_url)
            if self.is_noise_title(title):
                continue
            date_text = self.best_date(page_url, soup)
            if not is_in_scope(date_text):
                continue
            yielded.add(page_url)
            yield DocumentCandidate(
                bank=target.bank,
                category=target.category,
                title=title,
                date_original=date_text,
                source_url=page_url,
            )

    def discover_speeches(self, target: ScrapeTarget, urls: list[str]) -> Iterable[DocumentCandidate]:
        pattern = re.compile(r"https://www\.bankofcanada\.ca/(20\d{2})/\d{2}/speech(?:-by)?-[^/]+/?$")
        yielded_events: set[tuple[str, str]] = set()
        for page_url in self.sort_urls(urls):
            match = pattern.match(page_url)
            if not match or int(match.group(1)) < START_YEAR:
                continue
            soup = self.get_soup(page_url)
            if not soup:
                continue
            title = self.extract_page_title(soup) or self.title_from_url(page_url)
            if self.is_noise_title(title):
                continue
            if title.lower().startswith("speech:"):
                continue
            date_text = self.best_date(page_url, soup)
            if not is_in_scope(date_text):
                continue
            speaker = self.extract_boc_speaker(title)
            event_key = (date_text, speaker.lower() or title.lower())
            if event_key in yielded_events:
                continue
            yielded_events.add(event_key)
            text = soup.get_text(" ", strip=True)
            yield DocumentCandidate(
                bank=target.bank,
                category=target.category,
                title=title,
                date_original=date_text,
                source_url=page_url,
                speaker=speaker or self.extract_speaker(text, title, target.category),
            )

    def extract_page_title(self, soup: BeautifulSoup) -> str:
        h1 = soup.find("h1")
        if h1:
            return self.clean_text(h1.get_text(" ", strip=True))
        title = soup.find("title")
        if title:
            return self.clean_text(title.get_text(" ", strip=True).split("- Bank of Canada")[0])
        return ""

    def extract_page_date(self, soup: BeautifulSoup) -> str:
        for attrs in (
            {"property": "article:published_time"},
            {"name": "dcterms.date"},
            {"name": "DC.Date"},
            {"name": "date"},
        ):
            tag = soup.find("meta", attrs=attrs)
            value = tag.get("content", "") if tag else ""
            date_text = self.extract_date_text(value)
            if date_text:
                return date_text
        text = soup.get_text(" ", strip=True)
        return self.extract_date_text(text[:3000])

    def best_date(self, page_url: str, soup: BeautifulSoup) -> str:
        url_date = self.extract_boc_url_date(page_url) or self.extract_date_text(page_url)
        page_date = self.extract_page_date(soup)
        if re.fullmatch(r"20\d{2}-\d{2}", url_date or "") and re.fullmatch(r"20\d{2}", page_date or ""):
            return url_date
        return url_date if self.is_more_specific_date(url_date, page_date) else page_date or url_date

    def extract_boc_url_date(self, url: str) -> str:
        match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", url)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        match = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)-(\d{1,2})-(20\d{2})\b",
            url,
            re.I,
        )
        if match:
            month = {
                "january": "01",
                "february": "02",
                "march": "03",
                "april": "04",
                "may": "05",
                "june": "06",
                "july": "07",
                "august": "08",
                "september": "09",
                "october": "10",
                "november": "11",
                "december": "12",
            }[match.group(1).lower()]
            return f"{match.group(3)}-{month}-{int(match.group(2)):02d}"
        match = re.search(r"/publications/mpr/mpr-(20\d{2})-(\d{2})(?:/|$)", url)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        return ""

    def extract_boc_speaker(self, title: str) -> str:
        match = re.search(r"Speech by\s+(.+?)(?:,\s+|$)", title, re.I)
        if match:
            return self.clean_text(match.group(1))
        match = re.search(r"Speech:\s+(.+?)(?:,\s+|$)", title, re.I)
        return self.clean_text(match.group(1)) if match else ""

    def is_noise_title(self, title: str) -> bool:
        lowered = title.lower()
        return any(
            token in lowered
            for token in (
                "schedule",
                "webcast",
                "media interview",
                "parliamentary appearance",
                "appearance",
                "fireside chat",
            )
        )

    def sort_urls(self, urls: list[str]) -> list[str]:
        def key(url: str) -> tuple[str, str]:
            return self.extract_date_text(url), url

        return sorted(urls, key=key, reverse=True)
