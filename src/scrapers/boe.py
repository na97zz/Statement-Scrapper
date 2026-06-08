from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.config import START_YEAR, ScrapeTarget
from src.scraper_base import BaseScraper, DocumentCandidate
from src.utils.dates import is_in_scope


class BOEScraper(BaseScraper):
    bank = "BOE"
    sitemap_url = "https://www.bankofengland.co.uk/_api/sitemap/getsitemap"
    targets = [
        ScrapeTarget(
            bank=bank,
            category="Monetary Policy",
            url="https://www.bankofengland.co.uk/monetary-policy-report/monetary-policy-report",
            include_url_patterns=(r"/monetary-policy-report/20\d{2}/[a-z]+-20\d{2}$",),
        ),
        ScrapeTarget(
            bank=bank,
            category="Speeches",
            url="https://www.bankofengland.co.uk/news/speeches",
            include_url_patterns=(r"/speech/20\d{2}/.*",),
        ),
    ]

    def discover(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        urls = self.sitemap_urls()
        if target.category == "Monetary Policy":
            yield from self.discover_mpr(target, urls)
            return
        if target.category == "Speeches":
            yield from self.discover_speeches(target, urls)
            return
        yield from super().discover(target)

    def sitemap_urls(self) -> list[str]:
        response = self.fetch(self.sitemap_url)
        return re.findall(r"<loc>(.*?)</loc>", response.text)

    def discover_mpr(self, target: ScrapeTarget, urls: list[str]) -> Iterable[DocumentCandidate]:
        pattern = re.compile(r"https://www\.bankofengland\.co\.uk/monetary-policy-report/(20\d{2})/[a-z]+-20\d{2}$")
        for page_url in self.sort_boe_urls(urls):
            match = pattern.match(page_url)
            if not match or int(match.group(1)) < START_YEAR:
                continue
            soup = self.get_soup(page_url)
            if not soup:
                continue
            title = self.extract_page_title(soup) or self.title_from_url(page_url)
            date_text = self.extract_published_date(soup) or self.extract_month_year(title) or self.extract_month_year(page_url) or match.group(1)
            if not is_in_scope(date_text):
                continue
            pdf_url = self.find_pdf_link(
                soup,
                page_url,
                required_patterns=(r"/-/media/boe/files/monetary-policy-report/20\d{2}/.*\.pdf$",),
                excluded_patterns=(r"press-conference", r"opening-remarks", r"slides", r"transcript"),
            )
            yield DocumentCandidate(
                bank=target.bank,
                category=target.category,
                title=title,
                date_original=date_text,
                source_url=pdf_url or page_url,
                notes=f"HTML source page: {page_url}" if pdf_url else "",
            )

    def discover_speeches(self, target: ScrapeTarget, urls: list[str]) -> Iterable[DocumentCandidate]:
        pattern = re.compile(r"https://www\.bankofengland\.co\.uk/speech/(20\d{2})/.+")
        for page_url in self.sort_boe_urls(urls):
            match = pattern.match(page_url)
            if not match or int(match.group(1)) < START_YEAR:
                continue
            soup = self.get_soup(page_url)
            if not soup:
                continue
            date_text = self.extract_published_date(soup) or match.group(1)
            if not is_in_scope(date_text):
                continue
            title = self.extract_page_title(soup) or self.title_from_url(page_url)
            page_haystack = f"{page_url} {title}"
            if re.search(r"\b(appendix|slides?|charts?|tables?|transcript|presentation)\b", page_haystack, re.I):
                continue
            pdf_url = self.find_pdf_link(
                soup,
                page_url,
                required_patterns=(rf"/-/media/boe/files/speech/{match.group(1)}/.*\.pdf$",),
                excluded_patterns=(r"appendix", r"slides?", r"charts?", r"tables?", r"transcript", r"presentation"),
            )
            speaker = self.extract_boe_speaker(title, soup)
            yield DocumentCandidate(
                bank=target.bank,
                category=target.category,
                title=title,
                date_original=date_text,
                source_url=pdf_url or page_url,
                speaker=speaker,
                notes=f"HTML source page: {page_url}" if pdf_url else "",
            )

    def extract_published_date(self, soup: BeautifulSoup) -> str:
        text = soup.get_text(" ", strip=True)
        match = re.search(r"Published on\s+(\d{1,2}\s+[A-Za-z]+\s+20\d{2})", text)
        return match.group(1) if match else ""

    def extract_page_title(self, soup: BeautifulSoup) -> str:
        h1 = soup.find("h1")
        if h1:
            title = self.clean_text(h1.get_text(" ", strip=True))
            if title and title.lower() not in {"speeches", "monetary policy reports"}:
                return title
        title_tag = soup.find("title")
        if title_tag:
            return self.clean_text(title_tag.get_text(" ", strip=True).split("|")[0])
        return ""

    def extract_month_year(self, value: str) -> str:
        match = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)[\s-]+(20\d{2})\b",
            value,
            re.I,
        )
        return f"{match.group(1)} {match.group(2)}" if match else ""

    def find_pdf_link(
        self,
        soup: BeautifulSoup,
        page_url: str,
        required_patterns: tuple[str, ...],
        excluded_patterns: tuple[str, ...] = (),
    ) -> str:
        for link in soup.find_all("a", href=True):
            href = urljoin(page_url, link["href"].split("#", 1)[0])
            text = self.clean_text(link.get_text(" ", strip=True))
            if ".pdf" not in href.lower():
                continue
            haystack = f"{href} {text}"
            if required_patterns and not any(re.search(pattern, href, re.I) for pattern in required_patterns):
                continue
            if excluded_patterns and any(re.search(pattern, haystack, re.I) for pattern in excluded_patterns):
                continue
            return href
        return ""

    def extract_boe_speaker(self, title: str, soup: BeautifulSoup) -> str:
        match = re.search(r"speech by\s+(.+)$", title, re.I)
        if match:
            return self.clean_text(match.group(1))
        text = soup.get_text(" ", strip=True)
        match = re.search(r"([A-Z][A-Za-z .'-]+)\s+Member of the Monetary Policy Committee", text)
        return self.clean_text(match.group(1)) if match else ""

    def sort_boe_urls(self, urls: list[str]) -> list[str]:
        month_order = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }

        def key(url: str) -> tuple[int, int, str]:
            year_match = re.search(r"/(20\d{2})/", url)
            month_match = re.search(r"/([a-z]+)-20\d{2}(?:/|$)", url)
            if not month_match:
                month_match = re.search(r"/20\d{2}/([a-z]+)(?:/|$)", url)
            year = int(year_match.group(1)) if year_match else 0
            month = month_order.get(month_match.group(1), 0) if month_match else 0
            return year, month, url

        return sorted(urls, key=key, reverse=True)
