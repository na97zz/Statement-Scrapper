from __future__ import annotations

import re
from typing import Iterable

from src.config import START_YEAR, ScrapeTarget
from src.scraper_base import BaseScraper, DocumentCandidate


class RiksbankScraper(BaseScraper):
    bank = "RIKSBANK"
    sitemap_url = "https://www.riksbank.se/sitemap.xml"
    targets = [
        ScrapeTarget(
            bank=bank,
            category="Monetary Policy",
            url="https://www.riksbank.se/en-gb/monetary-policy/monetary-policy-report/",
            include_url_patterns=(r"/en-gb/monetary-policy/monetary-policy-report/20\d{2}/.*",),
        ),
        ScrapeTarget(
            bank=bank,
            category="Speeches",
            url="https://www.riksbank.se/en-gb/press-and-published/speeches-and-presentations/",
            include_url_patterns=(r"/en-gb/press-and-published/speeches-and-presentations/20\d{2}/.*",),
        ),
    ]

    def discover(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        urls = self.sitemap_urls()
        if target.category == "Monetary Policy":
            yield from self.discover_from_sitemap(target, urls, r"/en-gb/monetary-policy/monetary-policy-report/(20\d{2})/")
            return
        if target.category == "Speeches":
            yield from self.discover_from_sitemap(target, urls, r"/en-gb/press-and-published/speeches-and-presentations/(20\d{2})/")
            return
        yield from super().discover(target)

    def sitemap_urls(self) -> list[str]:
        response = self.fetch(self.sitemap_url)
        return re.findall(r"<loc>(.*?)</loc>", response.text)

    def discover_from_sitemap(self, target: ScrapeTarget, urls: list[str], pattern: str) -> Iterable[DocumentCandidate]:
        regex = re.compile(pattern)
        for page_url in self.sort_urls(url for url in urls if regex.search(url)):
            match = regex.search(page_url)
            if not match or int(match.group(1)) < START_YEAR:
                continue
            if re.search(r"/20\d{2}/?$", page_url):
                continue
            yield DocumentCandidate(
                bank=target.bank,
                category=target.category,
                title=self.clean_riksbank_title(self.title_from_url(page_url)),
                date_original=match.group(1),
                source_url=page_url,
            )

    def clean_riksbank_title(self, title: str) -> str:
        title = re.sub(r"\s+Date:\s*\d{1,2}/\d{1,2}/20\d{2}.*$", "", title)
        title = re.sub(r"^\d{1,2}/\d{1,2}/20\d{2}\s+", "", title)
        return self.clean_text(title)

    def sort_urls(self, urls: Iterable[str]) -> list[str]:
        def key(url: str) -> tuple[str, str]:
            return self.extract_date_text(url), url

        return sorted(set(urls), key=key, reverse=True)
