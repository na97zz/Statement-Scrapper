from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

from src.config import START_YEAR, ScrapeTarget
from src.scraper_base import BaseScraper, DocumentCandidate
from src.utils.dates import is_in_scope


class RBNZScraper(BaseScraper):
    bank = "RBNZ"
    mps_listing_url = (
        "https://www.rbnz.govt.nz/monetary-policy/monetary-policy-statement/"
        "monetary-policy-statement-filtered-listing-page"
    )
    speeches_url = "https://www.rbnz.govt.nz/news-and-events/speeches#sort=%40computedsortdate%20descending"
    targets = [
        ScrapeTarget(
            bank=bank,
            category="Monetary Policy",
            url=mps_listing_url,
            include_url_patterns=(r"/publications/monetary-policy-statements/20\d{2}/.*\.pdf$",),
        ),
        ScrapeTarget(
            bank=bank,
            category="Minutes",
            url=mps_listing_url,
            include_keywords=("minutes", "record"),
            include_url_patterns=(r"/(minutes|record).*\.pdf$",),
        ),
        ScrapeTarget(
            bank=bank,
            category="Speeches",
            url=speeches_url,
            include_url_patterns=(r"/publications/speeches/20\d{2}/.*\.pdf$",),
        ),
    ]

    def fetch(self, url: str):
        response = curl_requests.get(url, impersonate="chrome124", timeout=30)
        response.raise_for_status()
        return response

    def discover(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        if target.category == "Monetary Policy":
            yield from self.discover_mps(target)
            return
        if target.category == "Minutes":
            yield from self.discover_minutes(target)
            return
        if target.category == "Speeches":
            yield from self.discover_speeches(target)
            return
        yield from super().discover(target)

    def discover_mps(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        for result in self.coveo_results(self.mps_listing_url, "Monetary Policy Statement filtered listing page"):
            title = result["title"]
            page_url = result["url"]
            date_text = result["date"]
            if not self.in_scope_result(date_text):
                continue
            soup = self.get_soup(page_url)
            if not soup:
                continue
            pdf_url = self.find_pdf_link(soup, page_url, required_url_part="/publications/monetary-policy-statements/")
            if not pdf_url:
                pdf_url = self.find_pdf_link(soup, page_url, required_url_part="/publications/monetary-policy-statement/")
            if not pdf_url:
                continue
            yield DocumentCandidate(
                bank=target.bank,
                category=target.category,
                title=title,
                date_original=date_text,
                source_url=pdf_url,
                notes=f"HTML source page: {page_url}",
            )

    def discover_minutes(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        for result in self.coveo_results(self.mps_listing_url, "Monetary Policy Statement filtered listing page"):
            title = result["title"]
            page_url = result["url"]
            date_text = result["date"]
            if not self.in_scope_result(date_text):
                continue
            soup = self.get_soup(page_url)
            if not soup:
                continue
            pdf_url = self.find_pdf_link(
                soup,
                page_url,
                required_url_part="/publications/",
                title_patterns=(r"\bmeeting minutes\b", r"\brecord of meeting\b", r"\bminutes of\b"),
                excluded_patterns=(
                    r"\d+\s+minutes?\s+to\s+(read|view)",
                    r"\bbriefing\b",
                    r"\bsnapshot\b",
                    r"\bslides?\b",
                    r"\bdata\b",
                    r"\bstatement\b",
                ),
            )
            if not pdf_url:
                continue
            yield DocumentCandidate(
                bank=target.bank,
                category=target.category,
                title=title,
                date_original=date_text,
                source_url=pdf_url,
                notes=f"HTML source page: {page_url}",
            )

    def discover_speeches(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        for result in self.coveo_results(self.speeches_url, "Speeches listing"):
            title = result["title"]
            page_url = result["url"]
            date_text = result["date"]
            if not self.in_scope_result(date_text):
                continue
            soup = self.get_soup(page_url)
            if not soup:
                continue
            pdf_url = self.find_pdf_link(soup, page_url, required_url_part="/publications/speeches/")
            if not pdf_url:
                continue
            yield DocumentCandidate(
                bank=target.bank,
                category=target.category,
                title=title,
                date_original=date_text,
                source_url=pdf_url,
                speaker=self.extract_speaker(title, soup.get_text(" ", strip=True), "Speeches"),
                notes=f"HTML source page: {page_url}",
            )

    def coveo_results(self, page_url: str, tab: str) -> Iterable[dict]:
        page = self.fetch(page_url).text
        token = re.search(r'token:\s+"([^"]+)"', page).group(1)
        endpoint = re.search(r'restEndpoint:\s+"([^"]+)"', page).group(1)
        expression_match = re.search(r'"dataExpression":\s*\[,\'(.*?)\'\]', page, re.S)
        expression = expression_match.group(1).replace("\\'", "'") if expression_match else ""
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Origin": "https://www.rbnz.govt.nz",
            "Referer": page_url,
        }
        first = 0
        while first < 500:
            body = {
                "q": "",
                "aq": expression,
                "searchHub": "RBNZ_rbnz-104-prod_web",
                "tab": tab,
                "firstResult": first,
                "numberOfResults": 50,
                "sortCriteria": "@computedsortdate descending",
                "fieldsToInclude": [
                    "computedtitle",
                    "clickableuri",
                    "uri",
                    "computedz95xpublisheddate",
                    "computedsortdate",
                    "z95xtemplatename",
                ],
            }
            response = curl_requests.post(endpoint, headers=headers, json=body, impersonate="chrome124", timeout=30)
            response.raise_for_status()
            data = response.json()
            rows = data.get("results", [])
            if not rows:
                break
            for row in rows:
                raw = row.get("raw", {})
                yield {
                    "title": self.clean_text(raw.get("computedtitle") or row.get("title") or ""),
                    "url": raw.get("clickableuri") or raw.get("uri") or row.get("clickUri") or row.get("uri") or "",
                    "date": self.epoch_ms_to_date(raw.get("computedz95xpublisheddate") or raw.get("computedsortdate")),
                }
            first += len(rows)

    def find_pdf_link(
        self,
        soup: BeautifulSoup,
        page_url: str,
        required_url_part: str,
        title_patterns: tuple[str, ...] = (),
        excluded_patterns: tuple[str, ...] = (),
    ) -> str:
        for link in soup.find_all("a", href=True):
            href = urljoin(page_url, link["href"].split("#", 1)[0])
            text = self.clean_text(link.get_text(" ", strip=True))
            if ".pdf" not in href.lower() or required_url_part.lower() not in href.lower():
                continue
            haystack = f"{href} {text}"
            if excluded_patterns and any(re.search(pattern, haystack, re.I) for pattern in excluded_patterns):
                continue
            if title_patterns and not any(re.search(pattern, haystack, re.I) for pattern in title_patterns):
                continue
            return href
        return ""

    def in_scope_result(self, date_text: str) -> bool:
        return bool(date_text and is_in_scope(date_text))

    def epoch_ms_to_date(self, value) -> str:
        if not value:
            return ""
        try:
            from datetime import datetime

            year = datetime.fromtimestamp(int(value) / 1000).year
            if year < START_YEAR:
                return ""
            return datetime.fromtimestamp(int(value) / 1000).strftime("%Y-%m-%d")
        except Exception:
            return str(value)
