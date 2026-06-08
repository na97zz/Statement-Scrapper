from __future__ import annotations

import re
from typing import Iterable

from src.config import START_YEAR, ScrapeTarget
from src.scraper_base import BaseScraper, DocumentCandidate


class NorgesBankScraper(BaseScraper):
    bank = "NORGES_BANK"
    targets = [
        ScrapeTarget(
            bank=bank,
            category="Monetary Policy",
            url="https://www.norges-bank.no/aktuelt/publikasjoner/?selectedFacets[Type]=11404&skip=0",
            include_keywords=(),
            include_url_patterns=(
                r"/aktuelt/publikasjoner/Pengepolitisk-rapport/20\d{2}/(?:ppr-\d+20\d{2}|pengepolitisk-rapport-\d+20\d{2})/?$",
                r"/contentassets/.*ppr.*\.pdf$",
            ),
            max_pages=30,
        ),
        ScrapeTarget(
            bank=bank,
            category="Speeches",
            url="https://www.norges-bank.no/en/news-events/news/?selectedFacets[Type]=11532&skip=0",
            include_keywords=("speech", "address", "remarks", ".pdf"),
            include_url_patterns=(r"/en/news-events/news/Speeches/20\d{2}/.*", r"/contentassets/.*\.pdf$"),
            exclude_url_patterns=(r"/\d{4}-\d{2}-\d{2}-pc/?$",),
            max_pages=30,
        ),
    ]

    def discover(self, target: ScrapeTarget) -> Iterable[DocumentCandidate]:
        if "skip=" not in target.url:
            yield from super().discover(target)
            return

        page_size = 10
        original_url = target.url
        for skip in range(0, target.max_pages * page_size, page_size):
            target_url = original_url.replace("skip=0", f"skip={skip}").replace("skip=10", f"skip={skip}")
            page_target = ScrapeTarget(
                bank=target.bank,
                category=target.category,
                url=target_url,
                include_keywords=target.include_keywords,
                exclude_keywords=target.exclude_keywords,
                include_url_patterns=target.include_url_patterns,
                exclude_url_patterns=target.exclude_url_patterns,
                years=target.years,
                prefer_pdf=target.prefer_pdf,
                max_pages=1,
            )
            candidates = list(super().discover(page_target))
            if not candidates:
                break
            for candidate in candidates:
                year_match = re.search(r"/(20\d{2})/", candidate.source_url)
                if year_match and int(year_match.group(1)) < START_YEAR:
                    continue
                yield candidate
