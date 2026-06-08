from src.config import ScrapeTarget
from src.scraper_base import BaseScraper


class RBAScraper(BaseScraper):
    bank = "RBA"
    targets = [
        ScrapeTarget(
            bank=bank,
            category="Minutes",
            url="https://www.rba.gov.au/monetary-policy/rba-board-minutes/{year}/",
            include_keywords=(),
            include_url_patterns=(r"/monetary-policy/rba-board-minutes/20\d{2}/20\d{2}-\d{2}-\d{2}\.html?$",),
        ),
        ScrapeTarget(
            bank=bank,
            category="Monetary Policy",
            url="https://www.rba.gov.au/monetary-policy/int-rate-decisions/{year}/",
            include_keywords=(),
            include_url_patterns=(r"/media-releases/20\d{2}/mr-\d{2}-\d+\.html?$",),
        ),
        ScrapeTarget(
            bank=bank,
            category="Speeches",
            url="https://www.rba.gov.au/speeches/list.html",
            include_keywords=("speech", "/speeches/"),
            include_url_patterns=(r"/speeches/20\d{2}/[^/]+\.html?$",),
        ),
    ]
