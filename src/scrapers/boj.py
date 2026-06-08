from src.config import ScrapeTarget
from src.scraper_base import BaseScraper


class BOJScraper(BaseScraper):
    bank = "BOJ"
    targets = [
        ScrapeTarget(
            bank=bank,
            category="Monetary Policy",
            url="https://www.boj.or.jp/en/mopo/mpmdeci/state_{year}/index.htm",
            include_keywords=("statement", "monetary policy", ".pdf", ".htm"),
            include_url_patterns=(r"/en/mopo/mpmdeci/mpr_20\d{2}/k\d+[a-z]?\.pdf$",),
            exclude_url_patterns=(r"/state_20\d{2}/index\.htm$", r"/state_all/index\.htm$"),
        ),
        ScrapeTarget(
            bank=bank,
            category="Speeches",
            url="https://www.boj.or.jp/en/about/press/koen_{year}/index.htm",
            include_keywords=("speech", "remarks", ".pdf", ".htm"),
            include_url_patterns=(r"/en/about/press/koen_20\d{2}/ko\d+[a-z]?\.htm$", r"/en/about/press/koen_20\d{2}/.*\.pdf$"),
            exclude_url_patterns=(r"/koen_20\d{2}/index\.htm$", r"/koen_all/index\.htm$"),
        ),
    ]
