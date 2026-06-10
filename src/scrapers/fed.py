from src.config import ScrapeTarget
from src.scraper_base import BaseScraper


class FEDScraper(BaseScraper):
    bank = "FED"
    targets = [
        ScrapeTarget(
            bank=bank,
            category="Monetary Policy",
            url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            include_keywords=("monetary20", "fomcstatement", "fomc statement", "statement"),
            exclude_keywords=("minutes", "financial statements", "reportingforms", "press release"),
            include_url_patterns=(
                r"/monetarypolicy/files/monetary20\d{6}a1\.pdf$",
                r"/newsevents/pressreleases/monetary20\d{6}a\.htm$",
            ),
        ),
        ScrapeTarget(
            bank=bank,
            category="Monetary Policy",
            url="https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm",
            include_keywords=("monetary20", "fomcstatement", "fomc statement", "statement"),
            exclude_keywords=("minutes", "financial statements", "reportingforms", "press release"),
            include_url_patterns=(
                r"/newsevents/pressreleases/monetary20\d{6}a\.htm$",
                r"/newsevents/press/monetary/20\d{6}[a-z]?\.htm$",
                r"/boarddocs/press/(?:general|monetary)/20\d{2}/20\d{6}/(?:default\.htm)?$",
            ),
            years=tuple(range(2020, 1999, -1)),
        ),
        ScrapeTarget(
            bank=bank,
            category="Minutes",
            url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            include_keywords=("fomcminutes", "minutes"),
            exclude_keywords=("financial statements", "reportingforms"),
            include_url_patterns=(
                r"/monetarypolicy/files/fomcminutes20\d{6}\.pdf$",
                r"/monetarypolicy/fomcminutes20\d{6}\.htm$",
            ),
        ),
        ScrapeTarget(
            bank=bank,
            category="Speeches",
            url="https://www.federalreserve.gov/newsevents/{year}-speeches.htm",
            include_keywords=("/newsevents/speech/", "/newsevents/testimony/"),
            exclude_keywords=("speeches-testimony.htm", "feeds/", "videos", "calendar"),
            include_url_patterns=(r"/newsevents/(speech|testimony)/[a-z]+20\d{6}[a-z]?\.htm$",),
        ),
    ]
