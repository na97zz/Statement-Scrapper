from src.scrapers.boc import BOCScraper
from src.scrapers.boe import BOEScraper
from src.scrapers.boj import BOJScraper
from src.scrapers.ecb import ECBScraper
from src.scrapers.fed import FEDScraper
from src.scrapers.norges import NorgesBankScraper
from src.scrapers.rba import RBAScraper
from src.scrapers.rbnz import RBNZScraper
from src.scrapers.riksbank import RiksbankScraper
from src.scrapers.snb import SNBScraper


SCRAPER_CLASSES = [
    FEDScraper,
    ECBScraper,
    RBAScraper,
    BOJScraper,
    SNBScraper,
    RBNZScraper,
    BOEScraper,
    BOCScraper,
    RiksbankScraper,
    NorgesBankScraper,
]
