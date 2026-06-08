from __future__ import annotations

from datetime import datetime
from typing import Optional

import dateparser

from src.config import START_YEAR


def parse_date(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    parsed = dateparser.parse(
        value,
        settings={
            "PREFER_DAY_OF_MONTH": "first",
            "RETURN_AS_TIMEZONE_AWARE": False,
            "DATE_ORDER": "DMY",
        },
    )
    return parsed


def is_in_scope(value: str | None, start_year: int = START_YEAR) -> bool:
    parsed = parse_date(value)
    return bool(parsed and parsed.year >= start_year and parsed <= datetime.now())


def au_date(value: str | datetime | None) -> str:
    parsed = value if isinstance(value, datetime) else parse_date(value)
    return parsed.strftime("%d/%m/%Y") if parsed else ""


def filename_date(value: str | datetime | None) -> str:
    parsed = value if isinstance(value, datetime) else parse_date(value)
    return parsed.strftime("%Y-%m-%d") if parsed else "Unknown Date"
