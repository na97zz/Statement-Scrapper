from __future__ import annotations

import argparse

from src.scrapers import SCRAPER_CLASSES


def preview(limit: int, selected_banks: list[str] | None = None) -> None:
    for scraper_cls in SCRAPER_CLASSES:
        if selected_banks and scraper_cls.bank not in selected_banks:
            continue
        scraper = scraper_cls()
        print(f"\n## {scraper.bank}")
        for target in scraper.targets:
            print(f"\n### {target.category}")
            count = 0
            try:
                for candidate in scraper.dedupe_candidates(list(scraper.discover(target))):
                    count += 1
                    line = f"{count}. {candidate.date_original or 'Unknown date'} | {candidate.title} | {candidate.source_url}"
                    print(line.encode("ascii", "replace").decode("ascii"))
                    if count >= limit:
                        break
                if count == 0:
                    print("No candidates found.")
            except Exception as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview scraper candidates without downloading files.")
    parser.add_argument("--limit", type=int, default=10, help="Number of candidates per bank/category.")
    parser.add_argument("--banks", nargs="*", help="Optional bank codes, e.g. FED ECB RBA.")
    args = parser.parse_args()
    preview(args.limit, args.banks)


if __name__ == "__main__":
    main()
