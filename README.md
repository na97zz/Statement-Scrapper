# Central Bank Monetary Policy Scraper

Python scraper project for collecting central bank monetary policy documents, minutes, speeches, and ECB press conference Q&A from 2021 onward. It saves raw source files, extracted text, per-document metadata, a CSV master index, and a JSONL master index for later LLM hawkish/neutral/dovish scoring.

The scoring layer is intentionally not included.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Playwright is included for future JavaScript-heavy pages. The current pipeline uses `requests` and BeautifulSoup first.

```powershell
playwright install
```

## Run the Dashboard

```powershell
streamlit run src/app.py
```

Use the dashboard to:

- Start scraping
- Stop scraping
- Clear all data
- View progress by central bank
- View logs
- Inspect the summary table

## Output Location

Files are saved under:

```text
data/
```

Each document is stored by:

```text
data/{BANK}/{CATEGORY}/{YYYY-MM-DD - Title}
```

Speech files include the speaker:

```text
YYYY-MM-DD - Speaker Name - Speech Title.txt
```

Each scraped document gets:

- Raw `.pdf` or `.html`
- Extracted `.txt`
- Metadata `.json`

Indexes are maintained at:

```text
data/master_index.csv
data/master_index.jsonl
data/failed_scrapes.csv
```

## Verification

After scraping, check:

1. `data/master_index.csv` contains rows for each successful document.
2. Each row has `raw_file_path` and `text_file_path`.
3. Open a `.txt` file and confirm it starts with the metadata header, followed by `==== EXTRACTED TEXT START ====`.
4. Check `data/failed_scrapes.csv` for pages that failed without stopping the run.
5. Rerun the scraper and confirm duplicates are logged as skipped rather than downloaded again.

Before a long scrape, preview candidate URLs without downloading:

```powershell
python -m src.preview_candidates --limit 10
```

Preview one or two banks:

```powershell
python -m src.preview_candidates --banks ECB FED --limit 20
```

## Project Layout

```text
src/
  app.py
  config.py
  runner.py
  scraper_base.py
  state.py
  utils/
    dates.py
    files.py
    logging_utils.py
    pdf.py
  scrapers/
    fed.py
    ecb.py
    rba.py
    boj.py
    snb.py
    rbnz.py
    boe.py
    boc.py
    riksbank.py
    norges.py
```

## Notes

The scraper is polite by default: it checks `robots.txt` where possible, uses retry logic, and sleeps between requests. Central bank sites change over time, so bank-specific modules are deliberately thin and easy to update.

## Meeting Linkage For Hawkometer Scoring

`meeting_linkage.py` builds a meeting-cycle layer on top of `data/master_index.csv`. The purpose is to let the Hawkometer scorer analyse a whole policy cycle as one unit instead of scoring isolated documents.

The script classifies each document into a subtype:

- `statement`
- `minutes`
- `press_conference`
- `sep`
- `speech`
- `testimony`
- `other`

Meeting anchors are official monetary policy documents such as statements, monetary policy decisions, interest rate announcements, monetary policy reports, and monetary policy assessments. Each anchor creates a `meeting_id` using:

```text
BANK_YYYY_MM
```

If a bank has multiple anchor documents in the same month, a suffix is added:

```text
FED_2026_03_A
FED_2026_03_B
```

Documents are linked using deterministic rules:

- Statement anchor documents link to their own meeting.
- SEP and press conference documents link to the nearest same-day or same-month meeting.
- Minutes link to the most recent prior same-bank meeting.
- Speeches link to the most recent same-bank meeting window.
- Other documents link only if they fall inside a meeting window.

Run the linkage step with:

```powershell
python meeting_linkage.py --input data/master_index.csv --output processed/meeting_linkage
```

Optional date and window controls:

```powershell
python meeting_linkage.py --input data/master_index.csv --output processed/meeting_linkage --speech-window-days 45 --min-date 2021-01-01 --max-date 2026-12-31
```

The output folder contains:

- `updated_master_index.csv`: the master index with `doc_id`, `document_subtype`, `meeting_id`, `meeting_relation`, `meeting_date`, `days_after_meeting`, `linkage_confidence`, and `linkage_reason`.
- `meeting_index.csv`: one row per meeting anchor, including the meeting window and next meeting date.
- `meeting_documents.csv`: one row per linked document, ready for meeting-cycle scoring.
- `linkage_summary.csv`: bank-level counts and linkage coverage.
