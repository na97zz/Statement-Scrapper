# FED Phase 1 Meeting Linkage

This folder contains a FED-only Phase 1 meeting linkage dataset built from `data/master_index.csv` and the raw FED archive.

## Files

- `fed_master_index.csv`: FED-only master index with document subtype and meeting linkage columns.
- `fed_meeting_index.csv`: one row per FOMC statement anchor meeting.
- `fed_meeting_documents.csv`: one row per linked FED document.
- `fed_linkage_summary.csv`: basic counts and linkage coverage.

## Meeting IDs

Meeting IDs use:

```text
FED_YYYY_MM
```

## Linkage Rules

- FOMC statements are meeting anchors.
- Minutes link to the latest FOMC meeting on or before the minutes date.
- SEP and press conference documents link to the nearest same-day or same-month meeting.
- Speeches link to the meeting window from the latest FOMC statement through the day before the next statement.

## Run

```powershell
python fed_phase1_meeting_linkage.py --input data/master_index.csv --output processed/fed
```

## FED-Only Hawkometer Scoring

After Phase 1 linkage, run:

```powershell
python fed_run_hawkometer.py --input processed/fed/fed_master_index.csv --meeting-docs processed/fed/fed_meeting_documents.csv --phrase-library hawkometer.json --output processed/fed/hawkometer
```

The scoring runner uses the existing `src.hawkometer.scorer.HawkometerScorer` logic. It filters to FED rows only and writes:

- `processed/fed/hawkometer/fed_document_scores.csv`
- `processed/fed/hawkometer/fed_speaker_scores.csv`
- `processed/fed/hawkometer/fed_meeting_scores.csv`
- `processed/fed/hawkometer/fed_hawkometer_summary.csv`
- `processed/fed/hawkometer/fed_hawkometer.json`
