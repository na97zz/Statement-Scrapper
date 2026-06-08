# FED Hawkometer Scores

This folder contains FED-only Hawkometer outputs built from `processed/fed/fed_master_index.csv` using the existing `src.hawkometer.scorer.HawkometerScorer` logic and local phrase library.

## Run

```powershell
python fed_run_hawkometer.py --input processed/fed/fed_master_index.csv --meeting-docs processed/fed/fed_meeting_documents.csv --phrase-library hawkometer.json --output processed/fed/hawkometer
```

If `hawkometer.json` is not present at the project root, the runner automatically uses `data/hawkometer.json`.

## Outputs

- `fed_document_scores.csv`: one row per FED document.
- `fed_speaker_scores.csv`: speaker-level averages.
- `fed_meeting_scores.csv`: meeting-level weighted scores.
- `fed_hawkometer_summary.csv`: key run summary.
- `fed_hawkometer.json`: JSON bundle of all outputs.
