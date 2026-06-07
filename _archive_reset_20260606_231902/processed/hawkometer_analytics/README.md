# Hawkometer Analytics Layer

Phase 3 turns Phase 2 document scores into research-ready analytics outputs. It does not use an LLM and does not alter raw scraped data.

## Bank Trends

Bank trends aggregate scored documents by bank and date. The script calculates daily average scores plus 30-day, 90-day, and 180-day rolling averages.

## Momentum

Momentum is `rolling_30d_score - rolling_90d_score`. Positive momentum means recent language is becoming more hawkish relative to the 90-day baseline. Negative momentum means recent language is becoming more dovish.

## Speaker Shift Alerts

Speaker alerts use the Phase 2 rolling speaker scores. A hawkish alert is created when `shift_30_vs_prior_60` is above the configured threshold. A dovish alert is created when it is below the negative threshold.

Alert strength:

- `mild`: absolute shift from 1.0 to below 2.0
- `strong`: absolute shift from 2.0 to below 3.5
- `extreme`: absolute shift 3.5 or above

## Meeting Cycle Analysis

Meeting analysis compares each meeting score with the prior meeting for the same bank. It also calculates 3-meeting and 6-meeting rolling averages to show meeting-cycle trend.

## Cross-Bank Ranking

The ranking uses each bank's latest 90-day score, latest meeting score, and latest 30-day score. The default weighted score is 50% latest 90-day, 30% latest meeting, and 20% latest 30-day, with weights re-normalised when a component is missing.

## Limitations

These analytics inherit Phase 2 limitations: phrase scoring is context-blind, weak on negation and hypotheticals, sensitive to missing text, and does not detect novel wording. Rankings are language indicators, not forecasts or causal estimates.

## Run

```powershell
python hawkometer_analytics.py --input processed/hawkometer --meeting-linkage processed/meeting_linkage --output processed/hawkometer_analytics
```

Optional:

```powershell
python hawkometer_analytics.py --start-date 2021-01-01 --end-date 2026-12-31 --shift-threshold 1.0 --latest-window-days 120
```
