# Detailed Policy Decisions Review

`policy_decisions_review_detailed.csv` and `.xlsx` are self-contained review files for policy decision QC. They include the current extraction, suggested correction, matched phrase, source metadata, and enough context to review decisions without opening source documents.

## How To Review

Open the Excel file and filter `qc_priority = high` first. Read `full_context`, compare current and suggested fields, then edit:

- `final_decision`
- `final_rate_change_bps`
- `final_policy_rate_after`
- `final_confidence`
- `reviewer_notes`

Set `approved` to `TRUE` only for rows you want applied.

## Apply Approved Rows

CSV:

```powershell
python policy_decisions_qc_detailed.py --mode apply-review --input data/external/policy_decisions.csv --review data/external/policy_decisions_review_detailed.csv --output-dir data/external
```

Excel:

```powershell
python policy_decisions_qc_detailed.py --mode apply-review --input data/external/policy_decisions.csv --review data/external/policy_decisions_review_detailed.xlsx --output-dir data/external
```

This creates `policy_decisions_overrides.csv` and `policy_decisions_final.csv`.

## Rerun Phase 4 Backtest

```powershell
python hawkometer_backtest.py --hawkometer processed/hawkometer --analytics processed/hawkometer_analytics --meeting-linkage processed/meeting_linkage --output processed/hawkometer_backtest --policy-decisions data/external/policy_decisions_final.csv
```
