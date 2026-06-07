# Policy Decisions QC Workflow

This workflow reviews `policy_decisions.csv` without modifying it. The raw extraction remains unchanged. The backtest-ready output is `policy_decisions_final.csv`.

## Files

- `policy_decisions.csv`: raw automated extraction from scraped central bank documents.
- `policy_decisions_review.csv`: rows flagged for review, with reasons and suggested corrections.
- `policy_decisions_suggested_corrections.csv`: high-confidence automatic suggestions.
- `policy_decisions_overrides_template.csv`: editable override template. Suggestions are prefilled where available and `approved` is `false`.
- `policy_decisions_overrides.csv`: optional manual file you create by copying/editing the template.
- `policy_decisions_final.csv`: raw extraction plus approved overrides and QC metadata.

## Review Process

Open `policy_decisions_review.csv` and start with `qc_priority = high`. Check the current decision, the suggestion, and the matched excerpt. Unknown or inconsistent rows should be reviewed first.

## Overrides

Copy `policy_decisions_overrides_template.csv` to `policy_decisions_overrides.csv`, edit any rows you want to approve, and set `approved` to `true`. Approved rows replace decision, rate change, policy rate after, confidence, and notes in the final file.

## Rerun QC

```powershell
python policy_decisions_qc.py --input data/external/policy_decisions.csv --qa data/external/policy_decisions_qa.csv --meeting-linkage processed/meeting_linkage --output-dir data/external
```

## Rerun Phase 4 Backtest

```powershell
python hawkometer_backtest.py --hawkometer processed/hawkometer --analytics processed/hawkometer_analytics --meeting-linkage processed/meeting_linkage --output processed/hawkometer_backtest --policy-decisions data/external/policy_decisions_final.csv
```

Suggestions are aids, not truth. Review and approve corrections manually before treating them as validated policy decisions.
