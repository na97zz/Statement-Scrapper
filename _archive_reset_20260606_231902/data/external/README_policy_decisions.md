# Policy Decisions Builder

`policy_decisions.csv` is an external validation input for Phase 4 Hawkometer backtesting. It is generated from the scraped central bank meeting documents and can later be replaced or augmented with official rate datasets.

## Extraction Method

The builder starts from `processed/meeting_linkage/meeting_index.csv`, then looks for linked statement, monetary policy decision, interest rate announcement, press release, monetary policy, and minutes documents. It reads the selected text file and searches for policy-rate and basis-point decision phrases.

## Fed Target Ranges

For Federal Reserve target ranges, the midpoint is stored in `policy_rate_after`. The original target range is noted in `notes`.

## Confidence

- `1.0`: explicit basis-point change and policy rate after found
- `0.9`: explicit hold/unchanged and policy rate after found
- `0.8`: explicit basis-point change found but no rate after
- `0.7`: rate change calculated from consecutive policy rates
- `0.5`: decision inferred from hold keywords only
- `0.0`: unknown

## Manual Review

Review low-confidence rows in `policy_decisions_qa.csv`, especially rows with `unknown`, blank rates, or confidence below `0.7`. The QA file includes an excerpt around the matched phrase.

## Rerun Phase 4

```powershell
python hawkometer_backtest.py --hawkometer processed/hawkometer --analytics processed/hawkometer_analytics --meeting-linkage processed/meeting_linkage --output processed/hawkometer_backtest --policy-decisions data/external/policy_decisions.csv
```
