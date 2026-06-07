# Hawkometer Backtest

Phase 4 validates the phrase-based Hawkometer against optional policy decision and market reaction data. It does not use an LLM and does not alter raw archive files.

## Hawkometer Signals

Each meeting receives a directional signal from `meeting_cycle_score`:

- `hawkish` if the score is at or above the hawkish threshold
- `dovish` if the score is at or below the dovish threshold
- `neutral` otherwise

Strong signals use the configured strong threshold, defaulting to +/-2.5.

## Policy Decision Alignment

When `policy_decisions.csv` is available, the backtest compares Hawkometer signals to the same meeting decision and the next meeting decision. Hikes or positive rate changes are treated as hawkish, cuts or negative rate changes as dovish, and holds or zero changes as neutral.

Expected `policy_decisions.csv` columns:

- `bank`
- `meeting_date`
- `meeting_id` optional
- `decision`
- `rate_change_bps`
- `policy_rate_after`
- `next_decision`
- `next_rate_change_bps`

## Score Change Tests

The score-change backtest checks whether hawkish shifts precede hikes, dovish shifts precede cuts, and stable communication precedes holds. It also buckets meeting score changes into intuitive ranges.

## Market Reactions

When `market_reactions.csv` is available, meeting scores are merged to asset/window reactions and summarised by Hawkometer signal and meeting lean.

Expected `market_reactions.csv` columns:

- `bank`
- `meeting_date`
- `meeting_id` optional
- `asset`
- `window`
- `change`
- `change_bps` optional
- `change_pct` optional

## Limitations

- Hawkometer is a communication tone index, not a rate prediction model.
- Policy decisions depend on data, forecasts and committee voting, not speeches alone.
- A hawkish score may still occur before a hold.
- Market reactions depend on expectations, not just absolute tone.
- Phrase-based scores may misread negation or hypotheticals.
- Missing external validation data limits conclusions.

## Run

```powershell
python hawkometer_backtest.py --hawkometer processed/hawkometer --analytics processed/hawkometer_analytics --meeting-linkage processed/meeting_linkage --output processed/hawkometer_backtest
```

Optional:

```powershell
python hawkometer_backtest.py --policy-decisions data/external/policy_decisions.csv --market-reactions data/external/market_reactions.csv --start-date 2021-01-01 --end-date 2026-12-31 --hawkish-threshold 1.0 --dovish-threshold -1.0 --strong-threshold 2.5
```
