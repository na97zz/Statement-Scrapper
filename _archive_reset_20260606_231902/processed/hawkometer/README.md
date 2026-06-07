# Phrase-Based Hawkometer

This folder contains Phase 2 Hawkometer outputs. This phase is fully deterministic and does not use any local or remote LLM.

## What It Measures

The Hawkometer measures the balance of hawkish and dovish language in central bank documents using a fixed phrase library. Positive scores indicate tighter or more inflation-concerned language. Negative scores indicate easier or more growth/labour-market-concerned language.

## What It Does Not Measure

It does not forecast policy decisions, infer causality, understand strategy, or replace expert reading. It is a reproducible language signal designed to support later analysis.

## Phrase Scoring

Each phrase has a signed weight. Hawkish phrases are positive and dovish phrases are negative. Counts are scaled as `log1p(count) / log(2)`, then summed and clipped between -10 and +10.

Classification thresholds:

- `>= 2.5`: hawkish
- `1.0` to `< 2.5`: leaning hawkish
- `>-1.0` to `< 1.0`: neutral
- `>-2.5` to `<= -1.0`: leaning dovish
- `<= -2.5`: dovish

## Speaker Rolling Scores

Speaker scores are calculated for documents with a speaker name. The script calculates 90-day rolling average, 30-day rolling average, prior 60-day average, and the shift between recent and prior tone.

## Committee Scores

Committee scores are bank-date aggregates using the latest available 90-day rolling score per speaker. If `voter_status` is present, voters receive weight 1.0 and non-voters/observers receive 0.55.

## Meeting Scores

Meeting scores combine linked documents from the meeting-cycle layer:

- statement: 45%
- minutes: 25%
- press conference: 20%
- speeches: 10%

If a component is missing, weights are re-normalised across available components.

## Known Limitations

- Context blindness
- Weak handling of negation
- Weak handling of hypotheticals
- Novel phrases are not detected
- Translation effects across central banks
- No causal prediction

## How To Run

```powershell
python hawkometer_scorer.py --input processed/meeting_linkage/updated_master_index.csv --output processed/hawkometer
```

Optional:

```powershell
python hawkometer_scorer.py --phrase-library phrase_library.json --start-date 2021-01-01 --end-date 2026-12-31 --document-types speech statement minutes press_conference --min-phrase-matches 0
```
