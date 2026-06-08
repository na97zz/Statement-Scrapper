# FED Hawkometer Dashboard

Run from the project root:

```powershell
streamlit run fed_hawkometer_dashboard.py
```

## Required Files

The dashboard is FED only and expects these files:

- `processed/fed/fed_master_index.csv`
- `processed/fed/fed_meeting_index.csv`
- `processed/fed/fed_meeting_documents.csv`
- `processed/fed/hawkometer/fed_document_scores.csv`
- `processed/fed/hawkometer/fed_speaker_scores.csv`
- `processed/fed/hawkometer/fed_meeting_scores.csv`
- `processed/fed/hawkometer/fed_hawkometer_summary.csv`
- `processed/fed/hawkometer/fed_hawkometer.json`

## Pages

- Overview: FED corpus KPIs, latest documents, distribution charts, and current regime versus 2021 baseline.
- Meeting Timeline: FOMC meeting-cycle scores with 3-meeting and 6-meeting rolling averages.
- Speaker Dashboard: speaker rankings, rolling speaker scores, and shift alerts.
- Meeting Explorer: meeting-level score components and linked document table.
- Document Explorer: document filters, matched hawkish/dovish phrase summaries, metadata, and source text preview.
- Phrase Match Explorer: top phrases, phrase frequency, phrase usage over time, and documents containing a selected phrase.
- Regime View: yearly regime comparison from 2021 through 2026, heatmaps, phrase evolution, and speaker regime changes.
- Summary / Methodology: scoring method, classification thresholds, meeting weights, limitations, and CSV exports.

## Method Notes

The dashboard treats 2021 as the historical baseline. It calculates dashboard meeting-cycle scores with:

- Statement = 45%
- Minutes = 25%
- Press conference = 20%
- Speeches = 10%

When a component is missing, available components are reweighted proportionally.

## Known Limitations

- Phrase scoring can miss context, negation, and hypothetical language.
- Speaker labels depend on the scraped metadata.
- A zero-match document is not automatically neutral in an economic sense; it means no library phrase was detected.
- This dashboard has no multi-bank mode and no local LLM features yet.
