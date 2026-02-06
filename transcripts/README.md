# Earnings-call transcripts scaffold (local)

This folder is an **optional, local-only** input for the Bubble Early Warning System.

If you enable `RUN_TRANSCRIPTS_NLP = True` in `Bubble_resilient_portfolio_notebook.py`, Section 12 will scan this folder (or `EWS_TRANSCRIPTS_DIR`) for transcript text files and compute:

- `transcript_heat_0_100` (0–100): a lightweight keyword + risk-language proxy
- `transcript_nudge_buckets` (0–2): a small severity nudge used in `Nudge (+buckets)`

## Enable

- Set `RUN_TRANSCRIPTS_NLP = True` (near the top of `Bubble_resilient_portfolio_notebook.py`)
- Put transcript files under:
  - `transcripts/` (default), or
  - set `EWS_TRANSCRIPTS_DIR=/path/to/transcripts`

Optional tunables (env vars):
- `EWS_TRANSCRIPTS_MAX_FILES_PER_TICKER` (default: 2)
- `EWS_TRANSCRIPTS_TOP_TICKERS_PER_DOMAIN` (default: 3)
- `EWS_TRANSCRIPTS_MAX_CHARS_PER_DOC` (default: 250000)

## Accepted file formats

- `.txt`, `.md` (plain text)
- `.json` (simple schema; see below)

### Filename convention

The scorer infers the ticker from the **start of the filename** (before any separators).

Examples:
- `NVDA_2025Q4.txt`
- `MSFT-2026Q1.md`
- `COIN_2025-11-01.json`

### JSON schema (minimal)

```json
{
  "ticker": "NVDA",
  "date": "2025-11-20",
  "text": "...full transcript text..."
}
```

Alternatively, you can store segmented text:

```json
{
  "ticker": "NVDA",
  "segments": [{"text": "..."}, {"text": "..."}]
}
```

## Notes

- This is a best-effort **lexicon proxy**, not an LLM sentiment model.
- No scraping is performed; you control what goes into this folder.
