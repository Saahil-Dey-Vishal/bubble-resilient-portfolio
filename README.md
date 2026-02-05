# Bubble‑Resilient Portfolio Notebook (2026–2030)

A refresh‑safe research notebook that pulls **live Yahoo Finance** data and delivers two product‑style outputs:

- **Section 12 — Bubble Early Warning System (EWS):** a simple, outsider‑friendly dashboard (Severity / Probability Now / 30D Change / Drift / Confidence).
- **Section 13 — Bubble‑aware Automated Portfolio Rebalancing Engine:** a **rule‑based** engine that converts the Section 12 regimes into portfolio tilts + limited sign flips, with guardrails.

> This is a research / decision‑support project — not financial advice.

## Why this exists
Most bubble discussions are narrative. This notebook is built to be:
- **Refresh‑safe:** rerun and the outputs update automatically with new Yahoo Finance data.
- **Explainable:** the dashboard + engine are transparent and auditable.
- **Action‑oriented:** it connects early warning signals to a concrete, guardrailed rebalancing recommendation.

## Data sources
- **Market data:** Yahoo Finance via `yfinance` (pulled live on every run)
- **Inputs:** a roster CSV that defines the assets, positioning intent, and risk limits

No database layer is used in this repo (CSV + live Yahoo pulls).

## How to run
1) Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```
2) Open the notebook and run all:
- `Bubble_Resilient_Portfolio.ipynb`

## Developer workflow (notebook generator)
This notebook is generated from:
- `Bubble_resilient_portfolio_notebook.py`

After making edits to the generator, regenerate the notebook:
```bash
python3 Bubble_resilient_portfolio_notebook.py
```

## Outputs
- `rebalance_recommendations_latest.csv` (optional export from Section 13 when enabled)

## Health check
A reproducible smoke test script is provided:
```bash
python3 scripts/health_check.py
```
It writes `health_check_report.md`.

## Disclaimer
This project is for research/education purposes only and does not constitute investment advice. Any results depend on live data availability and the chosen proxy tickers.
