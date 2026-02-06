# Bubble‑Resilient Portfolio (AI, Crypto, Private Credit) Notebook (2026–2030)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Live Data](https://img.shields.io/badge/data-Yahoo%20Finance-orange.svg)](https://finance.yahoo.com/)

A **production-grade research notebook** that detects asset bubbles across AI, Crypto, and Private Credit — and automatically rebalances to protect against crashes.

## 🎯 What This Does

| Output | Description |
|--------|-------------|
| **Section 5 — Early Warning System** | Real-time bubble detection: Severity, Probability, Babson Stage (Stealth → Mania → Blow-off), Drift Flags |
| **Section 6 — Rebalancing Engine** | Rule-based portfolio tilts with Mean–CVaR optimizer and guardrails |
| **Section 6.2 — Analysis Dashboard** | Risk/Return frontier, Drawdown Protection, Monte Carlo simulations |
| **Section 7 — Walk-Forward Backtest** | Out-of-sample proof that the strategy is tradeable (not just theoretical) |

## 📊 Key Result (Section 7)

```
┌─────────────────────────────────────────────────────────────────┐
│  BUBBLE-PROTECTION STRATEGY BACKTEST RESULTS                   │
├─────────────────────────────────────────────────────────────────┤
│  Starting Capital: $100        Transaction Cost: 10 bps/trade  │
│                                                                 │
│  ✅ Profitable: $112 (12% gain)                                │
│  ✅ Positive Sharpe: 0.07 (risk-adjusted positive)             │
│  ✅ Bubble Signals Fired: 13 events detected                   │
│  ⚠️  Lags Buy-and-Hold by ~15% — this IS the insurance premium │
└─────────────────────────────────────────────────────────────────┘
```

**The $112 vs $131 gap is intentional.** This is a *bubble-protection* strategy, not a *beat-the-market* strategy. You're paying ~15% for crash insurance — like car insurance, you hope you never need it.

> ⚠️ This is a research / decision‑support project — not financial advice.

## Why this exists
Most bubble discussions are narrative. This notebook is built to be:
- **Refresh‑safe:** rerun and the outputs update automatically with new Yahoo Finance data.
- **Explainable:** the dashboard + engine are transparent and auditable.
- **Action‑oriented:** it connects early warning signals to a concrete, guardrailed rebalancing recommendation.
- **Downside-focused:** the rebalancing engine prioritizes crash protection over raw returns — lower expected return is the "insurance premium" for bubble resilience.

## 📚 Notebook Structure

| Section | Name | What It Does |
|---------|------|--------------|
| 0 | Header | Overview and navigation |
| 0.5 | Dependencies | Package installation |
| 1 | Config | Load CSV roster, define stress windows |
| 2 | Returns | Download live prices from Yahoo Finance |
| 3 | Data Quality | Ticker status check, data validation |
| 4 | Portfolio | Current weights, exposure, risk metrics |
| **5** | **Early Warning System** | Bubble probability, Babson Stage, drift detection |
| **6** | **Automated Rebalancing** | Rule-based tilts, sign flips, Mean–CVaR optimizer |
| **6.2** | **Analysis Dashboard** | Risk/Return, Drawdown Protection, Monte Carlo |
| **7** | **Walk-Forward Backtest** | Out-of-sample proof the strategy is tradeable |

## 🔬 How the Bubble Detection Works

### Babson Barometer (Section 5)
Inspired by Roger Babson's 1929 crash prediction. Classifies bubble lifecycle stages:

| Stage | Probability | Signal |
|-------|-------------|--------|
| 🟢 **Stealth** | <25% | Early accumulation, few recognize the trend |
| 🟡 **Awareness** | 25-50% | Institutional interest grows |
| 🟠 **Mania** | 50-80% | Retail FOMO, media hype |
| 🔴 **Blow-off** | >80% | Peak euphoria, extreme valuations |
| ⚫ **Return to Mean** | ≥60% falling | Crash in progress |

### The Insurance Premium Concept (Section 7)
A bubble-resilient portfolio **intentionally accepts lower returns** in exchange for crash protection:
- In bull markets with no crashes: strategy lags buy-and-hold (~15%)
- During bubble collapses: strategy protects capital
- Think of it like insurance: you pay premiums hoping you never need to claim

## 📦 Data Sources
| Source | Description |
|--------|-------------|
| Yahoo Finance | Live market data via `yfinance` (pulled on every run) |
| Ticker Roster CSV | 50+ assets with positioning intent and risk limits |
| Bubble Events DB | 20+ historical bubble events for validation |

## 🚀 Quick Start
```bash
# Clone and install
git clone https://github.com/Saahil-Dey-Vishal/bubble-resilient-portfolio.git
cd bubble-resilient-portfolio
pip install -r requirements.txt

# Run the notebook
jupyter notebook Bubble_Resilient_Portfolio.ipynb

# Or run health check
python3 scripts/health_check.py
```

## 📁 Project Structure
```
├── Bubble_Resilient_Portfolio.ipynb    # Main notebook (7 sections)
├── FICC_and_Alternatives_*.csv         # 50-ticker roster
├── bubble_events_database.csv          # Historical validation data
├── rebalance_recommendations_latest.csv # Output: recommended weights
├── scripts/                            # Maintenance & health check scripts
└── requirements.txt                    # Python dependencies
```

## ⚙️ Outputs
- **`rebalance_recommendations_latest.csv`** — Recommended portfolio weights from Section 6
- **3 live charts** — Risk/Return, Drawdown, Monte Carlo visualizations
- **Walk-forward backtest** — Out-of-sample performance metrics

## License
Licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## ⚠️ Disclaimer
This project is for **research and education purposes only** and does not constitute investment advice. Results depend on live data availability and chosen proxy tickers. Past performance does not guarantee future results.

---

Built with Python, pandas, scipy, and a healthy respect for market history.
