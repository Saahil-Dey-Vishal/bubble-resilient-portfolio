# Monte Carlo + Event DB Validation Report

- Run at: **2026-02-05 21:07:55 UTC**
- Python: `3.14.2`
- Platform: `macOS-26.2-arm64-arm-64bit-Mach-O`

## Dependency probe
- numpy: 2.4.1
- pandas: 3.0.0
- scipy: 1.17.0
- statsmodels: 0.14.6
- yfinance: 1.0

## Inputs
- Event DB: `/Users/saahildey/Bubble Resilient Portfolio/bubble_events_database.csv`
- Domains: ['AI bubble', 'Private Credit bubble', 'Crypto bubble']
- Total tickers downloaded (domains + macro): 34

## Real-history validation (event DB)
| domain | auc | best_f1 | best_thr | events_labeled_days | prob_last |
| --- | --- | --- | --- | --- | --- |
| AI bubble | 0.376 | 0.404 | 0.000 | 977.000 | 5.186 |
| Crypto bubble | 0.235 | 0.441 | 0.000 | 746.000 | 0.448 |
| Private Credit bubble | 0.325 | 0.492 | 0.000 | 1260.000 | 9.689 |

## Synthetic bubble injection + Monte Carlo
Simulation settings:
- N_SIM=10 per domain
- HORIZON_DAYS=756
- BUBBLE_PROB=0.50
- BUBBLE_BUILD_DAYS=252
- CRASH_DAYS=21
- CRASH_MAG=-0.40

| domain | detected_rate | lead_days_mean | lead_days_p50 | lead_days_p10 | lead_days_p90 | n_sims | bubble_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AI bubble | 1.000 | 537.500 | 556.000 | 475.800 | 560.000 | 10.000 | 0.500 |
| Crypto bubble | 1.000 | 550.300 | 557.000 | 531.800 | 560.000 | 10.000 | 0.500 |
| Private Credit bubble | 1.000 | 544.900 | 551.500 | 524.600 | 560.000 | 10.000 | 0.700 |