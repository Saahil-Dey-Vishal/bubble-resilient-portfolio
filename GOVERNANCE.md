# Scenario Governance and Reproducibility

## Scenario selection and calibration
- Start from narrative risks (AI correction, debt/credit stress, fragmented world/tariff shock, climate/food shock) and map each to numeric **driver shocks** (equity risk premium, credit spreads, rates/term premia, inflation expectations).
- Use historical analog windows to bound magnitudes (Dot-com, GFC, COVID, Crypto winter, 2025 tariff shock).
- Document the factor proxies used for each driver (currently: equity=QQQ, credit=HYG, rates=TLT, inflation proxy=TIP when available) and refresh quarterly.
- Record any manual overrides to shocks with date/time and rationale.

## Driver-to-asset-class mapping
- We compute recent betas of each asset vs factor proxies over a 1Y window and aggregate betas by AssetClass. This table lives in the notebook section **10.6 Driver-to-asset-class mapping**.
- Changes to roster or factor proxies should trigger a re-run and re-publication of the mapping table.

## Reproducibility controls
- Global seed: `RANDOM_SEED` (config), applied to `random`, `numpy`, and all RNGs derived from `numpy.random.default_rng`.
- Deterministic synthetic mode for CI/tests: set `USE_SYNTHETIC_DATA=1` to generate seeded synthetic prices (no network calls).
- Persisted benchmark choice: long-only benchmark built from current roster `RiskLimitPercent` (normalized) for relative ES.

## Testing and CI
- A smoke test runs the generator in synthetic mode to ensure the notebook builds end-to-end without network access.
- CI workflow: install deps, run smoke test with `USE_SYNTHETIC_DATA=1`.

## Early warning calibration
- To evaluate alert quality, supply labeled episodes per domain (start/end dates). The evaluation harness will compute precision/recall, lead time, and false-alarm rate. Calibration changes should be logged with dates and thresholds.
