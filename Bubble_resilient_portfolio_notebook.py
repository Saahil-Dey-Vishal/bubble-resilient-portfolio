from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    nb = nbf.v4.new_notebook()

    title_md = """# Bubble-Resilient Multi-Asset Portfolio (FICC + Alternatives) (2026–2030)

**Objective:** Build a roster-driven **multi-asset** long/short portfolio (FICC + alternatives) designed to be resilient during potential bubble bursts in **AI bubble / Private Credit bubble / Crypto bubble** expected in **2026–2030**.

**Live data policy (non-negotiable):** All prices/returns are pulled **live** from Yahoo Finance via `yfinance` each time you run the notebook. The roster CSV only supplies symbols/metadata/positioning; no static return series.

**Roster file:** `FICC_and_Alternatives_50_Ticker_Roster_Updated_Long_Short_Positions_2026_to_2030.csv`

**Important notes / limitations:**
- Some tickers in the roster may not have history back to 2000–2002; historical backtests renormalize weights to the subset with data in each window.
- Shorting, leverage, and derivatives/futures have financing/margin/roll costs not modeled here. This is a research notebook, not execution-ready.
- This is **not financial advice**.
"""

    run_md = """## How to run

1. Install dependencies (recommended from terminal):
   - `python3 -m pip install -r requirements.txt`
2. Open `Bubble_Resilient_Portfolio.ipynb` and **Run All**.

All key knobs (ticker universe, constraints, stress windows, bubble tickers) are in **Section 0 (Setup)** + **Section 1 (Configuration)**.
"""

    setup_code = """# 0) Setup: imports + plotting defaults

import importlib
import subprocess
import sys

REQUIRED = [
    'yfinance',
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'scipy',
    'statsmodels',
    'requests',
]

missing = []
for pkg in REQUIRED:
    try:
        importlib.import_module(pkg)
    except Exception:
        missing.append(pkg)

if missing:
    print('Installing missing packages:', missing)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + missing)
else:
    print('All required packages already installed.')

import numpy as np
import pandas as pd
import yfinance as yf
from IPython.display import display, Markdown

# Output/run controls (keep Section 12 user-facing by default)
# - Default: ONE simple early-warning table for the 3 bubbles.
# - Turn on flags below if you want the deeper research tables / validation / forecast outputs.
SHOW_ONLY_SIMPLE_EWS_TABLE = True
# News/headlines (Section 12): optional. Disabled by default to keep the notebook output simple.
RUN_SENTIMENT = False
RUN_YAHOO_HEADLINES = False
RUN_SEC_FILINGS_NLP = False
RUN_TRANSCRIPTS_NLP = False
RUN_DETAILED_VALIDATION = False
RUN_FORECAST = False
INCLUDE_2026_2030_PROJECTION = True
ENABLE_EMAIL_ALERTS = False
ENABLE_SLACK_ALERTS = False
ENABLE_EWS_AUDIT_LOG = True
ENABLE_REBALANCE_AUDIT_LOG = True

# Explanation helper (keeps the notebook interpretable even as live Yahoo Finance data changes)
SHOW_EXPLANATIONS = True


def explain(title: str, bullets: list[str]) -> None:
    if not bool(SHOW_EXPLANATIONS):
        return
    body = "\n".join([f"- {b}" for b in bullets if b])
    md = f"### What this means now: {title}\n{body}" if body else f"### What this means now: {title}"
    try:
        display(Markdown(md))
    except Exception:
        # Fallback when Markdown display isn't available
        print("\n[Interpretation] " + title)
        if body:
            print(body)

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import matplotlib.pyplot as plt
from IPython.display import display
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import requests

pd.set_option('display.max_columns', 80)
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Bubble ticker sets (single source of truth across the notebook)
# Policy: use ONLY the user-approved Yahoo Finance symbols.
BUBBLE_TICKERS = {
    'AI bubble': ['QQQ', 'ROBT', 'U8S1.SG', 'ESIFF', 'WTAI', 'NMX101010.FGI'],
    'Private Credit bubble': ['HYG', 'HYGU.L', 'HYIN', 'TAKMX', 'VPC'],
    'Crypto bubble': ['BTC-USD', '^SPCMCFUE'],
}
"""

    market_md = """## 0.5) Live market data update check (before Section 1)

This notebook pulls prices **live** from Yahoo Finance via `yfinance`.

Before we configure tickers/constraints, we run a quick **live-data check** so you can immediately see whether the latest Yahoo pull likely changed the rest of the notebook results.

We use a small watchlist:
We use a watchlist aligned to the bubble indicator tickers:
- **AI bubble:** QQQ, ROBT, U8S1.SG, ESIFF, WTAI, NMX101010.FGI
- **Private Credit bubble:** HYG, HYGU.L, HYIN, TAKMX, VPC
- **Crypto bubble:** BTC-USD, ^SPCMCFUE
"""

    market_code = """# 0.5) Live market data update check (intraday-aware)

from zoneinfo import ZoneInfo
import json
import os

if 'BUBBLE_TICKERS' not in globals() or not isinstance(BUBBLE_TICKERS, dict) or not BUBBLE_TICKERS:
    raise RuntimeError('[market] BUBBLE_TICKERS missing/empty; run Section 0 setup cell first.')

# Watchlist aligned to the bubble indicator tickers (single source of truth)
WATCH = {k: list(v) for k, v in BUBBLE_TICKERS.items()}

tz = ZoneInfo('America/New_York')

now = pd.Timestamp.now(tz)
today = now.normalize()
is_weekend = now.dayofweek >= 5  # 5=Sat, 6=Sun

tickers = sorted({t for ts in WATCH.values() for t in ts})

def _to_close_df(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        field = 'Close' if ('Close' in raw.columns.get_level_values(0)) else ('Adj Close' if ('Adj Close' in raw.columns.get_level_values(0)) else None)
        if field is None:
            return pd.DataFrame()
        close = raw[field].copy()
    else:
        field = 'Close' if ('Close' in raw.columns) else ('Adj Close' if ('Adj Close' in raw.columns) else None)
        if field is None:
            return pd.DataFrame()
        if len(tickers) == 1:
            close = raw[[field]].rename(columns={field: tickers[0]}).copy()
        else:
            return pd.DataFrame()
    close.index = pd.to_datetime(close.index)
    return close

daily_raw = yf.download(
    tickers,
    period='14d',
    interval='1d',
    auto_adjust=True,
    progress=False,
    group_by='column',
    threads=True,
)
daily = _to_close_df(daily_raw, tickers)
daily = daily.sort_index().dropna(how='all')

intraday_raw = yf.download(
    tickers,
    period='5d',
    interval='60m',
    auto_adjust=True,
    progress=False,
    group_by='column',
    threads=True,
)
intraday = _to_close_df(intraday_raw, tickers)
if not intraday.empty:
    intraday_idx = pd.to_datetime(intraday.index)
    if intraday_idx.tz is None:
        intraday.index = intraday_idx.tz_localize('UTC').tz_convert(tz)
    else:
        intraday.index = intraday_idx.tz_convert(tz)

rows = []
if not daily.empty:
    daily = daily.sort_index().dropna(how='all')
    for bubble_name, t_list in WATCH.items():
        for t in t_list:
            s = daily[t].dropna() if t in daily.columns else pd.Series(dtype=float)
            if len(s) == 0:
                rows.append({'bubble': bubble_name, 'ticker': t, 'last_date': None, 'last_close': None, 'day_change_%': None})
                continue
            last_date = pd.to_datetime(s.index[-1]).date()
            last_close = float(s.iloc[-1])
            prev_close = float(s.iloc[-2]) if len(s) >= 2 else None
            day_chg = ((last_close / prev_close) - 1.0) * 100.0 if prev_close and prev_close != 0 else None
            rows.append({'bubble': bubble_name, 'ticker': t, 'last_date': str(last_date), 'last_close': last_close, 'day_change_%': day_chg})

live_tbl = pd.DataFrame(rows)

def _last_intraday(ts: pd.DataFrame, ticker: str):
    if ts is None or ts.empty or ticker not in ts.columns:
        return None, None
    s = ts[ticker].dropna()
    if len(s) == 0:
        return None, None
    return s.index[-1], float(s.iloc[-1])

snap_path = '.yahoo_last_watch_snapshot.json'
prev = None
if os.path.exists(snap_path):
    try:
        with open(snap_path, 'r', encoding='utf-8') as f:
            prev = json.load(f)
    except Exception:
        prev = None

latest = {r['ticker']: r for r in rows}
try:
    with open(snap_path, 'w', encoding='utf-8') as f:
        json.dump(latest, f, ensure_ascii=False, indent=2)
except Exception:
    pass

changed = False
if prev:
    for t, entry in latest.items():
        if t not in prev:
            changed = True
            break
        prev_last = prev[t].get('last_close')
        cur_last = entry.get('last_close')
        if (prev_last is not None) and (cur_last is not None):
            if abs(cur_last - prev_last) / max(prev_last, 1e-9) > 1e-4:
                changed = True
                break

print("\n[market] Live Yahoo Finance update check (intraday-aware)")
print('now (NY):', now)
qqq_date = live_tbl.loc[live_tbl['ticker'] == 'QQQ', 'last_date'].iloc[0] if 'QQQ' in live_tbl['ticker'].values else None
hyg_date = live_tbl.loc[live_tbl['ticker'] == 'HYG', 'last_date'].iloc[0] if 'HYG' in live_tbl['ticker'].values else None
print('US equities latest close date (QQQ/HYG):', qqq_date or 'N/A', hyg_date or 'N/A')

if not intraday.empty:
    print("\n[intraday] last bar (NY time):")
    for bubble_name, t_list in WATCH.items():
        for t in t_list:
            ts, px = _last_intraday(intraday, t)
            if ts is None:
                print(f"- {t} ({bubble_name}): no intraday data")
                continue
            age_minutes = (now - ts).total_seconds() / 60.0
            print(f"- {t} ({bubble_name}): {ts}  px={px:.2f}  age={age_minutes:.1f} min")
else:
    print("\n[intraday] no intraday data returned")

if len(live_tbl):
    live_tbl = live_tbl.sort_values('ticker')
    max_move = live_tbl['day_change_%'].abs().max() if live_tbl['day_change_%'].notna().any() else None
    print('max abs 1-day move (watchlist):', f"{max_move:.2f}%" if max_move is not None else 'n/a')

print('Live data changed since last run?', bool(changed))

# Dynamic interpretation (changes with live Yahoo Finance pulls)
if 'explain' in globals():
    try:
        explain(
            '0.5) Live market data update check',
            [
                f"Snapshot time (NY): {now}",
                f"Live data changed vs last snapshot: {bool(changed)}",
                'If this is True, rerun the notebook (Sections 1–13) to refresh all downstream tables, backtests, and recommendations.',
                'If this is False, results should be close to your previous run (small differences can still happen due to intraday bars / Yahoo data revisions).',
            ],
        )
    except Exception:
        pass
    """

    config_md = """## 1) Configuration (tickers, bubbles, stress windows, constraints)

This is where you tune:
- The **roster-driven investable universe** (tickers + metadata + LONG/SHORT positioning)
- Bubble tickers (**AI bubble / Private Credit bubble / Crypto bubble**)
- Stress/backtest windows (Dot-com, GFC, COVID)
- Portfolio constraints (gross exposure, max position size)
- Roster CSV path (edit the CSV and the notebook picks up the new symbols dynamically)

### Simple Explanation of the Codebase, Output, how it's supporting the goal of resilience vs the AI / Private Credit / Crypto bubbles
- This is the **control panel** of the notebook.
- It loads your roster CSV (tickers + Long/Short + sizing + rationale) and defines which tickers represent each bubble.
- Everything downstream (tables, correlations, backtests, stress tests) uses these settings, so if you change the CSV you’re changing the portfolio “source of truth”.
"""

    config_code = """import pandas as pd
import numpy as np

from pandas.tseries.frequencies import to_offset

# pandas 3.x removed the 'M' (month-end) alias for offsets; 'ME' is the replacement.
# For compatibility across pandas versions, pick the best available month-end alias.
try:
    to_offset('ME')
    MONTHLY_FREQ = 'ME'
except Exception:
    MONTHLY_FREQ = 'M'

TODAY = pd.Timestamp.today().normalize()
HORIZON_START = pd.Timestamp('2026-01-01')
HORIZON_END = pd.Timestamp('2030-12-31')
FORECAST_START = max(TODAY, HORIZON_START)

print('Today:', TODAY.date())
print('Forecast horizon:', FORECAST_START.date(), '→', HORIZON_END.date())

# Historical macro stress windows (required): dot-com, GFC, COVID
EVENT_WINDOWS = {
    'Dot-com crash (AI bubble stress)': ('2000-03-10', '2002-10-09'),
    'GFC (Private Credit bubble stress)': ('2007-10-09', '2009-03-09'),
    'COVID crash': ('2020-02-19', '2020-03-23'),
    # Optional extra window to explicitly stress the Crypto bubble
    'Crypto winter (Crypto bubble stress)': ('2021-11-10', '2022-11-09'),
}

# Bubble ticker sets (single source of truth)
# NOTE: defined in Section 0 (Setup) so Section 0.5 watch + Section 12 EWS + downstream factor models stay aligned.
if 'BUBBLE_TICKERS' not in globals() or not isinstance(BUBBLE_TICKERS, dict) or not BUBBLE_TICKERS:
    raise RuntimeError('BUBBLE_TICKERS missing/empty; run Section 0 setup cell first.')

# Roster CSV (source of truth for investable universe + positioning guidance)
ROSTER_PATH = 'FICC_and_Alternatives_50_Ticker_Roster_Updated_Long_Short_Positions_2026_to_2030.csv'
roster = pd.read_csv(ROSTER_PATH)

REQUIRED_ROSTER_COLS = ['Ticker', 'Name', 'AssetClass', 'RiskLimitPercent', 'Position', 'Rationale']
missing_cols = [c for c in REQUIRED_ROSTER_COLS if c not in roster.columns]
if missing_cols:
    raise ValueError(f'Roster file is missing required columns: {missing_cols}')

roster = roster.copy()
roster['Ticker'] = roster['Ticker'].astype(str).str.strip()
roster['Position'] = roster['Position'].astype(str).str.strip().str.upper()
roster['RiskLimitPercent'] = pd.to_numeric(roster['RiskLimitPercent'], errors='coerce')
roster = roster[roster['Ticker'].ne('')].dropna(subset=['Ticker']).drop_duplicates(subset=['Ticker'])

# Universe tickers (dynamic: edit the CSV and the notebook updates automatically)
universe_tickers = roster['Ticker'].tolist()

# Convenience maps for later sections
name_by_ticker = roster.set_index('Ticker')['Name'].to_dict()
assetclass_by_ticker = roster.set_index('Ticker')['AssetClass'].to_dict()
position_by_ticker = roster.set_index('Ticker')['Position'].to_dict()
risklimit_by_ticker = roster.set_index('Ticker')['RiskLimitPercent'].to_dict()
rationale_by_ticker = roster.set_index('Ticker')['Rationale'].to_dict()

print('Roster rows:', len(roster))
print('Universe tickers:', len(universe_tickers))
roster.head(10)

# Portfolio constraints
GROSS_EXPOSURE = 1.00          # sum(abs(weights)) == 100% gross exposure
MAX_ABS_WEIGHT = 0.20          # fallback per-position cap (abs) for sampled portfolios
RANDOM_SEED = 42

# Optimization windows: use required ones by default (plus optional crypto window if you want)
OPTIMIZATION_WINDOWS = [
    'Dot-com crash (AI bubble stress)',
    'GFC (Private Credit bubble stress)',
    'COVID crash',
    # 'Crypto winter (Crypto bubble stress)',
]

# Dynamic interpretation (updates when roster CSV or knobs change)
if 'explain' in globals():
    try:
        bullets = []
        long_n = int((roster['Position'] == 'LONG').sum())
        short_n = int((roster['Position'] == 'SHORT').sum())

        bullets.append(f'Roster symbols: {len(roster)} ({long_n} LONG, {short_n} SHORT)')
        bullets.append(f'Gross exposure target: {GROSS_EXPOSURE:.0%} (sum abs(weights))')
        bullets.append(f'Sampling cap: MAX_ABS_WEIGHT={MAX_ABS_WEIGHT:.0%} per position (abs)')

        bubbles = '; '.join([f"{k}=[{', '.join(v)}]" for k, v in BUBBLE_TICKERS.items()])
        bullets.append(f'Bubble proxies used in Sections 4/10: {bubbles}')

        top = roster[['Ticker', 'Position', 'RiskLimitPercent']].copy()
        top['RiskLimitPercent'] = pd.to_numeric(top['RiskLimitPercent'], errors='coerce').fillna(0.0)
        top = top.sort_values('RiskLimitPercent', ascending=False).head(5)
        if len(top):
            top_txt = '; '.join(
                [f'{r["Ticker"]} {r["Position"]} {float(r["RiskLimitPercent"]):.1f}%' for _i, r in top.iterrows()]
            )
            bullets.append('Top RiskLimitPercent budgets: ' + top_txt)

        explain('1) Configuration snapshot', bullets)
    except Exception:
        pass
"""

    data_md = """## 2) Download live Yahoo Finance data (yfinance)

We download daily prices and align them to a **business-day** calendar.

### Simple Explanation of the Codebase, Output, how it's supporting the goal of resilience vs the AI / Private Credit / Crypto bubbles
- This section pulls **live price history** from Yahoo Finance for both (a) your portfolio tickers and (b) the bubble proxy tickers.
- If a symbol has no data, we flag it and drop it later.
- This is why the notebook is “live” and non-static: every run refreshes data.
"""

    data_code = """def download_close_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    import time

    # Robustness: large multi-ticker downloads can return partial/empty results; chunk + retry.
    tickers = list(dict.fromkeys([t for t in tickers if isinstance(t, str) and t.strip()]))
    if not tickers:
        raise ValueError('No tickers provided.')

    BATCH_SIZE = 50
    MAX_RETRIES = 3

    def _download(batch: list[str]) -> pd.DataFrame:
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by='column',
            threads=True,
        )
        if raw is None or len(raw) == 0:
            return pd.DataFrame()

        # yfinance returns MultiIndex columns for multi-ticker downloads, flat columns for a single ticker.
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0)
            field = 'Close' if ('Close' in lvl0) else ('Adj Close' if ('Adj Close' in lvl0) else None)
            if field is None:
                return pd.DataFrame()
            close = raw[field].copy()
        else:
            if len(batch) != 1:
                return pd.DataFrame()
            field = 'Close' if ('Close' in raw.columns) else ('Adj Close' if ('Adj Close' in raw.columns) else None)
            if field is None:
                return pd.DataFrame()
            close = raw[[field]].rename(columns={field: batch[0]}).copy()

        close.index = pd.to_datetime(close.index)
        close = close.sort_index()

        # Ensure every requested ticker exists as a column (yfinance may silently drop unknown symbols)
        for t in batch:
            if t not in close.columns:
                close[t] = np.nan

        return close[batch]

    chunks: list[pd.DataFrame] = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        out = pd.DataFrame()
        for attempt in range(MAX_RETRIES):
            try:
                out = _download(batch)
                if not out.empty:
                    break
            except Exception:
                out = pd.DataFrame()
            time.sleep(0.75 * (attempt + 1))

        if out.empty:
            print(f"[yfinance] warning: empty batch result ({len(batch)} tickers); will retry individually where needed")
        else:
            chunks.append(out)

    if not chunks:
        raise ValueError('No data returned from yfinance. Check network / ticker symbols.')

    close = pd.concat(chunks, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]

    # Retry tickers that are missing or all-NaN, one-by-one (often fixes transient Yahoo issues).
    retry = [t for t in tickers if (t not in close.columns) or (not close[t].notna().any())]
    retry = sorted(set(retry))
    if retry:
        print(f"[yfinance] retrying {len(retry)} tickers individually...")
        for t in retry:
            out = pd.DataFrame()
            for attempt in range(MAX_RETRIES):
                try:
                    out = _download([t])
                    if not out.empty and (t in out.columns):
                        break
                except Exception:
                    out = pd.DataFrame()
                time.sleep(0.75 * (attempt + 1))
            if not out.empty and (t in out.columns):
                close[t] = out[t]

    close = close.sort_index()

    # align to business days; forward-fill non-trading days
    close = close.resample('B').last().ffill()
    return close

# Pull enough history to cover dot-com → present
HIST_START = '1999-01-01'

# Investable universe tickers come from the roster CSV (dynamic)
ficc_tickers = universe_tickers.copy()
# Underlying bubble tickers (flattened). Bubble *factors* are computed later as composites.
bubble_tickers = sorted({t for ts in BUBBLE_TICKERS.values() for t in ts})

ALL_TICKERS = sorted(set(ficc_tickers + bubble_tickers))

prices = download_close_prices(ALL_TICKERS, start=HIST_START)

available = [c for c in prices.columns if prices[c].notna().any()]
missing = sorted(set(ALL_TICKERS) - set(available))

print('Downloaded columns:', len(prices.columns))
print('Available tickers:', len(available))
if missing:
    print('Tickers with no data (dropped later):', missing)

prices = prices[available]

# Dynamic interpretation (updates with live Yahoo Finance pulls)
if 'explain' in globals():
    try:
        start_dt = prices.index.min().date() if (prices is not None and not prices.empty) else None
        end_dt = prices.index.max().date() if (prices is not None and not prices.empty) else None

        miss_preview = ''
        if missing:
            miss_preview = ', '.join(missing[:10])
            if len(missing) > 10:
                miss_preview = miss_preview + ', ...'

        explain(
            '2) Data download coverage',
            [
                f'Tickers requested: {len(ALL_TICKERS)}',
                f'Tickers with data: {len(available)}',
                f'Tickers missing (no Yahoo data): {len(missing)}' + (f' [{miss_preview}]' if miss_preview else ''),
                f'Price date range (business days, forward-filled): {start_dt} → {end_dt}',
                'If you change the roster allocation/weights or re-run on a new trading day, downstream tables/backtests will update automatically from live Yahoo prices.',
            ],
        )
    except Exception:
        pass

prices.tail()
"""

    returns_md = """## 3) Compute returns + baseline stats

We compute **simple (percentage) returns** only (no log returns) and report key metrics in **%** terms.

### Simple Explanation of the Codebase, Output, how it's supporting the goal of resilience vs the AI / Private Credit / Crypto bubbles
- We convert prices into **% returns** so we can compare assets consistently.
- The baseline stats (annualized return/vol) are a sanity-check: are the tickers behaving roughly as expected?
- These return series are the inputs used for correlations, portfolio aggregation, backtests, and forecasting.
"""

    returns_code = """rets = prices.pct_change().dropna(how='all')

# Bubble factor composites (domain-level, built from the full ticker sets in BUBBLE_TICKERS)
# - We normalize each ticker's price to 1.0 at its first available observation, then average.
# - Returns are the pct_change of that composite index.
def bubble_composite_prices(prices: pd.DataFrame, bubble_map: dict[str, list[str]]) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    for bubble, tickers in (bubble_map or {}).items():
        cols = [t for t in (tickers or []) if t in prices.columns]
        if not cols:
            continue

        norm_series = []
        for t in cols:
            s = prices[t].dropna()
            if s.empty:
                continue
            base = float(s.iloc[0])
            norm_series.append((prices[t] / base).rename(t))

        if not norm_series:
            continue
        out[bubble] = pd.concat(norm_series, axis=1).mean(axis=1, skipna=True)
    return out


bubble_factor_prices = bubble_composite_prices(prices, BUBBLE_TICKERS)
bubble_factor_returns = bubble_factor_prices.pct_change().dropna(how='all')
print('Bubble factor composites available:', list(bubble_factor_returns.columns))

from functools import lru_cache

@lru_cache(maxsize=None)
def yahoo_name(symbol: str) -> str:
    # Best-effort: pull a human-readable name from Yahoo Finance metadata.
    try:
        info = yf.Ticker(symbol).info or {}
        return info.get('shortName') or info.get('longName') or info.get('name') or ''
    except Exception:
        return ''

symbol_name_table = pd.DataFrame(
    {'name': [yahoo_name(s) for s in prices.columns]},
    index=pd.Index(prices.columns, name='symbol'),
).sort_index()

print('Symbol + Name (sample):')
pd.set_option('display.max_rows', 60)

print('Roster tickers (all rows):')
roster[['Ticker', 'Name', 'AssetClass', 'Position', 'RiskLimitPercent']].sort_values('RiskLimitPercent', ascending=False)

def ann_return_from_daily(daily: pd.Series) -> float:
    daily = daily.dropna()
    if len(daily) == 0:
        return np.nan
    return (1 + daily).prod() ** (252 / len(daily)) - 1

def ann_vol_from_daily(daily: pd.Series) -> float:
    daily = daily.dropna()
    if len(daily) == 0:
        return np.nan
    return float(daily.std() * np.sqrt(252))

stats_table = pd.DataFrame({
    'name': [yahoo_name(s) for s in rets.columns],
    'ann_return_geo': rets.apply(ann_return_from_daily, axis=0),
    'ann_vol': rets.apply(ann_vol_from_daily, axis=0),
    'start': prices.apply(lambda s: s.first_valid_index()).astype('datetime64[ns]'),
}).sort_values('ann_return_geo', ascending=False)

stats_roster = stats_table.reindex(universe_tickers)

# Dynamic interpretation (updates with live Yahoo Finance pulls)
if 'explain' in globals():
    try:
        bullets = []
        bullets.append(f'Daily return rows: {len(rets)} (business days)')
        bullets.append(f'Tickers with data in this run: {len(rets.columns)}')

        s_ret = pd.to_numeric(stats_roster['ann_return_geo'], errors='coerce').dropna()
        if len(s_ret):
            top_ret = s_ret.sort_values(ascending=False).head(3)
            bullets.append(
                'Top roster annualized returns (geo): ' + ', '.join([f'{k} {v:.2%}' for k, v in top_ret.items()])
            )

        s_vol = pd.to_numeric(stats_roster['ann_vol'], errors='coerce').dropna()
        if len(s_vol):
            top_vol = s_vol.sort_values(ascending=False).head(3)
            bullets.append(
                'Most volatile roster tickers (ann vol): ' + ', '.join([f'{k} {v:.2%}' for k, v in top_vol.items()])
            )

        explain('3) Returns + baseline stats', bullets)
    except Exception:
        pass

stats_roster.style.format({'ann_return_geo': '{:.2%}', 'ann_vol': '{:.2%}'})
"""

    corr_md = """## 4) Assets vs Bubble correlation matrix (LONG/SHORT rule)

Rule (as requested):
- If an asset has **positive correlation** with the bubble-burst series ⇒ **SHORT**
- If an asset has **negative correlation** with the bubble-burst series ⇒ **LONG**

We compute correlations by stress window and average them. The **Position (Long/Short)** column shown is taken directly from your roster CSV.

Additional visuals (requested):
- Add the roster **Rationale** column so readers understand why each symbol is included.
- Show an **asset-to-asset correlation matrix** (allocated assets vs allocated assets) to assess cross-asset diversification/hedging.
- Show two **pie charts** of gross allocation by **AssetClass** and by **Position** (Long vs Short).

### Simple Explanation of the Codebase, Output, how it's supporting the goal of resilience vs the AI / Private Credit / Crypto bubbles
- This section answers: **“Which assets tend to move with each bubble proxy during stress windows?”**
- If something is positively correlated with a bubble proxy in stress periods, it may amplify drawdowns; negatively correlated assets can act as hedges.
- The cross-asset correlation matrix + allocation pies help verify the portfolio is **diversified/hedged** (not just a cluster of similar exposures).
"""

    corr_code = """from IPython.display import display

candidate_tickers = [t for t in ficc_tickers if t in prices.columns]

if 'bubble_factor_returns' not in globals() or bubble_factor_returns is None or bubble_factor_returns.empty:
    raise RuntimeError('bubble_factor_returns missing/empty; run Section 3 first.')

# Use domain-level composite bubble factor returns (3 columns: AI/Private Credit/Crypto)
bubble_returns = bubble_factor_returns.reindex(rets.index).dropna(how='all')

print('Candidate universe tickers available:', len(candidate_tickers))
print('Bubbles used:', list(bubble_returns.columns))

def corr_assets_vs_bubbles(asset_returns: pd.DataFrame, bubble_returns: pd.DataFrame) -> pd.DataFrame:
    bubbles = bubble_returns.rename(columns=lambda c: f'__bubble__{c}')
    common = asset_returns.join(bubbles, how='inner')
    out = pd.DataFrame(index=asset_returns.columns, columns=bubble_returns.columns, dtype=float)
    for b in bubble_returns.columns:
        out[b] = common[asset_returns.columns].corrwith(common[f'__bubble__{b}'])
    return out

corr_by_window: dict[str, pd.DataFrame] = {}

for name, (start, end) in EVENT_WINDOWS.items():
    a = rets.loc[start:end, candidate_tickers].dropna(how='all')
    b = bubble_returns.loc[start:end].dropna(how='all')
    if a.empty or b.empty:
        print(f'[skip] {name}: insufficient overlap for correlation')
        continue
    corr_by_window[name] = corr_assets_vs_bubbles(a, b)

if not corr_by_window:
    raise RuntimeError('No correlation windows produced any data. Check tickers and date ranges.')

# Average correlation across selected windows
corr_avg = pd.concat(corr_by_window, names=['window', 'asset']).groupby('asset').mean()

plt.figure(figsize=(10, max(6, 0.25 * len(corr_avg))))
sns.heatmap(corr_avg, annot=False, cmap='coolwarm', center=0)
plt.title('Assets vs Bubbles (avg correlation across stress windows)')
plt.tight_layout()
plt.show()

# Display correlation matrix along with the roster Position (source of truth)
corr_with_side = corr_avg.copy()
corr_with_side.insert(0, 'name', [yahoo_name(s) for s in corr_with_side.index])
corr_with_side['asset_class'] = [assetclass_by_ticker.get(s, '') for s in corr_with_side.index]
corr_with_side['risk_limit_%'] = [risklimit_by_ticker.get(s, np.nan) for s in corr_with_side.index]
corr_with_side['position'] = [position_by_ticker.get(s, '') for s in corr_with_side.index]

# Add roster Rationale column (user requested)
corr_with_side['rationale'] = [rationale_by_ticker.get(s, '') for s in corr_with_side.index]

display(
    corr_with_side.sort_values(['position', 'risk_limit_%', 'name'], ascending=[True, False, True]).style.format(
        {
            'risk_limit_%': '{:.2f}',
            'AI bubble': '{:.2f}',
            'Private Credit bubble': '{:.2f}',
            'Crypto bubble': '{:.2f}',
        }
    )
)

# Dynamic interpretation (updates with live Yahoo Finance pulls)
if 'explain' in globals():
    try:
        bullets = []
        for bubble in ['AI bubble', 'Private Credit bubble', 'Crypto bubble']:
            if bubble not in corr_with_side.columns:
                continue
            s = pd.to_numeric(corr_with_side[bubble], errors='coerce').dropna()
            if s.empty:
                continue

            hedges = s.sort_values().head(3)
            expos = s.sort_values(ascending=False).head(3)

            hedges_txt = ', '.join([f'{k} ({v:+.2f})' for k, v in hedges.items()])
            expos_txt = ', '.join([f'{k} ({v:+.2f})' for k, v in expos.items()])

            bullets.append(f'{bubble}: most negative corr (hedges) → {hedges_txt}')
            bullets.append(f'{bubble}: most positive corr (exposures) → {expos_txt}')

        if bullets:
            bullets.append('Rule of thumb: negative correlation vs a bubble proxy can hedge that bubble-burst; positive correlation can amplify it (correlations can shift with regimes).')
            explain('4) Correlations: hedges vs exposures', bullets)
    except Exception:
        pass

# ------------------------------------------------------------
# Cross-asset correlation matrix (allocated assets vs assets)
# ------------------------------------------------------------
alloc_tickers = [t for t in universe_tickers if t in prices.columns]
alloc_rets = rets[alloc_tickers].fillna(0.0)

asset_corr = alloc_rets.corr()

print()
print('Asset-to-asset correlation matrix (allocated assets):')
display(asset_corr.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format('{:.2f}'))

# ------------------------------------------------------------
# Allocation pie charts (by AssetClass and by Position)
# ------------------------------------------------------------
# Build roster-driven gross weights (same sizing logic as Section 5)
roster_sub = roster.set_index('Ticker').reindex(alloc_tickers)

abs_budget = roster_sub['RiskLimitPercent'].astype(float).fillna(0.0).clip(lower=0.0)
if float(abs_budget.sum()) <= 0:
    raise RuntimeError('Roster RiskLimitPercent sums to 0; cannot build allocation pies.')

abs_w = abs_budget / float(abs_budget.sum())
pos = roster_sub['Position'].astype(str).str.upper().fillna('LONG')
sign = pos.map({'LONG': 1.0, 'SHORT': -1.0}).fillna(1.0)

w = abs_w * sign
w = w * (GROSS_EXPOSURE / w.abs().sum())
w_abs = w.abs()

# Pie 1: AssetClass
ac = roster_sub['AssetClass'].astype(str).fillna('Unknown')
by_class = w_abs.groupby(ac).sum().sort_values(ascending=False)

def pie_with_legend_percent(series: pd.Series, title: str, legend_title: str, figsize: tuple[int, int]) -> None:
    vals = series.values
    labels = series.index.astype(str)
    pct = (vals / vals.sum()) * 100.0
    legend_labels = [f"{lab}: {p:.1f}%" for lab, p in zip(labels, pct)]

    plt.figure(figsize=figsize)
    wedges, _ = plt.pie(vals, labels=None, startangle=90)
    plt.title(title)
    plt.legend(wedges, legend_labels, title=legend_title, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

pie_with_legend_percent(
    by_class,
    title='Gross allocation by AssetClass (%)',
    legend_title='AssetClass',
    figsize=(9, 7),
)

# Pie 2: Position (Long vs Short)
by_pos = w_abs.groupby(pos).sum().sort_values(ascending=False)

pie_with_legend_percent(
    by_pos,
    title='Gross allocation by Position (%)',
    legend_title='Position',
    figsize=(7, 6),
)
"""

    portfolio_md = """## 5) Portfolio construction (roster-driven long/short allocation)

We build one portfolio allocation from the roster CSV:
- **Position (Long/Short)** comes from the `Position` column.
- **Sizing** is based on `RiskLimitPercent` (normalized to 100% gross exposure).

**Stress-window upgrade (optional):** This notebook can also *override* the roster sizing with a stress-window optimized weight vector that tries to maximize the **worst-case cumulative return** across the three historical stress windows used later in Section 7 (Dot-com, GFC, COVID). It keeps LONG/SHORT directions fixed to the roster, and only changes *how much* you allocate to each asset.

This produces the single required allocation list: **Symbol, Symbol name, Weight (%), Long/Short, and reasoning**.

### Simple Explanation of the Codebase, Output, how it's supporting the goal of resilience vs the AI / Private Credit / Crypto bubbles
- This is where we turn the roster CSV into an actual **portfolio weight vector**.
- `RiskLimitPercent` controls sizing; `Position` controls whether that weight is LONG or SHORT.
- The output allocation table is the “portfolio recipe” used everywhere else (backtests, simulations, stress tests).
"""

    portfolio_code = """def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    # Fill missing daily returns as 0 (no price change); avoids dropping entire rows
    aligned = asset_returns.reindex(columns=weights.index).fillna(0.0)
    return aligned.dot(weights)

def cum_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1 + r).prod() - 1)

def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return float(dd.min())
# Investable tickers with price data
asset_tickers = [t for t in universe_tickers if t in prices.columns]
print('Investable tickers (with price data):', len(asset_tickers))

roster_sub = roster.set_index('Ticker').reindex(asset_tickers)

# Build weights from roster RiskLimitPercent (normalized) and Position (Long/Short)
abs_budget = roster_sub['RiskLimitPercent'].astype(float).fillna(0.0).clip(lower=0.0)
if float(abs_budget.sum()) <= 0:
    raise RuntimeError('Roster RiskLimitPercent sums to 0; cannot build a portfolio.')

abs_w = abs_budget / float(abs_budget.sum())
pos = roster_sub['Position'].astype(str).str.upper().fillna('LONG')
sign = pos.map({'LONG': 1.0, 'SHORT': -1.0}).fillna(1.0)
signs = sign.values

base_w = (abs_w * sign).rename('weight')
base_w = base_w * (GROSS_EXPOSURE / base_w.abs().sum())

# ------------------------------------------------------------
# Stress-window optimized sizing (keeps roster LONG/SHORT signs)
# ------------------------------------------------------------
USE_STRESS_OPTIMIZED_WEIGHTS = True
STRESS_OPT_WINDOWS = [
    'Dot-com crash (AI bubble stress)',
    'GFC (Private Credit bubble stress)',
    'COVID crash',
]
INCLUDE_SECTION10_SCENARIOS = True   # bubble-burst forward-looking scenarios (Section 10)
INCLUDE_GPFG_SCENARIOS = True        # GPFG-style shocks (Section 10B)

N_OPT_CANDIDATES = 8000
DIRICHLET_ALPHA = 2.0  # >1 makes weights less extreme; lower values concentrate
SHOCK_MONTHS = 12


def _cum_return_from_daily_fast(daily: np.ndarray) -> float:
    daily = np.asarray(daily, dtype=float)
    if daily.size == 0:
        return np.nan
    # avoid log(<=0)
    daily = np.clip(daily, -0.999999, None)
    return float(np.expm1(np.log1p(daily).sum()))


def _cum_return_from_monthly_fast(monthly: np.ndarray) -> float:
    monthly = np.asarray(monthly, dtype=float)
    if monthly.size == 0:
        return np.nan
    monthly = np.clip(monthly, -0.999999, None)
    return float(np.expm1(np.log1p(monthly).sum()))


def _stress_cum_returns(weights: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for nm in STRESS_OPT_WINDOWS:
        if nm not in EVENT_WINDOWS:
            out[nm] = np.nan
            continue
        start, end = EVENT_WINDOWS[nm]
        window = rets.loc[start:end, asset_tickers]
        cols = [c for c in window.columns if window[c].notna().any()]
        if not cols:
            out[nm] = np.nan
            continue
        w = weights.reindex(cols).fillna(0.0)
        if float(w.abs().sum()) <= 0:
            out[nm] = np.nan
            continue
        w = w * (GROSS_EXPOSURE / float(w.abs().sum()))
        pr = window[cols].fillna(0.0).dot(w).to_numpy()
        out[nm] = _cum_return_from_daily_fast(pr)
    return out


best_w = base_w.copy()
if USE_STRESS_OPTIMIZED_WEIGHTS and len(asset_tickers) >= 3:
    n = len(asset_tickers)
    rng_opt = np.random.default_rng(RANDOM_SEED + 77)

    # Candidate absolute weights (each column sums to 1)
    A = rng_opt.dirichlet(np.ones(n) * float(DIRICHLET_ALPHA), size=int(N_OPT_CANDIDATES)).T

    # Enforce per-asset cap in the *global* portfolio (not window-renormalized)
    cap = float(MAX_ABS_WEIGHT)
    mask = (A.max(axis=0) <= cap)
    if mask.sum() >= max(1000, int(0.25 * N_OPT_CANDIDATES)):
        A = A[:, mask]

    if A.shape[1] >= 500:
        win_cum = []
        win_names: list[str] = []
        for nm in STRESS_OPT_WINDOWS:
            if nm not in EVENT_WINDOWS:
                continue
            start, end = EVENT_WINDOWS[nm]
            window = rets.loc[start:end, asset_tickers]
            cols = [c for c in window.columns if window[c].notna().any()]
            if not cols:
                continue

            idx = [asset_tickers.index(c) for c in cols]
            A_sub = A[idx, :]
            denom = A_sub.sum(axis=0)
            denom = np.where(denom <= 0, np.nan, denom)

            sign_sub = sign.reindex(cols).fillna(1.0).to_numpy(dtype=float)
            W_sub = (sign_sub[:, None] * A_sub) / denom[None, :] * float(GROSS_EXPOSURE)

            R = window[cols].fillna(0.0).to_numpy(dtype=float)
            pr = R @ W_sub
            pr = np.clip(pr, -0.999999, None)
            cum = np.expm1(np.log1p(pr).sum(axis=0))
            win_cum.append(cum)
            win_names.append(nm)

        # -----------------------------
        # Forward-looking factor scenarios (Section 10 + 10B)
        # -----------------------------
        scenario_paths: dict[str, pd.DataFrame] = {}
        bubble_factors: list[str] = []
        if 'bubble_factor_returns' in globals() and bubble_factor_returns is not None and not bubble_factor_returns.empty:
            bubble_factors = [c for c in ['AI bubble', 'Private Credit bubble', 'Crypto bubble'] if c in bubble_factor_returns.columns]
            if not bubble_factors:
                bubble_factors = list(bubble_factor_returns.columns)

        if (INCLUDE_SECTION10_SCENARIOS or INCLUDE_GPFG_SCENARIOS) and len(bubble_factors) >= 2:
            bubble_m = (1 + bubble_factor_returns[bubble_factors].reindex(rets.index).fillna(0.0)).resample(MONTHLY_FREQ).prod() - 1
            asset_m = (1 + rets[asset_tickers].fillna(0)).resample(MONTHLY_FREQ).prod() - 1

            betas: dict[str, pd.Series] = {}
            asset_mean_m = asset_m.mean()
            for a in asset_tickers:
                df = pd.concat([asset_m[a].rename('__asset__'), bubble_m], axis=1, join='inner').dropna()
                if df.shape[0] < 36:
                    continue
                y = df['__asset__']
                X = sm.add_constant(df[bubble_factors])
                try:
                    fit = sm.OLS(y, X).fit()
                except Exception:
                    continue
                betas[a] = fit.params

            def predict_portfolio_monthly_from_factors(factors: pd.DataFrame, weights: pd.Series) -> pd.Series:
                pred_assets = pd.DataFrame(index=factors.index, columns=asset_tickers, data=0.0)
                for a in asset_tickers:
                    if a in betas:
                        params = betas[a]
                        pred_assets[a] = float(params.get('const', 0.0))
                        for fac in factors.columns:
                            pred_assets[a] += float(params.get(fac, 0.0)) * factors[fac]
                    else:
                        pred_assets[a] = float(asset_mean_m.get(a, 0.0))
                return pred_assets.fillna(0.0).dot(weights.reindex(asset_tickers).fillna(0.0))

            def monthly_path(series: pd.Series, start: str, end: str) -> pd.Series:
                s = series.loc[start:end].copy()
                if len(s) == 0:
                    raise ValueError('Empty scenario path')
                return s

            ai_bubble = 'AI bubble'
            private_credit_bubble = 'Private Credit bubble'
            crypto_bubble = 'Crypto bubble'
            needed = [ai_bubble, private_credit_bubble, crypto_bubble]
            needed = [t for t in needed if t in bubble_m.columns]

            if needed:
                # Baseline factor fit
                train_start = pd.Timestamp.today().normalize() - pd.DateOffset(years=10)
                train_bubbles = bubble_m[needed].loc[train_start:].dropna()
                if train_bubbles.empty:
                    train_bubbles = bubble_m[needed].dropna()

                mu_f = train_bubbles.mean()
                cov_f = train_bubbles.cov()
                rng_future = np.random.default_rng(RANDOM_SEED + 2027)
                future_idx = pd.date_range(pd.Timestamp('2026-01-31'), periods=SHOCK_MONTHS, freq=MONTHLY_FREQ)
                base_samples = rng_future.multivariate_normal(mu_f.values, cov_f.values, size=len(future_idx))
                base_factors = pd.DataFrame(base_samples, index=future_idx, columns=needed)

                def apply_shock(df: pd.DataFrame, col: str, shock: pd.Series) -> pd.DataFrame:
                    df = df.copy()
                    idx = pd.date_range(df.index[0], periods=len(shock), freq=MONTHLY_FREQ)
                    idx = idx.intersection(df.index)
                    if len(idx) == 0:
                        return df
                    df.loc[idx, col] = shock.values[: len(idx)]
                    return df

                ai_shock_full = monthly_path(bubble_m[ai_bubble], *EVENT_WINDOWS['Dot-com crash (AI bubble stress)'])
                credit_shock_full = monthly_path(bubble_m[private_credit_bubble], *EVENT_WINDOWS['GFC (Private Credit bubble stress)'])
                crypto_shock_full = monthly_path(bubble_m[crypto_bubble], *EVENT_WINDOWS['Crypto winter (Crypto bubble stress)'])
                ai_shock = ai_shock_full.iloc[: min(SHOCK_MONTHS, len(ai_shock_full))]
                credit_shock = credit_shock_full.iloc[: min(SHOCK_MONTHS, len(credit_shock_full))]
                crypto_shock = crypto_shock_full.iloc[: min(SHOCK_MONTHS, len(crypto_shock_full))]

                if INCLUDE_SECTION10_SCENARIOS:
                    scenario_paths['Baseline (simulated 2026–2030)'] = base_factors
                    scenario_paths['AI bubble burst (2026–2030)'] = apply_shock(base_factors, ai_bubble, ai_shock)
                    scenario_paths['Private Credit bubble burst (2026–2030)'] = apply_shock(base_factors, private_credit_bubble, credit_shock)
                    scenario_paths['Crypto bubble burst (2026–2030)'] = apply_shock(base_factors, crypto_bubble, crypto_shock)
                    triple = base_factors.copy()
                    triple = apply_shock(triple, ai_bubble, ai_shock)
                    triple = apply_shock(triple, private_credit_bubble, credit_shock)
                    triple = apply_shock(triple, crypto_bubble, crypto_shock)
                    scenario_paths['Triple-bubble burst (2026–2030)'] = triple

                if INCLUDE_GPFG_SCENARIOS:
                    SCENARIO_SHOCKS = {
                        'AI correction (12m)': {ai_bubble: -0.53, private_credit_bubble: 0.05, crypto_bubble: -0.25},
                        'Fragmented world (12m)': {ai_bubble: -0.49, private_credit_bubble: -0.15, crypto_bubble: -0.30},
                        'Regional debt crisis (12m)': {ai_bubble: -0.39, private_credit_bubble: -0.25, crypto_bubble: -0.10},
                        'Extreme weather (12m)': {ai_bubble: -0.24, private_credit_bubble: -0.10, crypto_bubble: -0.05},
                        'Combined severe (12m)': {ai_bubble: -0.55, private_credit_bubble: -0.25, crypto_bubble: -0.30},
                    }
                    for nm, shocks in SCENARIO_SHOCKS.items():
                        df = base_factors.copy()
                        for fac, val in shocks.items():
                            if fac not in df.columns:
                                continue
                            # Interpret `val` as a 12-month total return and convert to a constant monthly path.
                            monthly_val = (1.0 + float(val)) ** (1.0 / float(SHOCK_MONTHS)) - 1.0
                            seq = pd.Series([monthly_val] * SHOCK_MONTHS, index=df.index[:SHOCK_MONTHS])
                            df = apply_shock(df, fac, seq)
                        scenario_paths[f'{nm} (GPFG)'] = df

        if win_cum:
            C = np.vstack(win_cum)  # shape: (n_windows, n_candidates)
            hist_min = np.nanmin(C, axis=0)
            hist_sum = np.nansum(C, axis=0)

            combined_min = hist_min.copy()
            combined_sum = hist_sum.copy()

            # Requirement: the final allocation must have non-negative cumulative return in *all*
            # historical stress windows. Treat this as a hard feasibility constraint.
            hist_ok = np.all(np.isfinite(C) & (C >= 0.0), axis=0)
            if hist_ok.any():
                combined_min = np.where(hist_ok, combined_min, -np.inf)
                combined_sum = np.where(hist_ok, combined_sum, -np.inf)
                print(
                    f"[allocation] Candidates passing all historical stress windows: {int(hist_ok.sum())} / {int(C.shape[1])}"
                )
            else:
                print(
                    '[allocation] WARNING: No candidates pass all historical stress windows under current constraints; falling back to unconstrained optimization.'
                )

            # Optional: include Section 10 + GPFG scenarios in the objective.
            # IMPORTANT: this must be fast; we evaluate all candidates at once.
            S_scen = None
            scen_names: list[str] = []
            base_scen = None
            if scenario_paths:
                # Factor order used in the scenario paths (built from `needed`).
                factor_cols = list(next(iter(scenario_paths.values())).columns)

                # Build per-asset regression params (const + betas) for fast portfolio-level evaluation.
                const_vec = np.zeros(len(asset_tickers), dtype=float)
                beta_mat = np.zeros((len(asset_tickers), len(factor_cols)), dtype=float)
                for i, a in enumerate(asset_tickers):
                    if a in betas:
                        params = betas[a]
                        const_vec[i] = float(params.get('const', 0.0))
                        for k, fac in enumerate(factor_cols):
                            beta_mat[i, k] = float(params.get(fac, 0.0))
                    else:
                        const_vec[i] = float(asset_mean_m.get(a, 0.0))

                # Candidate signed weights matrix W: (n_assets, n_candidates)
                sign_vec = sign.reindex(asset_tickers).fillna(1.0).to_numpy(dtype=float)
                W = sign_vec[:, None] * A * float(GROSS_EXPOSURE)

                # Portfolio-level exposures per candidate
                c = const_vec @ W  # shape: (n_candidates,)
                b = beta_mat.T @ W  # shape: (n_factors, n_candidates)

                # Baseline (non-optimized) portfolio exposures for comparison
                w_base_vec = base_w.reindex(asset_tickers).fillna(0.0).to_numpy(dtype=float)
                c_base = float(const_vec @ w_base_vec)
                b_base = beta_mat.T @ w_base_vec  # shape: (n_factors,)

                scen_cum_all = []
                base_cum_all = []
                for nm, fx in scenario_paths.items():
                    F = fx[factor_cols].fillna(0.0).to_numpy(dtype=float)  # (T, K)

                    # Candidate portfolio monthly returns (deterministic expectation)
                    Rm = F @ b + c[None, :]
                    Rm = np.clip(Rm, -0.999999, None)
                    cum = np.expm1(np.log1p(Rm).sum(axis=0))
                    scen_cum_all.append(cum)
                    scen_names.append(str(nm))

                    # Base portfolio scenario cumulative return
                    r_base = (F @ b_base) + c_base
                    base_cum_all.append(_cum_return_from_monthly_fast(r_base))

                if scen_cum_all:
                    S_scen = np.vstack(scen_cum_all)  # (n_scenarios, n_candidates)
                    base_scen = np.array(base_cum_all, dtype=float)
                    scen_min = np.nanmin(S_scen, axis=0)
                    scen_sum = np.nansum(S_scen, axis=0)
                    combined_min = np.minimum(combined_min, scen_min)
                    combined_sum = combined_sum + scen_sum

            # Maximize worst-case across historical + scenario sets; tie-breaker: sum
            best_j = int(np.nanargmax(combined_min + 1e-6 * combined_sum))
            opt_abs = A[:, best_j]
            best_w = (pd.Series(opt_abs, index=asset_tickers) * sign).rename('weight')
            best_w = best_w * (GROSS_EXPOSURE / float(best_w.abs().sum()))

            base_tbl = _stress_cum_returns(base_w)
            opt_tbl = _stress_cum_returns(best_w)
            comp = pd.DataFrame({'base': base_tbl, 'optimized': opt_tbl})
            comp = comp.T
            print('\n[allocation] Stress-window cumulative returns (base vs optimized):')
            display(comp.style.format('{:.2%}'))

            if (S_scen is not None) and (base_scen is not None) and (len(scen_names) == int(S_scen.shape[0])):
                scen_tbl = pd.DataFrame(
                    {
                        'base': base_scen,
                        'optimized': [float(S_scen[i, best_j]) for i in range(len(scen_names))],
                    },
                    index=pd.Index(scen_names, name='scenario'),
                ).sort_values('optimized')
                print('\n[allocation] Scenario cumulative returns (base vs optimized):')
                display(scen_tbl.style.format('{:.2%}'))
    else:
        print('[allocation] Skipping stress optimization (not enough feasible candidates under MAX_ABS_WEIGHT cap).')

allocation = pd.DataFrame(
    {
        'Symbol': asset_tickers,
        'Symbol name': roster_sub['Name'].values,
        'Weight (%)': (best_w.abs().values * 100.0),
        'Long/Short': pos.values,
        'Reasoning': roster_sub['Rationale'].values,
    }
).set_index('Symbol')

allocation = allocation.sort_values('Weight (%)', ascending=False)
allocation.style.format({'Weight (%)': '{:.2f}'})

# Basic stress-window performance of this roster-driven allocation
port = portfolio_returns(rets[asset_tickers], best_w)

# Note: historical stress-window backtest metrics are summarized in Section 7 (after the graphs)
# using window-renormalized weights, so the numbers and plots are consistent.

# RNG for later sections (frontier sampling / Monte Carlo)
rng = np.random.default_rng(RANDOM_SEED)

# Dynamic interpretation (updates when weights/data change)
if 'explain' in globals():
    try:
        bullets = []
        bullets.append(f'Investable assets used (with price data): {len(asset_tickers)}')

        try:
            gross = float(best_w.abs().sum())
            bullets.append(f'Gross exposure (sum abs weights): {gross:.2f} (target={GROSS_EXPOSURE:.2f})')
        except Exception:
            pass

        try:
            n_long = int((best_w > 0).sum())
            n_short = int((best_w < 0).sum())
            bullets.append(f'Long/Short counts: {n_long} LONG, {n_short} SHORT')
        except Exception:
            pass

        try:
            idx = best_w.reindex(asset_tickers).fillna(0.0).abs().sort_values(ascending=False).head(5).index
            top_w = best_w.reindex(idx)
            top_txt = '; '.join([f'{k} {v * 100:+.2f}%' for k, v in top_w.items()])
            if top_txt:
                bullets.append('Largest weights (signed): ' + top_txt)
        except Exception:
            pass

        bullets.append('If Yahoo prices update or you edit the roster CSV (Position/RiskLimitPercent), this allocation and all downstream sections will change on the next run.')
        explain('5) Portfolio allocation (live snapshot)', bullets)
    except Exception:
        pass
"""

    backtest_md = """## 7) Historical backtests (Dot-com / GFC / COVID)

We plot cumulative returns for each stress window and compute basic drawdown stats.

Note: some roster tickers may not have price history going back to 2000–2002. For each window we (1) report how many assets have data and (2) renormalize weights to the assets that existed in that window.

### Simple Explanation of the Codebase, Output, how it's supporting the goal of resilience vs the AI / Private Credit / Crypto bubbles
- This section replays history to answer: **“How would this portfolio have behaved in major crashes?”**
- The equity curves show the path; the summary table shows the exact cumulative return + max drawdown for those same curves.
- These windows are used as analogs for the 3-bubble risk (AI/Dot-com, Credit/GFC, Crypto/COVID-like stress).
"""

    backtest_code = """port = portfolio_returns(rets[asset_tickers], best_w)

def window_available_assets(start: str, end: str) -> list[str]:
    window = rets.loc[start:end, asset_tickers]
    return [c for c in window.columns if window[c].notna().any()]

def renormalized_window_weights(start: str, end: str) -> pd.Series:
    cols = window_available_assets(start, end)
    if not cols:
        return pd.Series(dtype=float)
    w = best_w.reindex(cols).fillna(0.0)
    if w.abs().sum() == 0:
        return pd.Series(dtype=float)
    return w * (GROSS_EXPOSURE / w.abs().sum())

def window_portfolio_returns(start: str, end: str) -> pd.Series:
    cols = window_available_assets(start, end)
    if not cols:
        return pd.Series(dtype=float)
    w = renormalized_window_weights(start, end)
    return portfolio_returns(rets.loc[start:end, cols], w)

def plot_equity_curve(r: pd.Series, title: str) -> None:
    if r is None or len(r) == 0:
        print('[skip] no data for:', title)
        return
    eq = (1 + r.fillna(0)).cumprod()
    plt.figure(figsize=(12, 5))
    plt.plot(eq.index, eq.values)
    plt.title(title)
    plt.ylabel('Equity (growth of $1)')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

# Full history plot
plot_equity_curve(port, 'Roster-driven bubble-resilient portfolio — Full History')

# Stress-window plots (renormalize to assets with data in each window)
summary_rows = []
for name in ['Dot-com crash (AI bubble stress)', 'GFC (Private Credit bubble stress)', 'COVID crash']:
    start, end = EVENT_WINDOWS[name]
    cols = window_available_assets(start, end)
    print(f"{name}: using {len(cols)} of {len(asset_tickers)} assets (weights renormalized within window)")
    wret = window_portfolio_returns(start, end)
    plot_equity_curve(wret, f'Portfolio — {name} ({start} → {end})')

    if wret is None or len(wret) == 0:
        summary_rows.append(
            {
                'window': name,
                'start': start,
                'end': end,
                'assets_used': len(cols),
                'cum_return': np.nan,
                'max_drawdown': np.nan,
            }
        )
        continue

    eq = (1 + wret.fillna(0)).cumprod()
    summary_rows.append(
        {
            'window': name,
            'start': start,
            'end': end,
            'assets_used': len(cols),
            'cum_return': float(eq.iloc[-1] - 1),
            'max_drawdown': max_drawdown(eq),
        }
    )

# Backtest summary table (these numbers correspond to the graphs above)
bt = pd.DataFrame(summary_rows).set_index('window')
try:
    from IPython.display import display
    display(bt.style.format({'cum_return': '{:.2%}', 'max_drawdown': '{:.2%}'}))
except Exception:
    # Fallback for non-notebook execution
    print(bt.to_string())

# Dynamic interpretation (updates with live Yahoo Finance pulls)
if 'explain' in globals():
    try:
        bullets = []
        cr = pd.to_numeric(bt['cum_return'], errors='coerce').dropna()
        dd = pd.to_numeric(bt['max_drawdown'], errors='coerce').dropna()

        if len(cr):
            best_window = cr.idxmax()
            worst_window = cr.idxmin()
            bullets.append(f'Best cumulative return window: {best_window} ({float(cr.loc[best_window]):.2%})')
            bullets.append(f'Worst cumulative return window: {worst_window} ({float(cr.loc[worst_window]):.2%})')

        if len(dd):
            worst_dd_w = dd.idxmin()
            best_dd_w = dd.idxmax()
            bullets.append(f'Worst max drawdown window: {worst_dd_w} ({float(dd.loc[worst_dd_w]):.2%})')
            bullets.append(f'Smallest max drawdown window: {best_dd_w} ({float(dd.loc[best_dd_w]):.2%})')

        bullets.append('Weights are renormalized within each window to the subset of assets that have data in that window (see the “using X of Y assets” lines above).')
        explain('7) Backtests: quick read', bullets)
    except Exception:
        pass

print()
print('Stress-window cumulative returns (computed from the same series used in the graphs):')
for w in ['Dot-com crash (AI bubble stress)', 'GFC (Private Credit bubble stress)', 'COVID crash']:
    if w not in bt.index:
        continue
    s = bt.loc[w, 'start']
    e = bt.loc[w, 'end']
    val = bt.loc[w, 'cum_return']
    if pd.isna(val):
        print(f' - {w} ({s} → {e}): n/a')
    else:
        print(f' - {w} ({s} → {e}): {float(val):.2%}')

# Metrics summary over full history
full_eq = (1 + port.fillna(0)).cumprod()
full_cum = float(full_eq.iloc[-1] - 1)
full_mdd = max_drawdown(full_eq)
full_ann_ret = ann_return_from_daily(port)
full_ann_vol = float(port.std() * np.sqrt(252))
full_sharpe = (full_ann_ret / full_ann_vol) if full_ann_vol and not np.isnan(full_ann_vol) else np.nan

pd.DataFrame({
    'metric': ['Cumulative return', 'CAGR (geo)', 'Ann vol', 'Sharpe (rf=0)', 'Max drawdown'],
    'value': [full_cum, full_ann_ret, full_ann_vol, full_sharpe, full_mdd],
})

# Display returns/risk metrics in % terms (Sharpe is unitless)
metrics = pd.DataFrame({
    'metric': ['Cumulative return', 'CAGR (geo)', 'Ann vol', 'Sharpe (rf=0)', 'Max drawdown'],
    'value': [full_cum, full_ann_ret, full_ann_vol, full_sharpe, full_mdd],
})

metrics['value'] = metrics.apply(
    lambda r: f"{r['value']:.2f}" if r['metric'].startswith('Sharpe') else f"{r['value']:.2%}",
    axis=1,
)
metrics
"""

    frontier_md = """## 6) Portfolio construction graph (Expected Return vs Expected Risk)

We sample many random long/short portfolios (same sign rule) and plot:
- X-axis: expected return
- Y-axis: expected risk (volatility)

We also overlay an **approximate efficient frontier curve** (best return achieved for each risk level among the sampled portfolios) so the chart reads like a classic efficient frontier under your long/short constraints.

### Simple Explanation of the Codebase, Output, how it's supporting the goal of resilience vs the AI / Private Credit / Crypto bubbles
- Blue dots = **different possible portfolios** under the same rules (Long/Short signs, weight caps).
- Orange line = an **approximate efficient frontier** (best return you can get for a given risk level, within what we sampled).
- Red star = your **roster-driven portfolio** from Section 5.
- Important: this is **baseline mean/variance**. Bubble-burst resilience is judged more directly in Sections 7/8/10.
"""

    frontier_code = """# Use recent window for expected return/risk estimates
train_start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
train = rets.loc[train_start:, asset_tickers].dropna(how='all').fillna(0.0)

mu = train.mean() * 252
cov = train.cov() * 252

def port_mu_sigma(weights: np.ndarray) -> tuple[float, float]:
    m = float(np.dot(weights, mu.values))
    v = float(np.dot(weights, np.dot(cov.values, weights)))
    return m, float(np.sqrt(max(v, 0.0)))

N_SAMPLES = 4000
pts = []
for _ in range(N_SAMPLES):
    raw = rng.dirichlet(np.ones(len(asset_tickers)))
    w = raw * signs
    w = w * (GROSS_EXPOSURE / np.sum(np.abs(w)))
    if np.max(np.abs(w)) > MAX_ABS_WEIGHT:
        continue
    m, s = port_mu_sigma(w)
    pts.append((m, s))

pts = np.array(pts)
roster_m, roster_s = port_mu_sigma(best_w.values)

# Approximate efficient frontier from sampled portfolios (upper envelope)
# Sort by risk (sigma), then keep portfolios that improve the best return so far.
order = np.argsort(pts[:, 1])
sorted_pts = pts[order]
front = []
best_m = -np.inf
for m, s in sorted_pts:
    if m > best_m:
        front.append((m, s))
        best_m = m
front = np.array(front)

plt.figure(figsize=(10, 6))
plt.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.25, label='Sampled portfolios')
if len(front) > 1:
    plt.plot(front[:, 0], front[:, 1], color='orange', linewidth=2.5, label='Efficient frontier (sampled)')
plt.scatter([roster_m], [roster_s], s=140, marker='*', color='red', label='Roster-driven bubble-resilient portfolio (Section 5)')
plt.xlabel('Expected return (annualized)')
plt.ylabel('Expected risk (annualized vol)')
plt.title('Return vs Risk — sampled long/short FICC portfolios')
plt.legend()
plt.tight_layout()
plt.show()

print('Roster-driven expected return (annualized):', f'{roster_m:.2%}')
print('Roster-driven expected risk (annualized vol):', f'{roster_s:.2%}')

# Dynamic interpretation (updates with live Yahoo Finance pulls)
if 'explain' in globals():
    try:
        bullets = []
        bullets.append(f'Sampled portfolios kept after caps: {int(len(pts))} of {int(N_SAMPLES)}')
        bullets.append(f'Roster expected return: {roster_m:.2%} (annualized)')
        bullets.append(f'Roster expected risk: {roster_s:.2%} (annualized vol)')
        if len(front) > 1:
            bullets.append(f'Frontier envelope points (sampled): {int(len(front))}')
        explain('6) Return vs risk chart (how to read now)', bullets)
    except Exception:
        pass
"""

    mc_md = """## 8) Forward-looking stress Monte Carlo (annual bubble-factor shocks) for 2026–2030

We simulate **5,000** paths of **annual** bubble-factor shocks (AI bubble / Private Credit bubble / Crypto bubble) across **2026–2030**, then compound yearly portfolio returns to produce a **5-year distribution**:
- Mean / Median 5-year return
- 5th / 95th percentile 5-year return
- Annualized volatility (based on simulated yearly portfolio returns)

### Simple Explanation of the Codebase, Output, how it's supporting the goal of resilience vs the AI / Private Credit / Crypto bubbles
- Instead of one forecast, we generate **many possible 2026–2030 futures** by shocking the **3 composite bubble factors** (AI bubble / Private Credit bubble / Crypto bubble), built from the full per-bubble Yahoo Finance symbol sets.
- The output is a distribution (mean/median/5th/95th) so you can see both upside and downside.
- This is a forward-looking way to gauge “what could happen” under bubble-factor turbulence.
"""

    mc_code = """# Forward-looking stress Monte Carlo settings
N_PATHS = 5000
YEARS = list(range(2026, 2031))  # 2026–2030 inclusive
N_YEARS = len(YEARS)
BURST_YEAR = 2027               # choose when the triple-bubble burst happens inside 2026–2030
TAIL_K = 5                      # tail years used to source burst-year shocks per factor
TRAIN_YEARS = 10                # baseline distribution fit window

from pandas.tseries.frequencies import to_offset

try:
    to_offset('YE')
    YEARLY_FREQ = 'YE'
except Exception:
    YEARLY_FREQ = 'Y'

# Annual returns (from daily returns)
asset_y = (1 + rets[asset_tickers].fillna(0)).resample(YEARLY_FREQ).prod() - 1

if 'bubble_factor_returns' not in globals() or bubble_factor_returns is None or bubble_factor_returns.empty:
    raise RuntimeError('bubble_factor_returns missing/empty; run Section 3 first.')
bubble_factors = [c for c in ['AI bubble', 'Private Credit bubble', 'Crypto bubble'] if c in bubble_factor_returns.columns]
if not bubble_factors:
    bubble_factors = list(bubble_factor_returns.columns)

factor_y = (1 + bubble_factor_returns[bubble_factors].reindex(rets.index).fillna(0.0)).resample(YEARLY_FREQ).prod() - 1

# Fit annual factor models: asset_annual ~ const + factors
betas = {}
resid_sigmas = {}

for a in asset_tickers:
    y_series = asset_y[a].rename('__asset__')
    df = pd.concat([y_series, factor_y[bubble_factors]], axis=1, join='inner').dropna()
    if df.shape[0] < 10:
        continue
    y = df['__asset__']
    X = sm.add_constant(df[bubble_factors])
    fit = sm.OLS(y, X).fit()
    betas[a] = fit.params
    resid_sigmas[a] = float(fit.resid.std())

print('Annual factor models fit:', len(betas), 'of', len(asset_tickers), 'assets')

# Baseline factor distribution fit (use last TRAIN_YEARS years if available)
train_start = pd.Timestamp.today().normalize() - pd.DateOffset(years=TRAIN_YEARS)
train_f = factor_y.loc[train_start:].dropna()
if train_f.empty:
    train_f = factor_y.dropna()

mu = train_f[bubble_factors].mean()
cov = train_f[bubble_factors].cov()

# Tail shocks per factor (burst-year shocks sourced from worst annual returns)
tail_samples = {}

# Tail shocks per factor (burst-year shocks sourced from worst annual returns)
tail_samples = {}
for f in bubble_factors:
    s = factor_y[f].dropna().sort_values()
    tail_samples[f] = s.head(min(TAIL_K, len(s))).to_numpy()

rng_mc = np.random.default_rng(RANDOM_SEED + 123)

# Simulate baseline annual factor returns for all years
base = rng_mc.multivariate_normal(mu.values, cov.values, size=(N_PATHS, N_YEARS))
factors = base.copy()  # shape: (paths, years, factors)

# Inject triple-bubble burst in BURST_YEAR by replacing that year's factor returns with tail shocks
burst_idx = YEARS.index(BURST_YEAR) if BURST_YEAR in YEARS else None
if burst_idx is not None:
    for j, f in enumerate(bubble_factors):
        tails = tail_samples.get(f)
        if tails is not None and len(tails) > 0:
            factors[:, burst_idx, j] = rng_mc.choice(tails, size=N_PATHS, replace=True)

# Build beta matrices (const + factor loadings) for vectorized simulation
n_assets = len(asset_tickers)
n_factors = len(bubble_factors)

B = np.zeros((n_assets, n_factors))
c = np.zeros(n_assets)
sig = np.zeros(n_assets)
asset_mean_y = asset_y.mean()

for i, a in enumerate(asset_tickers):
    if a in betas:
        params = betas[a]
        c[i] = float(params.get('const', 0.0))
        for j, f in enumerate(bubble_factors):
            B[i, j] = float(params.get(f, 0.0))
        sig[i] = float(resid_sigmas.get(a, 0.0))
    else:
        c[i] = float(asset_mean_y.get(a, 0.0))
        sig[i] = 0.0

# Simulate annual asset returns and annual portfolio returns
asset_pred = factors @ B.T + c[None, None, :]
eps = rng_mc.normal(loc=0.0, scale=sig[None, None, :], size=asset_pred.shape)
asset_sim = asset_pred + eps

w = best_w.reindex(asset_tickers).fillna(0.0).to_numpy()
port_yearly = asset_sim @ w  # shape: (paths, years)

# Compound annual returns into equity and 5-year cumulative return
eq = (1 + port_yearly).cumprod(axis=1)
cum_5y = eq[:, -1] - 1

mean5 = float(np.mean(cum_5y))
median5 = float(np.median(cum_5y))
p5 = float(np.percentile(cum_5y, 5))
p95 = float(np.percentile(cum_5y, 95))
ann_vol = float(port_yearly.std())  # annual volatility of yearly portfolio returns

stats_table = pd.DataFrame(
    {
        'metric': [
            'Mean 5-year portfolio return',
            'Median 5-year portfolio return',
            '5th percentile 5-year return',
            '95th percentile 5-year return',
            'Annualized volatility (yearly)',
        ],
        'value': [mean5, median5, p5, p95, ann_vol],
    }
)
stats_table['value'] = stats_table['value'].apply(lambda x: f"{x:.2%}")
stats_table

# Distribution plot
plt.figure(figsize=(10, 5))
plt.hist(cum_5y, bins=60, alpha=0.8)
plt.axvline(p5, color='red', linestyle='--', alpha=0.8, label=f'5th pct: {p5:.2%}')
plt.axvline(p95, color='green', linestyle='--', alpha=0.8, label=f'95th pct: {p95:.2%}')
plt.axvline(median5, color='black', linestyle='--', alpha=0.8, label=f'Median: {median5:.2%}')
plt.title('Stress Monte Carlo (annual bubble-factor shocks): 5-year return distribution (2026–2030)')
plt.xlabel('5-year cumulative return')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# Sample equity paths
t = np.array(YEARS)
plt.figure(figsize=(12, 5))
plt.plot(t, eq[:30].T, alpha=0.5)
if burst_idx is not None:
    plt.axvline(YEARS[burst_idx], color='black', linestyle='--', alpha=0.6, label='burst year')
plt.title('Sample Monte Carlo equity paths (annual compounding) — 2026–2030')
plt.ylabel('Equity (growth of $1)')
plt.xlabel('Year')
if burst_idx is not None:
    plt.legend()
plt.tight_layout()
plt.show()

# Fan chart percentiles
bands = np.percentile(eq, [5, 50, 95], axis=0)
plt.figure(figsize=(12, 5))
plt.fill_between(t, bands[0], bands[2], alpha=0.2, label='5–95% band')
plt.plot(t, bands[1], color='black', linewidth=2, label='Median')
if burst_idx is not None:
    plt.axvline(YEARS[burst_idx], color='black', linestyle='--', alpha=0.6)
plt.title('Monte Carlo fan chart: equity percentiles — 2026–2030')
plt.ylabel('Equity (growth of $1)')
plt.xlabel('Year')
plt.legend()
plt.tight_layout()
plt.show()

# Dynamic interpretation (updates with live Yahoo Finance pulls)
if 'explain' in globals():
    try:
        explain(
            '8) Monte Carlo (2026–2030): quick read',
            [
                f'Paths simulated: {N_PATHS} | Burst year: {BURST_YEAR}',
                f'5-year return: mean={mean5:.2%}, median={median5:.2%}',
                f'Downside/upside band (5th–95th): {p5:.2%} to {p95:.2%}',
                f'Annualized volatility (yearly returns): {ann_vol:.2%}',
                'These figures update when Yahoo factor history or the portfolio weights change.',
            ],
        )
    except Exception:
        pass
"""

    forecast_md = """## 9) Time-series forecasting (Auto-ARIMA + Seasonal SARIMAX + Regime/Volatility models) for 2026–2030

This section upgrades forecasting realism in three complementary ways:

1) **Automatic ARIMA order selection** (AIC-based) on monthly portfolio returns
2) **Seasonal SARIMAX** candidates (monthly seasonality)
3) **Regime / volatility modeling** (Markov switching variance)

For each model we show the mean forecast + CI *and* a simulated fan chart, so you can see realistic volatility/uncertainty rather than a flat mean line.

### Simple Explanation of the Codebase, Output, how it's supporting the goal of resilience vs the AI / Private Credit / Crypto bubbles
- These models generate **baseline future paths** from historical behavior (not explicit bubble-burst scenarios).
- Use the **fan charts** to understand uncertainty/volatility; use the comparison table to see how model assumptions change the downside range.
- For explicit “AI/Private Credit/Crypto bubble burst” scenarios, Section 10 is the main forward-looking stress test.
"""

    forecast_code = """# Monthly returns for portfolio and selected assets
port_monthly = (1 + port.fillna(0)).resample(MONTHLY_FREQ).prod() - 1
port_monthly = port_monthly.dropna()

asset_monthly = (1 + rets[asset_tickers].fillna(0)).resample(MONTHLY_FREQ).prod() - 1
asset_monthly = asset_monthly.dropna(how='all')

# Forecast months until 2030-12 (month-end)
last_month = port_monthly.index.max()
end_month = pd.Timestamp('2030-12-31').to_period('M').to_timestamp(how='end')
steps = (end_month.to_period('M') - last_month.to_period('M')).n
steps = max(int(steps), 1)
future_index = pd.date_range(last_month + pd.offsets.MonthEnd(1), periods=steps, freq=MONTHLY_FREQ)

print('Portfolio monthly returns span:', port_monthly.index.min().date(), '→', port_monthly.index.max().date())
print('Forecast horizon:', future_index[0].date(), '→', future_index[-1].date(), f'({steps} months)')

# Notebook-friendly display helper
def _display_df(df: pd.DataFrame, fmt: dict | None = None) -> None:
    try:
        from IPython.display import display

        if fmt:
            display(df.style.format(fmt))
        else:
            display(df)
    except Exception:
        print(df)

hist_eq = (1 + port_monthly).cumprod()
start_eq = float(hist_eq.iloc[-1]) if len(hist_eq) else 1.0

def _normalize_sim(sim_arr: np.ndarray, n_steps: int, n_paths: int) -> np.ndarray:
    sim_arr = np.asarray(sim_arr)
    if sim_arr.ndim == 1:
        sim_arr = sim_arr.reshape(-1, 1)
    if sim_arr.shape[0] != n_steps and sim_arr.shape[1] == n_steps:
        sim_arr = sim_arr.T
    return sim_arr[:n_steps, :n_paths]

def _simulate_from_model(model, mean_series: pd.Series, n_paths: int, rng: np.random.Generator, fallback_sigma: float) -> np.ndarray:
    sim_arr = None
    try:
        sim = model.simulate(nsimulations=len(mean_series), anchor='end', repetitions=n_paths)
        sim_arr = np.asarray(sim)
    except Exception:
        sim_arr = None

    if sim_arr is None or np.asarray(sim_arr).size == 0:
        sigma = fallback_sigma
        try:
            sigma = float(model.resid.std())
        except Exception:
            pass
        z = rng.normal(size=(len(mean_series), n_paths))
        sim_arr = mean_series.to_numpy()[:, None] + sigma * z

    return _normalize_sim(sim_arr, len(mean_series), n_paths)

def _fan_chart(title: str, mean_returns: pd.Series, sim_returns: np.ndarray) -> dict:
    mean_eq = start_eq * (1 + mean_returns).cumprod()
    eq_paths = start_eq * (1 + sim_returns).cumprod(axis=0)
    p5, p50, p95 = np.percentile(eq_paths, [5, 50, 95], axis=1)

    plt.figure(figsize=(12, 5))
    plt.plot(hist_eq.index, hist_eq.values, label='history', alpha=0.7)
    plt.plot(mean_returns.index, eq_paths[:, :20], color='steelblue', alpha=0.15)
    plt.fill_between(mean_returns.index, p5, p95, alpha=0.2, label='5–95% band')
    plt.plot(mean_returns.index, p50, color='black', linewidth=2, label='Median')
    plt.plot(mean_returns.index, mean_eq.values, color='red', linestyle='--', label='Mean path')
    plt.axvline(hist_eq.index[-1], color='gray', linestyle='--', alpha=0.6)
    plt.title(title)
    plt.ylabel('Equity (growth of $1)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    cum_paths = eq_paths[-1, :] / start_eq - 1
    return {
        'mean_cum_return': float(mean_eq.iloc[-1] / start_eq - 1),
        'p5_cum_return': float(np.percentile(cum_paths, 5)),
        'p50_cum_return': float(np.percentile(cum_paths, 50)),
        'p95_cum_return': float(np.percentile(cum_paths, 95)),
    }


print()
print('--- 9A) Auto-ARIMA order selection (AIC) ---')

def fit_arima_auto(series: pd.Series, p_values=(0, 1, 2), d_values=(0, 1), q_values=(0, 1, 2)):
    candidates = []
    best_model = None
    best_aic = np.inf
    best_spec = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                trend = 'c' if d == 0 else 'n'
                try:
                    m = ARIMA(series, order=(p, d, q), trend=trend).fit()
                    aic = float(m.aic)
                    candidates.append({'p': p, 'd': d, 'q': q, 'trend': trend, 'aic': aic})
                    if np.isfinite(aic) and aic < best_aic:
                        best_aic = aic
                        best_model = m
                        best_spec = {'order': (p, d, q), 'trend': trend, 'aic': aic}
                except Exception:
                    continue

    tbl = pd.DataFrame(candidates).sort_values('aic')
    return best_model, best_spec, tbl


arima_model, arima_spec, arima_tbl = fit_arima_auto(port_monthly)
if arima_model is None:
    raise RuntimeError('ARIMA auto-selection failed: no candidate models could be fit.')

print('Selected ARIMA:', arima_spec)
_display_df(arima_tbl.head(10), fmt={'aic': '{:.2f}'})

arima_fc = arima_model.get_forecast(steps=steps)
arima_mean = pd.Series(arima_fc.predicted_mean.to_numpy()[:steps], index=future_index)

rng_arima = np.random.default_rng(RANDOM_SEED + 999)
sim_arima = _simulate_from_model(arima_model, arima_mean, 300, rng_arima, fallback_sigma=float(port_monthly.std()))

print()
print('Simple-English (9A Auto-ARIMA):')
print(' - What it is: a baseline time-series model that assumes ONE stable return regime with short-memory dynamics.')
print(' - What the fan chart means: many plausible 2026–2030 equity paths for this portfolio, not a single prediction line.')
print(' - How to read it: focus on the 5–95% band and especially the 5th percentile (p5) as a conservative downside baseline.')
print(' - Bubble context: this is a *baseline* forecast; the explicit bubble-burst stress tests are still Sections 7/8/10.')
summary_arima = _fan_chart('Portfolio equity: Auto-ARIMA fan chart (through 2030)', arima_mean, sim_arima)


print()
print('--- 9B) Seasonal SARIMAX candidates (monthly seasonality) ---')

seasonal_candidates = [
    (0, 0, 0, 12),
    (1, 0, 0, 12),
    (0, 0, 1, 12),
    (1, 0, 1, 12),
]

sarimax_rows = []
sarimax_model = None
sarimax_seasonal = None
sarimax_best_aic = np.inf

for seas in seasonal_candidates:
    try:
        m = sm.tsa.statespace.SARIMAX(
            port_monthly,
            order=arima_spec['order'],
            seasonal_order=seas,
            trend=arima_spec['trend'],
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        aic = float(m.aic)
        sarimax_rows.append(
            {
                'order': str(arima_spec['order']),
                'seasonal_order': str(seas),
                'trend': arima_spec['trend'],
                'aic': aic,
            }
        )
        if np.isfinite(aic) and aic < sarimax_best_aic:
            sarimax_best_aic = aic
            sarimax_model = m
            sarimax_seasonal = seas
    except Exception:
        continue

summary_sarimax = None
if sarimax_rows:
    sarimax_tbl = pd.DataFrame(sarimax_rows).sort_values('aic')
    _display_df(sarimax_tbl, fmt={'aic': '{:.2f}'})
else:
    sarimax_tbl = pd.DataFrame([])
    print('No SARIMAX seasonal models could be fit.')

if sarimax_model is not None:
    print('Selected seasonal_order:', sarimax_seasonal)
    fc = sarimax_model.get_forecast(steps=steps)
    mean = pd.Series(fc.predicted_mean.to_numpy()[:steps], index=future_index)

    rng_sarimax = np.random.default_rng(RANDOM_SEED + 1999)
    sim_sarimax = _simulate_from_model(sarimax_model, mean, 300, rng_sarimax, fallback_sigma=float(port_monthly.std()))

    print()
    print('Simple-English (9B Seasonal SARIMAX):')
    print(' - What it is: like ARIMA, but it also allows repeating monthly patterns (seasonality) if the data actually contains them.')
    print(' - What to compare vs 9A: if this fan chart meaningfully shifts the downside (p5) or widens/narrows the band, seasonality matters.')
    print(' - Bubble context: still baseline behavior; use bubble stress sections to test explicit shock scenarios.')
    summary_sarimax = _fan_chart('Portfolio equity: Seasonal SARIMAX fan chart (through 2030)', mean, sim_sarimax)


print()
print('--- 9C) Regime / volatility model (Markov switching variance) ---')

summary_regime = None
try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    mr = MarkovRegression(port_monthly, k_regimes=2, trend='c', switching_variance=True)
    mr_res = mr.fit(disp=False)

    probs = mr_res.smoothed_marginal_probabilities

    params = mr_res.params

    def _pget(key: str, default: float) -> float:
        try:
            return float(params[key])
        except Exception:
            return default

    p00 = _pget('p[0->0]', np.nan)
    p10 = _pget('p[1->0]', np.nan)
    if np.isfinite(p00) and np.isfinite(p10):
        P = np.array([[p00, 1 - p00], [p10, 1 - p10]])
    else:
        reg = probs.idxmax(axis=1).to_numpy()
        counts = np.zeros((2, 2))
        for a, b in zip(reg[:-1], reg[1:]):
            counts[int(a), int(b)] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        P = counts / row_sums

    mu0 = _pget('const[0]', np.nan)
    mu1 = _pget('const[1]', np.nan)
    v0 = _pget('sigma2[0]', np.nan)
    v1 = _pget('sigma2[1]', np.nan)

    if not (np.isfinite(mu0) and np.isfinite(mu1) and np.isfinite(v0) and np.isfinite(v1)):
        reg = probs.idxmax(axis=1)
        mu0 = float(port_monthly[reg == 0].mean())
        mu1 = float(port_monthly[reg == 1].mean())
        v0 = float(port_monthly[reg == 0].var())
        v1 = float(port_monthly[reg == 1].var())

    sig0 = float(np.sqrt(abs(v0)))
    sig1 = float(np.sqrt(abs(v1)))

    # Label regimes as Calm vs Stress based on volatility (sigma)
    if sig0 <= sig1:
        calm_regime = 0
        stress_regime = 1
    else:
        calm_regime = 1
        stress_regime = 0

    print(f'Regime mapping: Regime {calm_regime} = CALM (lower vol); Regime {stress_regime} = STRESS (higher vol).')

    plt.figure(figsize=(12, 3))
    plt.plot(probs.index, probs[calm_regime], label=f'Calm prob (Regime {calm_regime})')
    plt.plot(probs.index, probs[stress_regime], label=f'Stress prob (Regime {stress_regime})')
    plt.title('Smoothed regime probabilities: Calm vs Stress (Markov switching)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    regime_table = pd.DataFrame(
        {
            'label': [
                'Calm' if calm_regime == 0 else 'Stress',
                'Calm' if calm_regime == 1 else 'Stress',
            ],
            'mean_monthly_return': [mu0, mu1],
            'vol_monthly': [sig0, sig1],
            'P(stay)': [float(P[0, 0]), float(P[1, 1])],
        },
        index=pd.Index([0, 1], name='regime'),
    )
    regime_table = regime_table.sort_values('vol_monthly')
    _display_df(regime_table, fmt={'mean_monthly_return': '{:.2%}', 'vol_monthly': '{:.2%}', 'P(stay)': '{:.2%}'})

    print()
    print('Simple-English (9C Regime-switching / Markov volatility):')
    print(' - What it is: the portfolio can switch between a calmer regime and a higher-volatility regime.')
    print(f' - Mapping used here: Regime {calm_regime} = Calm (lower vol) and Regime {stress_regime} = Stress (higher vol).')
    print(' - The regime-probability chart shows when the portfolio historically behaved more like “calm” vs “stress”.')
    print(' - The fan chart simulates 2026–2030 paths that can jump between regimes — often closer to crisis/bubble-burst style behavior.')

    N_SIM = 300
    rng_reg = np.random.default_rng(RANDOM_SEED + 3030)
    init_p = probs.iloc[-1].to_numpy()
    init_p = init_p / init_p.sum() if init_p.sum() > 0 else np.array([0.5, 0.5])

    regimes = rng_reg.choice([0, 1], size=N_SIM, p=init_p)
    sim_rets = np.zeros((steps, N_SIM), dtype=float)

    for t in range(steps):
        r0 = rng_reg.normal(loc=mu0, scale=sig0, size=N_SIM)
        r1 = rng_reg.normal(loc=mu1, scale=sig1, size=N_SIM)
        sim_rets[t] = np.where(regimes == 0, r0, r1)

        if t < steps - 1:
            u = rng_reg.random(size=N_SIM)
            next_reg = np.where(
                regimes == 0,
                np.where(u < P[0, 0], 0, 1),
                np.where(u < P[1, 0], 0, 1),
            )
            regimes = next_reg

    # Mean path via propagated expected regime probabilities
    pi = init_p.copy()
    mean_path = []
    for _ in range(steps):
        mean_path.append(pi[0] * mu0 + pi[1] * mu1)
        pi = pi @ P
    mean_path = pd.Series(mean_path, index=future_index)

    summary_regime = _fan_chart('Portfolio equity: Regime-switching (Markov) fan chart (through 2030)', mean_path, sim_rets)
except Exception as e:
    print('Regime-switching model could not be fit:', str(e)[:200])


print()
print('--- 9D) Model comparison (cumulative return to 2030) ---')

rows = [{'model': 'Auto ARIMA', **summary_arima}]
if summary_sarimax is not None:
    rows.append({'model': 'Seasonal SARIMAX', **summary_sarimax})
if summary_regime is not None:
    rows.append({'model': 'Regime-switching (Markov)', **summary_regime})

compare = pd.DataFrame(rows).set_index('model')
_display_df(compare, fmt={
    'mean_cum_return': '{:.2%}',
    'p5_cum_return': '{:.2%}',
    'p50_cum_return': '{:.2%}',
    'p95_cum_return': '{:.2%}',
})

print()
print('Simple-English interpretation (in the context of beating the AI / Private Credit / Crypto bubbles):')
print(' - These 3 models are *different ways to simulate baseline 2026–2030 return paths* for the SAME portfolio.')
print(' - Auto ARIMA: assumes one stable regime (single volatility level) with short-memory dynamics.')
print(' - Seasonal SARIMAX: same idea, but allows repeating monthly seasonality patterns (if they exist in the data).')
print(' - Regime-switching (Markov): allows switching between calm vs high-volatility regimes — often closer to crisis/bubble-burst behavior.')
print(' - How to read the table:')
print('    * mean_cum_return: expected baseline growth (not guaranteed).')
print('    * p5_cum_return: conservative downside (bad but plausible). Higher p5 → more resilient baseline.')
print('    * p50/p95: median/upside ranges.')
print(' - For the “triple-bubble” goal: use this table for *baseline uncertainty*, and use Sections 7/8/10 for *explicit bubble-burst stress tests*.')


print()
print('--- 9E) Asset-level forecasts (baseline) + implied portfolio fan chart from asset distributions ---')

def fit_asset_arima(series: pd.Series):
    series = series.dropna()
    if len(series) < 60:
        return None
    try:
        return ARIMA(series, order=arima_spec['order'], trend=arima_spec['trend']).fit()
    except Exception:
        try:
            return ARIMA(series, order=(1, 0, 0), trend='c').fit()
        except Exception:
            return None

asset_fc_means = {}
asset_fc_sigmas = {}
rows = []

for sym in asset_tickers:
    s = asset_monthly[sym].dropna()
    m = fit_asset_arima(s)
    if m is None:
        continue

    fc = m.get_forecast(steps=steps)
    mean_values = fc.predicted_mean.to_numpy()
    if len(mean_values) < steps:
        mean_values = np.pad(mean_values, (0, steps - len(mean_values)), mode='edge')
    mean_values = mean_values[:steps]

    try:
        sigma = float(m.resid.std())
    except Exception:
        sigma = float(s.std())
    sigma = float(sigma) if np.isfinite(sigma) else 0.0

    asset_fc_means[sym] = mean_values
    asset_fc_sigmas[sym] = sigma
    rows.append(
        {
            'symbol': sym,
            'name': yahoo_name(sym),
            'weight': float(best_w.get(sym, 0.0)),
            'hist_months': int(len(s)),
            'mean_monthly_forecast': float(np.mean(mean_values)),
            'cum_return_forecast': float((1 + pd.Series(mean_values)).prod() - 1),
        }
    )

asset_forecast_table = pd.DataFrame(rows).set_index('symbol').sort_values(
    'weight', key=lambda s: np.abs(s), ascending=False
)

_display_df(asset_forecast_table, fmt={
    'weight': '{:.2%}',
    'mean_monthly_forecast': '{:.2%}',
    'cum_return_forecast': '{:.2%}',
})

hist_mu = asset_monthly.mean()
hist_sigma = asset_monthly.std()

mu_mat = np.zeros((steps, len(asset_tickers)), dtype=float)
sig_mat = np.zeros((steps, len(asset_tickers)), dtype=float)

for j, sym in enumerate(asset_tickers):
    if sym in asset_fc_means:
        mu_mat[:, j] = asset_fc_means[sym]
        sig = float(asset_fc_sigmas.get(sym, 0.0))
    else:
        mu = float(hist_mu.get(sym, 0.0))
        sig = float(hist_sigma.get(sym, 0.0))
        mu_mat[:, j] = mu if np.isfinite(mu) else 0.0

    sig = sig if np.isfinite(sig) else 0.0
    sig_mat[:, j] = sig

print('Assets with ARIMA forecasts used:', len(asset_fc_means), 'of', len(asset_tickers))

w = best_w.reindex(asset_tickers).fillna(0.0).to_numpy()
port_mean = mu_mat @ w

N_SIM_ASSET_PORT = 300
rng_assets = np.random.default_rng(RANDOM_SEED + 2021)
z = rng_assets.normal(size=(steps, N_SIM_ASSET_PORT, len(asset_tickers)))
asset_sims = mu_mat[:, None, :] + sig_mat[:, None, :] * z
port_sims = np.tensordot(asset_sims, w, axes=([2], [0]))

mean_eq = start_eq * (1 + pd.Series(port_mean, index=future_index)).cumprod()
sim_eq2 = start_eq * (1 + port_sims).cumprod(axis=0)
q5, q50, q95 = np.percentile(sim_eq2, [5, 50, 95], axis=1)

plt.figure(figsize=(12, 5))
plt.plot(hist_eq.index, hist_eq.values, label='history', alpha=0.7)
plt.plot(future_index, sim_eq2[:, :20], color='steelblue', alpha=0.15)
plt.fill_between(future_index, q5, q95, alpha=0.2, label='5–95% band')
plt.plot(future_index, q50, color='black', linewidth=2, label='Median')
plt.plot(future_index, mean_eq.values, color='red', linestyle='--', label='Mean path')
plt.axvline(hist_eq.index[-1], color='gray', linestyle='--', alpha=0.6)
plt.title('Implied portfolio equity from asset forecast distributions (fan chart) (through 2030)')
plt.ylabel('Equity (growth of $1)')
plt.legend()
plt.tight_layout()
plt.show()

print()
print('Simple-English (9E Asset-implied fan chart):')
print(' - What it is: we forecast each asset (baseline ARIMA), then combine them using the portfolio weights to simulate portfolio equity paths.')
print(' - Why it helps: it’s a cross-check — it tells a similar story to 9A/9B/9C when the assumptions are consistent.')
print(' - If 9E differs a lot: usually it means limited asset model coverage (few assets had enough history) or different volatility assumptions by asset.')

print('Implied mean monthly portfolio forecast (from asset models):', f"{float(np.mean(port_mean)):.2%}")
print('Implied mean cumulative return (from asset models):', f"{float(mean_eq.iloc[-1] / start_eq - 1):.2%}")

# Dynamic interpretation (updates with live Yahoo Finance pulls)
if 'explain' in globals():
    try:
        bullets = []
        try:
            bullets.append(f'Forecast horizon: {future_index[0].date()} → {future_index[-1].date()} ({steps} months)')
        except Exception:
            pass

        try:
            p5s = pd.to_numeric(compare['p5_cum_return'], errors='coerce').dropna()
            if len(p5s):
                best_p5 = p5s.idxmax()
                worst_p5 = p5s.idxmin()
                bullets.append(f'Best downside (highest p5 cum return): {best_p5} ({float(p5s.loc[best_p5]):.2%})')
                bullets.append(f'Worst downside (lowest p5 cum return): {worst_p5} ({float(p5s.loc[worst_p5]):.2%})')
        except Exception:
            pass

        try:
            means = pd.to_numeric(compare['mean_cum_return'], errors='coerce').dropna()
            if len(means):
                means_txt = '; '.join([f'{k} {v:.2%}' for k, v in means.items()])
                bullets.append('Mean cumulative return to 2030 by model: ' + means_txt)
        except Exception:
            pass

        try:
            bullets.append(f'Assets with ARIMA forecasts used in 9E: {len(asset_fc_means)} of {len(asset_tickers)}')
        except Exception:
            pass

        explain('9) Forecasting (live summary)', bullets)
    except Exception:
        pass
"""

    stress_md = """## 10) Stress tests: Forward-looking triple-bubble burst scenario (2026–2030)

We fit a **multi-factor regression** of each portfolio asset on the **composite bubble factors**:
- **AI bubble** (composite)
- **Private Credit bubble** (composite)
- **Crypto bubble** (composite)

Then we build a **forward-looking 2026–2030** scenario by simulating monthly bubble returns and **injecting historical analog shock sequences** into that future window.

**Output policy for this section:** keep it minimal. We plot the bubble paths + portfolio equity curves, plus an additional bar chart of scenario cumulative returns, and show the forward-looking results table.

### Simple Explanation of the Codebase, Output, how it's supporting the goal of resilience vs the AI / Private Credit / Crypto bubbles
- This is the most direct answer to the project objective: **“What happens if each bubble bursts in 2026–2030? And what if all three burst?”**
- We simulate bubble-factor paths and inject historical crash sequences as “shock blocks” into the future.
- Then we translate those bubble moves into portfolio returns and compare scenarios side-by-side.
"""

    stress_code = """# Build monthly returns for assets and bubbles
asset_m = (1 + rets[asset_tickers].fillna(0)).resample(MONTHLY_FREQ).prod() - 1

if 'bubble_factor_returns' not in globals() or bubble_factor_returns is None or bubble_factor_returns.empty:
    raise RuntimeError('bubble_factor_returns missing/empty; run Section 3 first.')
bubble_factors = [c for c in ['AI bubble', 'Private Credit bubble', 'Crypto bubble'] if c in bubble_factor_returns.columns]
if not bubble_factors:
    bubble_factors = list(bubble_factor_returns.columns)

bubble_m = (1 + bubble_factor_returns[bubble_factors].reindex(rets.index).fillna(0.0)).resample(MONTHLY_FREQ).prod() - 1

# Fit per-asset OLS: asset ~ const + bubbles
betas = {}
resid_sigmas = {}

for a in asset_tickers:
    # Rename y so it never collides with a bubble column name (e.g., 'HYG').
    y_series = asset_m[a].rename('__asset__')
    df = pd.concat([y_series, bubble_m], axis=1, join='inner').dropna()
    if df.shape[0] < 36:
        continue
    y = df['__asset__']
    X = sm.add_constant(df[bubble_factors])
    fit = sm.OLS(y, X).fit()
    betas[a] = fit.params
    resid_sigmas[a] = float(fit.resid.std())

print('Factor models fit:', len(betas), 'of', len(asset_tickers), 'assets')

def monthly_path(series: pd.Series, start: str, end: str) -> pd.Series:
    s = series.loc[start:end].copy()
    if len(s) == 0:
        raise ValueError('Empty scenario path')
    return s

# Bubble factors (domain-level composites)
ai_bubble = 'AI bubble'
private_credit_bubble = 'Private Credit bubble'
crypto_bubble = 'Crypto bubble'


# NOTE: Historical analog scenarios are used only to source the shock sequences.
# We intentionally do NOT plot the historical analog equity curves here to keep this section
# focused on the forward-looking 2026–2030 stress test and to avoid extra images.

# -----------------------------
# Forward-looking stress test: 2026–2030
# -----------------------------

# Configure the burst placement inside the future window
FUTURE_BURST_START = pd.Timestamp('2027-06-30')  # change this if you want the burst to start earlier/later
SHOCK_MONTHS = 12  # truncate each historical analog shock to this many months for insertion

def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.to_period('M').to_timestamp(how='end')

future_start = month_end(FORECAST_START)
future_end = month_end(HORIZON_END)
future_idx = pd.date_range(future_start, future_end, freq=MONTHLY_FREQ)

needed = [ai_bubble, private_credit_bubble, crypto_bubble]
missing_needed = [x for x in needed if x not in bubble_m.columns]
if missing_needed:
    raise RuntimeError(f'Missing bubble columns for future scenario: {missing_needed}')

# Baseline future bubble-factor path: multivariate normal fit on recent history
train_start = month_end(pd.Timestamp.today().normalize() - pd.DateOffset(years=10))
train_bubbles = bubble_m[needed].loc[train_start:].dropna()
if train_bubbles.empty:
    train_bubbles = bubble_m[needed].dropna()

mu_f = train_bubbles.mean()
cov_f = train_bubbles.cov()

rng_future = np.random.default_rng(RANDOM_SEED)
base_samples = rng_future.multivariate_normal(mu_f.values, cov_f.values, size=len(future_idx))
base_factors = pd.DataFrame(base_samples, index=future_idx, columns=needed)

# Historical analog shock paths (truncate)
ai_shock_full = monthly_path(bubble_m[ai_bubble], *EVENT_WINDOWS['Dot-com crash (AI bubble stress)'])
credit_shock_full = monthly_path(bubble_m[private_credit_bubble], *EVENT_WINDOWS['GFC (Private Credit bubble stress)'])
crypto_shock_full = monthly_path(bubble_m[crypto_bubble], *EVENT_WINDOWS['Crypto winter (Crypto bubble stress)'])

ai_shock = ai_shock_full.iloc[: min(SHOCK_MONTHS, len(ai_shock_full))]
credit_shock = credit_shock_full.iloc[: min(SHOCK_MONTHS, len(credit_shock_full))]
crypto_shock = crypto_shock_full.iloc[: min(SHOCK_MONTHS, len(crypto_shock_full))]

burst_start = month_end(FUTURE_BURST_START)
if burst_start not in base_factors.index:
    # snap to nearest available month-end in the simulated index
    burst_start = base_factors.index[base_factors.index.get_indexer([burst_start], method='nearest')[0]]

def apply_shock(df: pd.DataFrame, col: str, shock: pd.Series, start: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    idx = pd.date_range(start, periods=len(shock), freq=MONTHLY_FREQ)
    idx = idx.intersection(df.index)
    if len(idx) == 0:
        return df
    df.loc[idx, col] = shock.values[: len(idx)]
    return df

# Build future scenarios
future_scenarios = {
    'Baseline (simulated 2026–2030)': base_factors,
    'AI bubble burst (2026–2030)': apply_shock(base_factors, ai_bubble, ai_shock, burst_start),
    'Private Credit bubble burst (2026–2030)': apply_shock(base_factors, private_credit_bubble, credit_shock, burst_start),
    'Crypto bubble burst (2026–2030)': apply_shock(base_factors, crypto_bubble, crypto_shock, burst_start),
}

triple_future = base_factors.copy()
triple_future = apply_shock(triple_future, ai_bubble, ai_shock, burst_start)
triple_future = apply_shock(triple_future, private_credit_bubble, credit_shock, burst_start)
triple_future = apply_shock(triple_future, crypto_bubble, crypto_shock, burst_start)
future_scenarios['Triple-bubble burst (2026–2030)'] = triple_future

# Predict portfolio returns under future scenarios
w = best_w.reindex(asset_tickers).fillna(0.0)
asset_mean_m = asset_m.mean()

def predict_portfolio_monthly(factors: pd.DataFrame) -> pd.Series:
    pred_assets = pd.DataFrame(index=factors.index, columns=asset_tickers, data=0.0)
    for a in asset_tickers:
        if a in betas:
            params = betas[a]
            const = float(params.get('const', 0.0))
            pred_assets[a] = const
            pred_assets[a] += float(params.get(ai_bubble, 0.0)) * factors[ai_bubble]
            pred_assets[a] += float(params.get(private_credit_bubble, 0.0)) * factors[private_credit_bubble]
            pred_assets[a] += float(params.get(crypto_bubble, 0.0)) * factors[crypto_bubble]
        else:
            # fallback for assets without factor model
            pred_assets[a] = float(asset_mean_m.get(a, 0.0))
    return pred_assets.fillna(0.0).dot(w)

def summarize_path(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    cum = float(eq.iloc[-1] - 1)
    peak = eq.cummax()
    mdd = float((eq / peak - 1.0).min())
    return {'cum_return': cum, 'max_drawdown': mdd}

future_results = []
future_eq = {}

for name, fx in future_scenarios.items():
    pr = predict_portfolio_monthly(fx)
    future_eq[name] = (1 + pr).cumprod()
    s = summarize_path(pr)
    s['scenario'] = name
    future_results.append(s)

# Plot injected triple-bubble factor path
plt.figure(figsize=(12, 4))
plt.plot(triple_future.index, triple_future[ai_bubble], label=f'AI bubble ({ai_bubble})')
plt.plot(triple_future.index, triple_future[private_credit_bubble], label=f'Private Credit bubble ({private_credit_bubble})')
plt.plot(triple_future.index, triple_future[crypto_bubble], label=f'Crypto bubble ({crypto_bubble})')
plt.axvline(burst_start, color='black', linestyle='--', alpha=0.6, label='burst start')
plt.title('Forward-looking bubble factors with injected triple-bubble shock (monthly) — 2026–2030')
plt.legend()
plt.tight_layout()
plt.show()

# Plot portfolio equity curves
plt.figure(figsize=(12, 5))
for name, eq in future_eq.items():
    plt.plot(eq.index, eq.values, label=name)
plt.axvline(burst_start, color='black', linestyle='--', alpha=0.6)
plt.title('Forward-looking stress test: portfolio equity curves (2026–2030)')
plt.ylabel('Equity (growth of $1)')
plt.legend()
plt.tight_layout()
plt.show()

future_table = pd.DataFrame(future_results).set_index('scenario').sort_values('cum_return')

# Additional graph requested: expected return if the bubbles burst (scenario comparison)
plt.figure(figsize=(10, 4))
plt.barh(future_table.index, (future_table['cum_return'] * 100.0))
plt.xlabel('Cumulative return (%)')
plt.title('Expected portfolio return by bubble-burst scenario (2026–2030)')
plt.tight_layout()
plt.show()

# Dynamic interpretation (updates with live Yahoo Finance pulls)
if 'explain' in globals():
    try:
        bullets = []
        st = future_table.copy()
        for col in ['cum_return', 'max_drawdown']:
            if col in st.columns:
                st[col] = pd.to_numeric(st[col], errors='coerce')

        cr = st['cum_return'].dropna() if 'cum_return' in st.columns else pd.Series(dtype=float)
        dd = st['max_drawdown'].dropna() if 'max_drawdown' in st.columns else pd.Series(dtype=float)

        if len(cr):
            worst_cr = cr.idxmin()
            best_cr = cr.idxmax()
            bullets.append(f'Worst cumulative return scenario: {worst_cr} ({float(cr.loc[worst_cr]):.2%})')
            bullets.append(f'Best cumulative return scenario: {best_cr} ({float(cr.loc[best_cr]):.2%})')

        if len(dd):
            worst_dd = dd.idxmin()
            bullets.append(f'Worst max drawdown scenario: {worst_dd} ({float(dd.loc[worst_dd]):.2%})')

        bullets.append('These scenarios are forward-looking 2026–2030 factor-path simulations; results update when live Yahoo factor history or the portfolio weights change.')
        explain('10) Forward-looking bubble-burst scenarios: quick read', bullets)
    except Exception:
        pass

future_table.style.format({'cum_return': '{:.2%}', 'max_drawdown': '{:.2%}'})
"""

    scenario_md = """## 10B) Hypothetical macro scenarios (GPFG-style) + Expected shortfall

We layer in the GPFG-style narratives (AI correction, fragmented world, regional debt crisis, extreme weather) and map them to **factor shocks** on the bubble proxies to estimate portfolio impact. We also report **expected shortfall (ES)** alongside cumulative return and max drawdown.

**How this differs from 10A:**
- 10A injects historical analog shock blocks into simulated future bubble paths.
- 10B applies calibrated shock magnitudes to the bubble factors directly (12 months by default) to mimic the GPFG scenario narratives.

We also compute **historical ES** for the portfolio (daily) for baseline context.
"""

    scenario_code = """# ---------- Helpers ----------
def expected_shortfall(r: pd.Series, alpha: float = 0.95) -> float:
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return np.nan
    q = r.quantile(1 - alpha)
    tail = r[r <= q]
    if len(tail) == 0:
        return float(q)
    return float(tail.mean())


# Baseline ES for the existing portfolio (daily returns)
hist_es = pd.DataFrame(
    {
        'metric': ['Portfolio ES (daily)'],
        'ES_95': [expected_shortfall(port, 0.95) if 'port' in globals() else np.nan],
        'ES_99': [expected_shortfall(port, 0.99) if 'port' in globals() else np.nan],
    }
)

display(hist_es.style.format({'ES_95': '{:.2%}', 'ES_99': '{:.2%}'}))


# ---------- GPFG-style factor shock scenarios (12-month shocks) ----------
# Shock magnitudes approximate GPFG table impacts (equities most negative in AI/fragmented; credit hit in fragmented/debt; mild positive credit in AI correction).
SCENARIO_SHOCKS = {
    'AI correction (12m)': {
        ai_bubble: -0.53,
        private_credit_bubble: 0.05,
        crypto_bubble: -0.25,
    },
    'Fragmented world (12m)': {
        ai_bubble: -0.49,
        private_credit_bubble: -0.15,
        crypto_bubble: -0.30,
    },
    'Regional debt crisis (12m)': {
        ai_bubble: -0.39,
        private_credit_bubble: -0.25,
        crypto_bubble: -0.10,
    },
    'Extreme weather (12m)': {
        ai_bubble: -0.24,
        private_credit_bubble: -0.10,
        crypto_bubble: -0.05,
    },
    'Combined severe (12m)': {
        ai_bubble: -0.55,
        private_credit_bubble: -0.25,
        crypto_bubble: -0.30,
    },
}


def build_custom_scenario(shocks: dict[str, float]) -> pd.DataFrame:
    df = base_factors.copy()
    seq_idx = pd.date_range(burst_start, periods=SHOCK_MONTHS, freq=MONTHLY_FREQ)
    for fac, val in shocks.items():
        if fac not in df.columns:
            continue
        # Interpret `val` as a 12-month total return and convert to a constant monthly path.
        monthly_val = (1.0 + float(val)) ** (1.0 / float(len(seq_idx))) - 1.0
        seq = pd.Series([monthly_val] * len(seq_idx), index=seq_idx)
        df = apply_shock(df, fac, seq, burst_start)
    return df


scenario_results: list[dict] = []
scenario_eq: dict[str, pd.Series] = {}

for scen_name, shocks in SCENARIO_SHOCKS.items():
    fx = build_custom_scenario(shocks)
    pr = predict_portfolio_monthly(fx)
    eq = (1 + pr).cumprod()
    scenario_eq[scen_name] = eq

    scenario_results.append(
        {
            'scenario': scen_name,
            'cum_return': float(eq.iloc[-1] - 1),
            'max_drawdown': summarize_path(pr)['max_drawdown'],
            'ES_95': expected_shortfall(pr, 0.95),
            'ES_99': expected_shortfall(pr, 0.99),
        }
    )


scenario_table = pd.DataFrame(scenario_results).set_index('scenario').sort_values('cum_return')

print('\nGPFG-style scenario results (12-month shocks on factors):')
display(
    scenario_table.style.format(
        {
            'cum_return': '{:.2%}',
            'max_drawdown': '{:.2%}',
            'ES_95': '{:.2%}',
            'ES_99': '{:.2%}',
        }
    )
)

# Dynamic interpretation (worst-case scenarios; updates with live Yahoo Finance pulls)
if 'explain' in globals():
    try:
        bullets = []
        st = scenario_table.copy()
        for col in ['cum_return', 'max_drawdown', 'ES_95', 'ES_99']:
            if col in st.columns:
                st[col] = pd.to_numeric(st[col], errors='coerce')

        cr = st['cum_return'].dropna() if 'cum_return' in st.columns else pd.Series(dtype=float)
        dd = st['max_drawdown'].dropna() if 'max_drawdown' in st.columns else pd.Series(dtype=float)
        es99 = st['ES_99'].dropna() if 'ES_99' in st.columns else pd.Series(dtype=float)

        if len(cr):
            worst_cr = cr.idxmin()
            best_cr = cr.idxmax()
            bullets.append(f'Worst cumulative return scenario: {worst_cr} ({float(cr.loc[worst_cr]):.2%})')
            bullets.append(f'Best cumulative return scenario: {best_cr} ({float(cr.loc[best_cr]):.2%})')

        if len(dd):
            worst_dd = dd.idxmin()
            bullets.append(f'Worst max drawdown scenario: {worst_dd} ({float(dd.loc[worst_dd]):.2%})')

        if len(es99):
            worst_es99 = es99.idxmin()
            bullets.append(f'Worst tail risk (ES_99): {worst_es99} ({float(es99.loc[worst_es99]):.2%})')

        if bullets:
            bullets.append('More negative ES/drawdown implies worse tail-loss risk under that narrative shock; if this is unacceptable, iterate the roster allocation (Section 5) and re-run.')
            explain('10B) GPFG-style scenarios: quick read', bullets)
    except Exception:
        pass


# Plot equity curves for the custom scenarios
plt.figure(figsize=(12, 5))
for name, eq in scenario_eq.items():
    plt.plot(eq.index, eq.values, label=name)
plt.axvline(burst_start, color='black', linestyle='--', alpha=0.6)
plt.title('GPFG-style scenario equity curves (2026–2030 window, 12m shocks)')
plt.ylabel('Equity (growth of $1)')
plt.legend()
plt.tight_layout()
plt.show()
"""

    summary_md = """## 11) Summary + knobs to tune

If you want to iterate quickly, the highest-leverage knobs are:
- `ROSTER_PATH` (your investable universe + positioning; edit the CSV to change symbols/weights/Long-Short/reasons)
- `BUBBLE_TICKERS` (your bubble factor tickers)
- `OPTIMIZATION_WINDOWS` and their exact dates
- `GROSS_EXPOSURE`, `MAX_ABS_WEIGHT`, `RANDOM_SEED`

If you update the roster CSV (tickers, Position, RiskLimitPercent, Rationale), the notebook will automatically pull fresh live Yahoo Finance data and re-run the full analysis pipeline.

### Simple Explanation of the Codebase, Output, how it's supporting the goal of resilience vs the AI / Private Credit / Crypto bubbles
- If you want to improve resilience, the highest-leverage step is changing the roster CSV: **what you hold, long vs short, and sizing**.
- Use Section 4 to check diversification/hedging, Section 7 to validate crisis behavior, and Section 10 to validate forward-looking bubble-burst scenarios.
- Iterate: adjust roster → re-run → see if downside improves under the bubble stress cases.
"""

    ews_md = """## 12) AI, Private Credit, Crypto Bubble Detection Prototype & Early Warning System

This is a **prototype early-warning dashboard** intended for corporate / stakeholder impact: *spotting elevated bubble risk early so risk can be managed proactively*.

### Model card (how to defend this to experts)

A simple way to defend this dashboard is to frame it as **Level + Trend + Drift + Evidence**:
- **Level** → `Probability Now (%)` (where risk is now)
- **Trend** → `30D Change (pp)` and `Trend` (how fast it’s moving)
- **Drift** → `Drift Flag` (any drift detector; Yes = significant upward shift vs baseline)
- **Evidence** → `Confidence` and `Backtest Confidence` (stability + lightweight historical alignment)

The 2026E–2030E **Mom/MR columns are scenarios, not predictions**:
- **Mom** (momentum) assumes recent trend continues (with decay).
- **MR** (mean reversion) assumes risk reverts toward a historical baseline (conservative anchor).

What this section does (live-data, no static datasets):
- Pulls **live market time-series** from Yahoo Finance via `yfinance` for representative tickers in each bubble domain.
- Computes **mathematical bubble indicators** (examples):
  - Price deviation from long-run trend (z-score)
  - Momentum / acceleration
  - Volatility clustering signals (short vol vs long vol)
  - Simple change-point flags
  - Hurst exponent (persistence vs mean reversion)
- Builds a **multi-signal composite bubble score** per domain, then fits a **4+ regime Markov-switching model (HMM-like)** and maps smoothed regime probabilities into a **0–100 bubble-likelihood** proxy over time.
- Pulls recent Yahoo Finance headlines (when available) and computes a lightweight **headline sentiment** score.

Outputs:
- A multi-domain **risk regime chart** over time
- A **current alert table** (Normal / Elevated / High) per domain
- A lightweight **historical validation table** (lead time + precision/recall) per domain
- A forward-looking **2026–2030 bubble-likelihood forecast** (prototype) + first projected threshold crossings

### Simple Explanation of the Codebase, Output, how it's supporting the goal of resilience vs the AI / Private Credit / Crypto bubbles
- This section does **not** change portfolio weights automatically. It provides **early-warning signals** so you (or stakeholders) can tighten risk limits, hedge, or reduce exposure **before** a bubble burst.
- If a domain’s bubble-likelihood stays high and rising, that’s a cue to revisit the roster (Section 1/5) and the scenario stress tests (Section 10).
"""

    ews_code = """# -----------------------------
# Section 12: Bubble Detection & Early Warning System (Prototype)
# -----------------------------

# NOTE: This is a prototype analytics tool. It uses live Yahoo Finance data via yfinance.
# Some optional data sources (e.g., deep social media sentiment) are not accessible without API keys;
# this section uses best-effort Yahoo headline data when available.

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# 12.1 Domain configuration
# -----------------------------

# Representative tickers per domain
# Policy: keep Section 12 aligned with the notebook-wide bubble ticker sets (Section 1).
# Use ONLY the symbols listed by the user (Yahoo Finance via yfinance, best-effort)
if 'BUBBLE_TICKERS' not in globals() or not isinstance(BUBBLE_TICKERS, dict) or not BUBBLE_TICKERS:
    raise RuntimeError('[ews] BUBBLE_TICKERS missing/empty; run Section 1 first.')
DOMAIN_TICKERS = {k: list(v) for k, v in BUBBLE_TICKERS.items()}

# Additional Yahoo-only leading indicators (macro + options/vol proxies + sector/style spreads)
# Default: enabled (best-effort). Set ENABLE_MACRO_SIGNALS=False to restrict the engine to DOMAIN_TICKERS only.
ENABLE_MACRO_SIGNALS = bool(globals().get('ENABLE_MACRO_SIGNALS', True))

MACRO_TICKERS: list[str] = []
if ENABLE_MACRO_SIGNALS:
    MACRO_TICKERS = [
        # Options / volatility proxies (Yahoo indices; best-effort availability)
        '^VIX',
        '^VIX9D',
        '^VIX3M',
        '^VIX6M',
        '^VVIX',

        # Rates / curve
        '^IRX',
        '^FVX',
        '^TNX',
        '^TYX',

        # Risk appetite / style spreads
        'SPY',
        'IWM',
        'QQQ',
        'ARKK',
        'SOXX',
        'XLK',
        'XLY',
        'XLP',
        'XLU',

        # Credit / duration proxies
        'HYG',
        'LQD',
        'TIP',
        'IEF',
    ]

# Pull prices live (Close). We reuse HIST_START from Section 2.
ALL_DETECT_TICKERS = sorted({t for ts in DOMAIN_TICKERS.values() for t in ts} | set(MACRO_TICKERS))

try:
    detect_prices
except NameError:
    detect_prices = pd.DataFrame()

if detect_prices is None or detect_prices.empty:
    detect_prices = download_close_prices(ALL_DETECT_TICKERS, start=HIST_START)

if detect_prices is None or detect_prices.empty:
    raise RuntimeError('[ews] No price data returned for detect tickers; rerun earlier sections or adjust DOMAIN_TICKERS.')

# Data coverage (deterministic): how many tickers per domain have recent usable prices in this run
domain_coverage = {}
try:
    asof_px = pd.to_datetime(detect_prices.index.max()) if (detect_prices is not None and not detect_prices.empty) else None
    for dom, tks in DOMAIN_TICKERS.items():
        tks = list(tks) if isinstance(tks, (list, tuple, set)) else []
        ok = 0
        for t in tks:
            if t not in detect_prices.columns:
                continue
            s = pd.to_numeric(detect_prices[t], errors='coerce')
            if s.dropna().shape[0] < 60:
                continue
            last = s.last_valid_index()
            if (asof_px is not None) and (last is not None) and ((asof_px - pd.to_datetime(last)).days <= 10):
                ok += 1
        domain_coverage[dom] = f"{ok}/{len(tks)}"
except Exception:
    domain_coverage = {}

# Volume is useful as a market-activity / flow proxy. Fetch best-effort volume for any tickers that support it.
def download_volume(tickers: list[str], start: str) -> pd.DataFrame:
    tickers = list(dict.fromkeys(tickers))
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by='column',
        threads=True,
    )
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=tickers)

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        if 'Volume' in lvl0:
            vol = raw['Volume'].copy()
        else:
            vol = pd.DataFrame(index=pd.to_datetime(raw.index))
    else:
        if len(tickers) == 1 and ('Volume' in raw.columns):
            vol = raw[['Volume']].rename(columns={'Volume': tickers[0]}).copy()
        else:
            vol = pd.DataFrame(index=pd.to_datetime(raw.index))

    vol.index = pd.to_datetime(vol.index)
    vol = vol.sort_index().resample('B').last()

    # Ensure every requested ticker exists as a column (yfinance may omit Volume for some symbols)
    for t in tickers:
        if t not in vol.columns:
            vol[t] = np.nan

    vol = vol[tickers].fillna(0.0)
    return vol


if ('detect_vol' not in globals()) or detect_vol is None or detect_vol.empty:
    detect_vol = download_volume(ALL_DETECT_TICKERS, start=HIST_START)

detect_vol = detect_vol.reindex(detect_prices.index).fillna(0.0)


# -----------------------------
# 12.2 Indicator functions
# -----------------------------

ZSCORE_WINDOW = 252 * 5  # ~5y rolling window; avoids full-sample lookahead bias


def zscore(s: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    s = s.astype(float)
    if window and window > 3:
        minp = max(30, int(window * 0.25))
        mu = s.rolling(window, min_periods=minp).mean()
        sd = s.rolling(window, min_periods=minp).std()
        z = (s - mu) / sd
        return z.replace([np.inf, -np.inf], np.nan)

    mu = float(s.mean())
    sd = float(s.std())
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd


def hurst_exponent(x: np.ndarray) -> float:
    # DFA-based Hurst estimate (more robust than R/S for finite samples).
    # We compute DFA-1 (linear detrending) on log returns; output is clipped to [0, 1].
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 252:
        return np.nan

    lx = np.log(np.clip(x, 1e-12, None))
    r = np.diff(lx)
    r = r[np.isfinite(r)]
    n = int(len(r))
    if n < 240:
        return np.nan

    r = r - float(np.mean(r))
    y = np.cumsum(r)

    s_min = 10
    s_max = max(20, int(n / 4))
    if s_max <= s_min + 1:
        return np.nan

    scales = np.unique(np.logspace(np.log10(s_min), np.log10(s_max), num=12).astype(int))
    scales = scales[(scales >= s_min) & (scales <= s_max)]

    Fs = []
    Ss = []
    t = None
    for s in scales:
        k = int(n // s)
        if k < 2:
            continue
        seg = y[: k * s].reshape(k, s)
        if t is None or len(t) != s:
            t = np.arange(s, dtype=float)

        rms = []
        for j in range(k):
            try:
                coef = np.polyfit(t, seg[j], deg=1)
                trend = coef[0] * t + coef[1]
                rms.append(float(np.sqrt(np.mean((seg[j] - trend) ** 2))))
            except Exception:
                continue

        if not rms:
            continue
        Fs.append(float(np.mean(rms)))
        Ss.append(int(s))

    if len(Fs) < 4:
        return np.nan

    F = np.asarray(Fs, dtype=float)
    S = np.asarray(Ss, dtype=float)
    mask = np.isfinite(F) & (F > 0) & np.isfinite(S) & (S > 0)
    if mask.sum() < 4:
        return np.nan

    alpha = float(np.polyfit(np.log(S[mask]), np.log(F[mask]), deg=1)[0])
    return float(np.clip(alpha, 0.0, 1.0))


def rolling_hurst(price: pd.Series, window: int = 252, step: int = 21) -> pd.Series:
    # Compute Hurst on a monthly-ish grid (every ~21 business days), then forward-fill.
    price = price.dropna()
    if len(price) < window:
        return pd.Series(index=price.index, dtype=float)
    out = []
    for i in range(window, len(price) + 1, step):
        sub = price.iloc[i - window : i]
        out.append((sub.index[-1], hurst_exponent(sub.values)))
    s = pd.Series({k: v for k, v in out}).sort_index()
    return s.reindex(price.index).ffill()


def rolling_hurst_robust(price: pd.Series, windows: list[int] | None = None, step: int = 21) -> pd.Series:
    # Robust Hurst: reduce window sensitivity by computing across multiple windows and taking the median.
    # This helps stabilize the indicator while still allowing longer-horizon Hurst to influence the signal.
    if windows is None:
        windows = [252, 252 * 2, 252 * 5]

    series = []
    for w in windows:
        try:
            series.append(rolling_hurst(price, window=int(w), step=int(step)).rename(f'hurst_{int(w)}'))
        except Exception:
            continue

    if not series:
        p = price.dropna()
        return pd.Series(index=p.index, dtype=float)

    df = pd.concat(series, axis=1)
    return df.median(axis=1, skipna=True)


# Stationarity tests (bubble regime proxy)
# - These are best-effort statistical probes (not definitive). High p-values indicate non-stationarity.
# - We compute them on a coarse grid (quarterly) to keep runtime reasonable.
STATIONARITY_WINDOW = 252 * 2
STATIONARITY_STEP = 63


def rolling_adf_pvalue(series: pd.Series, window: int = STATIONARITY_WINDOW, step: int = STATIONARITY_STEP) -> pd.Series:
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception:
        return pd.Series(index=series.index, dtype=float)

    s = pd.to_numeric(series, errors='coerce').astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < int(window):
        return pd.Series(index=series.index, dtype=float)

    out = []
    for i in range(int(window), len(s) + 1, int(step)):
        sub = s.iloc[i - int(window) : i]
        p = np.nan
        try:
            # Fast, fixed-lag ADF (we’re using this as a regime feature, not a publication-grade unit root test)
            res = adfuller(sub.to_numpy(dtype=float), maxlag=1, regression='c', autolag=None)
            p = float(res[1])
        except Exception:
            p = np.nan
        out.append((sub.index[-1], p))

    pv = pd.Series({k: v for k, v in out}).sort_index()
    return pv.reindex(series.index).ffill()


# Volatility clustering + jump proxies (jump-diffusion-ish features)
# We implement a fast GARCH(1,1)-style filter (fixed params) so we don't need extra deps.
GARCH_ALPHA = float(globals().get('GARCH_ALPHA', 0.05))
GARCH_BETA = float(globals().get('GARCH_BETA', 0.94))


def garch_sigma(r: pd.Series, alpha: float = GARCH_ALPHA, beta: float = GARCH_BETA) -> pd.Series:
    r = pd.to_numeric(r, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
    rr = r.fillna(0.0).to_numpy(dtype=float)
    if rr.size == 0:
        return pd.Series(index=r.index, dtype=float)

    alpha = float(np.clip(alpha, 1e-6, 0.20))
    beta = float(np.clip(beta, 1e-6, 0.999))
    if (alpha + beta) >= 0.999:
        beta = 0.999 - alpha

    var = float(np.nanvar(rr))
    var = var if np.isfinite(var) and var > 1e-12 else 1e-6
    omega = float(var * (1.0 - alpha - beta))
    omega = omega if np.isfinite(omega) and omega > 0 else 1e-9

    sig2 = np.empty_like(rr)
    sig2[0] = var
    for i in range(1, rr.size):
        sig2[i] = omega + alpha * (rr[i - 1] ** 2) + beta * sig2[i - 1]

    sig = np.sqrt(np.maximum(sig2, 1e-12))
    return pd.Series(sig, index=r.index)


def rolling_pp_pvalue(series: pd.Series, window: int = STATIONARITY_WINDOW, step: int = STATIONARITY_STEP) -> pd.Series:
    try:
        from statsmodels.tsa.stattools import phillips_perron
    except Exception:
        return pd.Series(index=series.index, dtype=float)

    s = pd.to_numeric(series, errors='coerce').astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < int(window):
        return pd.Series(index=series.index, dtype=float)

    out = []
    for i in range(int(window), len(s) + 1, int(step)):
        sub = s.iloc[i - int(window) : i]
        p = np.nan
        try:
            # Signature differs across statsmodels versions; keep this defensive.
            res = phillips_perron(sub.to_numpy(dtype=float))
            if isinstance(res, (tuple, list)) and len(res) >= 2:
                p = float(res[1])
        except Exception:
            p = np.nan
        out.append((sub.index[-1], p))

    pv = pd.Series({k: v for k, v in out}).sort_index()
    return pv.reindex(series.index).ffill()


def change_point_flag(x: pd.Series, short: int = 21, long: int = 252, z_thresh: float = 3.0) -> pd.Series:
    # Prototype change-point signal: z-score of short-mean vs long-mean.
    x = x.astype(float)
    mu_s = x.rolling(short).mean()
    mu_l = x.rolling(long).mean()
    sd_l = x.rolling(long).std()
    z = (mu_s - mu_l) / sd_l
    return (z.abs() > z_thresh).fillna(False)


def compute_indicators(price: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    price = price.dropna()
    r = price.pct_change()

    ma_long = price.rolling(252).mean()
    sd_long = price.rolling(252).std()
    z_price = (price - ma_long) / sd_long

    mom_1m = price.pct_change(21)
    mom_3m = price.pct_change(63)
    accel_1m = mom_1m - mom_3m

    vol_1m = r.rolling(21).std()
    vol_6m = r.rolling(126).std()
    vol_ratio = vol_1m / vol_6m

    # GARCH-style conditional volatility (volatility clustering proxy)
    sig = garch_sigma(r)
    sig_1m = sig.rolling(21).mean()
    sig_6m = sig.rolling(126).mean()
    garch_vol_ratio = sig_1m / sig_6m

    # Jump intensity proxy: fraction of days with |r| > k * sigma_conditional
    k_jump = float(globals().get('JUMP_SIGMA_K', 3.0) or 3.0)
    z = (r.abs() / (sig + 1e-12)).replace([np.inf, -np.inf], np.nan)
    jump_flag = (z > k_jump).astype(float)
    jump_rate_3m = jump_flag.rolling(63).mean()
    jump_mag_3m = z.where(jump_flag > 0).rolling(63).mean()

    dd_1y = price / price.rolling(252).max() - 1.0

    h = rolling_hurst_robust(price)
    adf_p = rolling_adf_pvalue(z_price)
    pp_p = rolling_pp_pvalue(z_price)
    cp = change_point_flag(r)

    out = pd.DataFrame(
        {
            'z_price': z_price,
            'mom_1m': mom_1m,
            'mom_3m': mom_3m,
            'accel_1m': accel_1m,
            'vol_ratio': vol_ratio,
            'garch_vol_ratio': garch_vol_ratio,
            'jump_rate_3m': jump_rate_3m,
            'jump_mag_3m': jump_mag_3m,
            'drawdown_1y': dd_1y,
            'hurst': h,
            'adf_p': adf_p,
            'pp_p': pp_p,
            'change_point': cp.astype(float),
        }
    )

    if volume is not None:
        v = volume.reindex(price.index).astype(float).fillna(0.0)
        out['z_volume'] = zscore(v)

        # Flow/activity proxies (NOT true ETF fund flows): signed-volume and OBV-style measures.
        signed = np.sign(r.fillna(0.0))
        signed_vol = signed * v
        denom = (v.rolling(21).sum() + 1e-12)
        out['flow_1m'] = (signed_vol.rolling(21).sum() / denom).clip(-1.0, 1.0)
        obv = signed_vol.cumsum()
        out['obv_1m'] = obv.diff(21)

    return out


# -----------------------------
# 12.2b Macro / cross-domain leading indicators (yfinance proxies + optional FRED)
# -----------------------------


def build_fred_raw(index: pd.Index, start: str) -> pd.DataFrame:
    # Best-effort FRED series fetch. Only runs if FRED_API_KEY is set in the environment.
    # This keeps the default workflow yfinance-only.
    import os
    import json
    import urllib.parse
    import urllib.request

    idx = pd.to_datetime(index)
    out = pd.DataFrame(index=idx)

    api_key = str(os.getenv('FRED_API_KEY', '') or '').strip()
    if not api_key:
        return out

    start = str(start or '').strip() or '2000-01-01'

    def _fetch_series(series_id: str) -> pd.Series:
        try:
            base = 'https://api.stlouisfed.org/fred/series/observations'
            qs = urllib.parse.urlencode(
                {
                    'series_id': str(series_id),
                    'api_key': api_key,
                    'file_type': 'json',
                    'observation_start': start,
                }
            )
            url = base + '?' + qs
            req = urllib.request.Request(url, headers={'User-Agent': 'bubble-ews/1.0'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode('utf-8', errors='ignore'))
            obs = payload.get('observations', []) if isinstance(payload, dict) else []

            pairs = []
            for r in obs:
                if not isinstance(r, dict):
                    continue
                dt = r.get('date', None)
                val = r.get('value', None)
                try:
                    dtx = pd.to_datetime(dt)
                except Exception:
                    continue
                try:
                    x = float(val)
                except Exception:
                    x = np.nan
                pairs.append((dtx, x))

            s = pd.Series({d: v for d, v in pairs}).sort_index()
            s = s.replace([np.inf, -np.inf], np.nan)
            return s
        except Exception:
            return pd.Series(dtype=float)

    # Funding / rates (levels)
    sof = _fetch_series('SOFR').rename('sofr')
    dff = _fetch_series('DFF').rename('dff')  # Effective Federal Funds Rate (daily)

    # Spreads / stress proxies
    ted = _fetch_series('TEDRATE').rename('ted_spread')
    hy = _fetch_series('BAMLH0A0HYM2').rename('hy_oas')  # ICE BofA US High Yield OAS
    ig = _fetch_series('BAMLC0A0CM').rename('ig_oas')    # ICE BofA US Corporate OAS

    if not sof.empty:
        out['sofr'] = sof
    if not dff.empty:
        out['dff'] = dff
    if ('sofr' in out.columns) and ('dff' in out.columns):
        out['sofr_ff_spread'] = out['sofr'] - out['dff']
    if not ted.empty:
        out['ted_spread'] = ted
    if not hy.empty:
        out['hy_oas'] = hy
    if not ig.empty:
        out['ig_oas'] = ig
    if ('hy_oas' in out.columns) and ('ig_oas' in out.columns):
        out['hy_ig_oas_spread'] = out['hy_oas'] - out['ig_oas']

    out = out.sort_index()
    out = out.reindex(idx).ffill()
    return out


def build_macro_features(prices: pd.DataFrame, fred: pd.DataFrame | None = None) -> pd.DataFrame:
    idx = prices.index
    macro = pd.DataFrame(index=idx)

    # Volatility proxy (VIX level)
    if '^VIX' in prices.columns:
        macro['vix_z'] = zscore(prices['^VIX'].astype(float))

    # Volatility-of-vol proxy (VVIX)
    if '^VVIX' in prices.columns:
        macro['vvix_z'] = zscore(prices['^VVIX'].astype(float))

    # Put/Call ratio proxy (CBOE equity put/call) — optional (Yahoo symbol ^CPC is often unavailable)
    if '^CPC' in prices.columns:
        macro['put_call_z'] = zscore(prices['^CPC'].astype(float))

    # VIX term structure (stress tends to invert the curve; ratios > 1 can indicate near-term panic)
    def _vix_ratio(a: str, b: str, name: str):
        if (a in prices.columns) and (b in prices.columns):
            r = (prices[a].astype(float) / prices[b].astype(float)).replace(0, np.nan)
            macro[name] = zscore(np.log(r))

    _vix_ratio('^VIX9D', '^VIX', 'vix_ts_9d_1m_z')
    _vix_ratio('^VIX', '^VIX3M', 'vix_ts_1m_3m_z')
    _vix_ratio('^VIX', '^VIX6M', 'vix_ts_1m_6m_z')

    # Yield curve slope proxies (Yahoo rate indexes)
    # Funding-rate proxy: short-end yields (level)
    if '^IRX' in prices.columns:
        macro['funding_rate_z'] = zscore(prices['^IRX'].astype(float))
    if ('^TNX' in prices.columns) and ('^IRX' in prices.columns):
        slope_10_3 = prices['^TNX'].astype(float) - prices['^IRX'].astype(float)
        macro['yc_slope_z'] = zscore(slope_10_3)  # 10Y-3M
    if ('^FVX' in prices.columns) and ('^IRX' in prices.columns):
        slope_5_3 = prices['^FVX'].astype(float) - prices['^IRX'].astype(float)
        macro['yc_slope_5_3_z'] = zscore(slope_5_3)  # 5Y-3M
    if ('^TYX' in prices.columns) and ('^IRX' in prices.columns):
        slope_30_3 = prices['^TYX'].astype(float) - prices['^IRX'].astype(float)
        macro['yc_slope_30_3_z'] = zscore(slope_30_3)  # 30Y-3M
    if ('^TYX' in prices.columns) and ('^TNX' in prices.columns):
        slope_30_10 = prices['^TYX'].astype(float) - prices['^TNX'].astype(float)
        macro['yc_slope_30_10_z'] = zscore(slope_30_10)  # 30Y-10Y

    # Credit risk proxy: HY vs IG (price ratio)
    if ('HYG' in prices.columns) and ('LQD' in prices.columns):
        ratio = (prices['HYG'].astype(float) / prices['LQD'].astype(float)).replace(0, np.nan)
        macro['credit_ratio_z'] = zscore(np.log(ratio))

    # Risk-on/off equity proxies
    def _log_ratio(a: str, b: str, name: str):
        if (a in prices.columns) and (b in prices.columns):
            r = (prices[a].astype(float) / prices[b].astype(float)).replace(0, np.nan)
            macro[name] = zscore(np.log(r))

    _log_ratio('XLY', 'XLP', 'xly_xlp_z')   # consumer disc vs staples
    _log_ratio('XLY', 'XLU', 'xly_xlu_z')   # disc vs utilities
    _log_ratio('XLK', 'XLU', 'xlk_xlu_z')   # tech vs utilities
    _log_ratio('IWM', 'SPY', 'iwm_spy_z')   # small vs large
    _log_ratio('ARKK', 'SPY', 'arkk_spy_z') # high beta/innovation vs broad mkt
    _log_ratio('SOXX', 'SPY', 'soxx_spy_z') # semis vs broad mkt
    _log_ratio('QQQ', 'SPY', 'qqq_spy_z')   # growth vs broad mkt

    # Duration / real-yield proxy (TIP vs IEF)
    _log_ratio('TIP', 'IEF', 'tip_ief_z')

    # Optional FRED series (activated only when FRED_API_KEY exists)
    if fred is not None and isinstance(fred, pd.DataFrame) and (not fred.empty):
        try:
            if 'sofr_ff_spread' in fred.columns:
                macro['sofr_ff_spread_z'] = zscore(pd.to_numeric(fred['sofr_ff_spread'], errors='coerce').reindex(idx).ffill())
            if 'ted_spread' in fred.columns:
                macro['ted_spread_z'] = zscore(pd.to_numeric(fred['ted_spread'], errors='coerce').reindex(idx).ffill())
            if 'hy_oas' in fred.columns:
                macro['hy_oas_z'] = zscore(pd.to_numeric(fred['hy_oas'], errors='coerce').reindex(idx).ffill())
            if 'hy_ig_oas_spread' in fred.columns:
                macro['hy_ig_oas_spread_z'] = zscore(pd.to_numeric(fred['hy_ig_oas_spread'], errors='coerce').reindex(idx).ffill())
        except Exception:
            pass

    return macro


FRED_RAW = pd.DataFrame(index=detect_prices.index)
try:
    FRED_RAW = build_fred_raw(detect_prices.index, start=str(HIST_START))
except Exception:
    FRED_RAW = pd.DataFrame(index=detect_prices.index)

MACRO = build_macro_features(detect_prices, fred=FRED_RAW)


# -----------------------------
# 12.3 1D Gaussian Mixture (2 components) via EM
# -----------------------------

def fit_gmm_1d(x: np.ndarray, n_iter: int = 50) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return {'ok': False}

    # Init
    mu1, mu2 = np.percentile(x, [30, 70])
    s1 = float(np.std(x)) if float(np.std(x)) > 1e-8 else 1.0
    s2 = s1
    w1 = 0.5
    w2 = 0.5

    def normpdf(xx, mu, sig):
        sig = max(float(sig), 1e-6)
        return np.exp(-0.5 * ((xx - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

    for _ in range(n_iter):
        p1 = w1 * normpdf(x, mu1, s1)
        p2 = w2 * normpdf(x, mu2, s2)
        denom = p1 + p2
        denom = np.where(denom == 0, 1e-12, denom)
        g1 = p1 / denom
        g2 = 1.0 - g1

        w1 = float(np.mean(g1))
        w2 = 1.0 - w1
        mu1 = float(np.sum(g1 * x) / np.sum(g1))
        mu2 = float(np.sum(g2 * x) / np.sum(g2))
        s1 = float(np.sqrt(np.sum(g1 * (x - mu1) ** 2) / np.sum(g1)))
        s2 = float(np.sqrt(np.sum(g2 * (x - mu2) ** 2) / np.sum(g2)))

    # Ensure component 2 is the "higher mean" component
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        s1, s2 = s2, s1
        w1, w2 = w2, w1

    return {'ok': True, 'mu_low': mu1, 'mu_high': mu2, 'sig_low': s1, 'sig_high': s2, 'w_low': w1, 'w_high': w2}


def gmm_high_posterior(x: np.ndarray, params: dict) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    def normpdf(xx, mu, sig):
        sig = max(float(sig), 1e-6)
        return np.exp(-0.5 * ((xx - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

    p_low = params['w_low'] * normpdf(x, params['mu_low'], params['sig_low'])
    p_high = params['w_high'] * normpdf(x, params['mu_high'], params['sig_high'])
    denom = p_low + p_high
    denom = np.where(denom == 0, 1e-12, denom)
    return p_high / denom


# -----------------------------
# 12.4 Per-domain composite score + bubble likelihood
# -----------------------------

def domain_composite_score(domain: str, tickers: list[str]) -> pd.Series:
    frames = []
    for t in tickers:
        p = detect_prices[t] if t in detect_prices.columns else pd.Series(dtype=float)
        v = detect_vol[t] if ('detect_vol' in globals() and t in detect_vol.columns) else None
        ind = compute_indicators(p.dropna(), volume=v)
        if ind.empty:
            continue

        # Normalize indicators into z-scores for combination
        # Note: drawdown is negative in drawdowns; for "bubble" (late-cycle) we focus on price stretch + momentum + vol.
        z = pd.DataFrame(index=ind.index)
        z['z_price'] = zscore(ind['z_price'].dropna()).reindex(ind.index)
        z['mom_1m'] = zscore(ind['mom_1m'].dropna()).reindex(ind.index)
        z['mom_3m'] = zscore(ind['mom_3m'].dropna()).reindex(ind.index)
        z['accel_1m'] = zscore(ind['accel_1m'].dropna()).reindex(ind.index)
        z['vol_ratio'] = zscore(ind['vol_ratio'].dropna()).reindex(ind.index)
        if 'garch_vol_ratio' in ind.columns:
            z['garch_vol_ratio'] = zscore(ind['garch_vol_ratio']).reindex(ind.index)
        if 'jump_rate_3m' in ind.columns:
            z['jump_rate_3m'] = zscore(ind['jump_rate_3m']).reindex(ind.index)
        if 'jump_mag_3m' in ind.columns:
            z['jump_mag_3m'] = zscore(ind['jump_mag_3m']).reindex(ind.index)
        z['hurst'] = zscore(ind['hurst'].dropna()).reindex(ind.index)
        if 'adf_p' in ind.columns:
            z['adf_nonstat'] = zscore(ind['adf_p']).reindex(ind.index)
        if 'pp_p' in ind.columns:
            z['pp_nonstat'] = zscore(ind['pp_p']).reindex(ind.index)
        z['cp'] = ind['change_point']
        if 'z_volume' in ind.columns:
            z['z_volume'] = zscore(ind['z_volume'].dropna()).reindex(ind.index)
        if 'flow_1m' in ind.columns:
            z['flow_1m'] = zscore(ind['flow_1m'].dropna()).reindex(ind.index)
        if 'obv_1m' in ind.columns:
            z['obv_1m'] = zscore(ind['obv_1m'].dropna()).reindex(ind.index)

        frames.append(z)

    if not frames:
        return pd.Series(dtype=float)

    # pandas>=3 removed DataFrame.groupby(axis=...). To average across the column MultiIndex
    # (tickers x features), transpose -> groupby on index level -> transpose back.
    tmp = pd.concat(frames, axis=1, keys=range(len(frames)))
    Z = tmp.T.groupby(level=1).mean().T

    # Join macro features (same for all domains; yfinance-only proxies)
    if 'MACRO' in globals() and isinstance(MACRO, pd.DataFrame) and (not MACRO.empty):
        Z = Z.join(MACRO.reindex(Z.index), how='left')

    # Weighted composite (multi-signal fusion; adjust weights as desired)
    w = {
        'z_price': 0.22,
        'mom_1m': 0.10,
        'mom_3m': 0.18,
        'accel_1m': 0.10,
        'vol_ratio': 0.12,
        'garch_vol_ratio': 0.05,
        'jump_rate_3m': 0.03,
        'jump_mag_3m': 0.02,
        'hurst': 0.08,
        'adf_nonstat': 0.03,
        'pp_nonstat': 0.02,
        'cp': 0.05,
        'flow_1m': 0.07,
        'obv_1m': 0.05,
        'z_volume': 0.05,

        # Macro proxies (signs matter)
        'vix_z': 0.04,
        'vvix_z': 0.02,
        'put_call_z': 0.02,
        'vix_ts_9d_1m_z': 0.02,
        'vix_ts_1m_3m_z': 0.02,
        'vix_ts_1m_6m_z': 0.01,
        'funding_rate_z': 0.01,
        'yc_slope_z': -0.04,          # inversion/flattening => higher risk
        'yc_slope_5_3_z': -0.02,
        'yc_slope_30_3_z': -0.02,
        'credit_ratio_z': -0.04,      # HY underperforming IG => higher risk
        'sofr_ff_spread_z': 0.02,     # funding dislocation => higher risk
        'ted_spread_z': 0.02,         # interbank stress proxy
        'hy_oas_z': 0.03,             # wider HY spreads => higher risk
        'hy_ig_oas_spread_z': 0.01,

        # Sector / style spreads (risk appetite)
        'xly_xlp_z': 0.03,
        'xly_xlu_z': 0.02,
        'xlk_xlu_z': 0.02,
        'iwm_spy_z': 0.02,
        'arkk_spy_z': 0.03,
        'soxx_spy_z': 0.03,
        'qqq_spy_z': 0.02,

        # Inflation expectations proxy
        'tip_ief_z': 0.01,
    }

    # --------------- SHAP explainability (exact for linear model) ---------------
    # For a weighted-sum model  score = Σ w_k · x_k  the exact Shapley value is:
    #   φ_k = w_k · (x_k − E[x_k])
    # where E[x_k] is the background mean of the z-scored feature.
    domain_shap_store = globals().setdefault('domain_shap_values', {})
    if 'domain_driver_summary' in globals():
        try:
            Z_fin = Z.fillna(0.0)
            bg_means = Z_fin.mean()
            last_dt = Z_fin.index.max()
            if last_dt is not None:
                row = Z_fin.loc[last_dt]
                shap_vals: dict[str, float] = {}
                for k, wk in w.items():
                    if k in row.index:
                        shap_vals[k] = float(wk * (float(row[k]) - float(bg_means.get(k, 0.0))))
                domain_shap_store[domain] = shap_vals
                # Top-5 by |SHAP value|
                top5 = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                if top5:
                    domain_driver_summary[domain] = '; '.join([f"{k}:{v:+.4f}" for k, v in top5])
        except Exception:
            pass

    score = pd.Series(0.0, index=Z.index)
    for k, wk in w.items():
        if k in Z.columns:
            score = score.add(wk * Z[k].fillna(0.0), fill_value=0.0)

    return score.rename(f'{domain}__score')


def markov_bubble_probability(score: pd.Series, k_regimes: int = 4) -> pd.Series:
    # Regime model (HMM-like): Markov switching regression on the composite score.
    # General mapping for k>=3: order regimes by mean(score) and blend all regime probs into 0..100.
    # This makes it easy to run 4+ regimes (more granular states) without changing downstream code.
    y = score.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(y) < 450:
        return pd.Series(index=score.index, dtype=float)

    # Standardize for numerical stability
    y_std = zscore(y).dropna()
    if len(y_std) < 350:
        return pd.Series(index=score.index, dtype=float)

    def _fit(k: int) -> pd.Series | None:
        try:
            mod = MarkovRegression(y_std, k_regimes=int(k), trend='c', switching_variance=True)
            res = mod.fit(disp=False)
        except Exception:
            return None

        probs = res.smoothed_marginal_probabilities
        if not isinstance(probs, pd.DataFrame):
            probs = pd.DataFrame(probs, index=y_std.index)

        yv = y_std.to_numpy(dtype=float)
        means = {}
        for col in probs.columns:
            w = probs[col].to_numpy(dtype=float)
            denom = float(np.nansum(w))
            means[col] = float(np.nansum(w * yv) / denom) if denom > 0 else -np.inf

        ordered = sorted(means, key=means.get)
        if len(ordered) < 2:
            return None

        # Blend probabilities with weights increasing from low->high regime.
        # For k=2 this reduces to P(high). For k=3 it becomes 0*P(low)+0.5*P(mid)+1*P(high).
        weights = np.linspace(0.0, 1.0, num=len(ordered), dtype=float)
        p = pd.Series(0.0, index=probs.index)
        for w_i, reg in zip(weights, ordered):
            p = p.add(float(w_i) * probs[reg], fill_value=0.0)
        p = p * 100.0

        return p.rename('bubble_likelihood')

    out = _fit(int(k_regimes))
    if out is None and int(k_regimes) != 2:
        out = _fit(2)
    if out is None:
        return pd.Series(index=score.index, dtype=float)

    return out.reindex(score.index)


def logistic_bubble_probability(score: pd.Series) -> pd.Series:
    # Conservative mapping from composite score -> probability-like risk (%)
    # Uses a rolling z-score then a logistic transform to avoid saturated 0/100 outputs.
    y = score.replace([np.inf, -np.inf], np.nan).astype(float)
    if len(y.dropna()) < 60:
        return pd.Series(index=score.index, dtype=float)

    w = 252 * 5
    minp = int(min(w, max(252, int(w * 0.6))))

    mu = y.rolling(w, min_periods=minp).mean()
    sd = y.rolling(w, min_periods=minp).std().replace(0.0, np.nan)
    z = (y - mu) / sd

    # Warmup: expanding stats to avoid all-NaN early on
    mu2 = y.expanding(min_periods=60).mean()
    sd2 = y.expanding(min_periods=60).std().replace(0.0, np.nan)
    z2 = (y - mu2) / sd2
    z = z.fillna(z2)

    # Bubble-zone center ~ +1σ, slope controls how quickly we move toward extremes
    z0 = 1.0
    k = 1.2
    p = 100.0 / (1.0 + np.exp(-float(k) * (z - float(z0))))
    return p.rename('bubble_likelihood')


MARKOV_K_REGIMES = int(globals().get('MARKOV_K_REGIMES', 4) or 4)
MARKOV_K_REGIMES = int(np.clip(MARKOV_K_REGIMES, 2, 8))
MARKOV_BLEND_WEIGHT = 0.35

# Wasserstein k-means (WK-means): distributional regime clustering on sliding return windows
# - Treat each window as an empirical distribution (sorted atoms / quantiles)
# - Distance: 1-Wasserstein between empirical measures with equal atoms => mean(|sorted_x - sorted_y|)
# - Centroid: 1-Wasserstein barycenter => coordinate-wise median (per-order statistic)
ENABLE_WKMEANS_REGIMES = bool(globals().get('ENABLE_WKMEANS_REGIMES', True))
WK_WINDOW_DAYS = int(globals().get('WK_WINDOW_DAYS', 63) or 63)
WK_STEP_DAYS = int(globals().get('WK_STEP_DAYS', 5) or 5)
WK_K = int(globals().get('WK_K', 4) or 4)
WK_MAX_ITERS = int(globals().get('WK_MAX_ITERS', 15) or 15)
WK_RANDOM_SEED = int(globals().get('WK_RANDOM_SEED', 42) or 42)
WK_VALIDATE_MMD = bool(globals().get('WK_VALIDATE_MMD', False))


def wkmeans_regime_probability(
    returns: pd.Series,
    k: int = 4,
    window: int = 63,
    step: int = 5,
    max_iters: int = 15,
    seed: int = 42,
    validate_mmd: bool = False,
) -> tuple[pd.Series, dict]:
    # WK-means on 1D empirical distributions (windows of returns).
    # Returns:
    #   - probability-like regime score in 0..100, indexed like returns (ffill)
    #   - metadata: centroids/labels/inertia + optional separation metrics

    meta: dict = {}
    r = pd.to_numeric(returns, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return pd.Series(index=returns.index, dtype=float), meta

    k = int(max(2, k))
    window = int(max(20, window))
    step = int(max(1, step))
    max_iters = int(max(1, max_iters))

    if len(r) < (window * 6):
        return pd.Series(index=returns.index, dtype=float), meta

    ends = []
    X = []
    w_mean = []
    for end in range(window - 1, len(r), step):
        sub = r.iloc[end - window + 1 : end + 1].to_numpy(dtype=float)
        if sub.shape[0] != window:
            continue
        sub = sub[np.isfinite(sub)]
        if sub.shape[0] != window:
            continue
        X.append(np.sort(sub))
        ends.append(r.index[end])
        w_mean.append(float(np.mean(sub)))

    if len(X) < (k * 3):
        return pd.Series(index=returns.index, dtype=float), meta

    X = np.asarray(X, dtype=float)  # shape M x window
    M = int(X.shape[0])
    rng = np.random.default_rng(int(seed))

    # init centroids: sample windows
    take = rng.choice(M, size=min(k, M), replace=False)
    cent = X[take, :].copy()
    if cent.shape[0] < k:
        # pad (rare) by repeating
        pad = np.repeat(cent[-1:, :], repeats=(k - cent.shape[0]), axis=0)
        cent = np.vstack([cent, pad])

    labels = np.zeros(M, dtype=int)

    def _dist_to_centroids(x: np.ndarray, C: np.ndarray) -> np.ndarray:
        # W1 for equal atoms: mean(|sorted_x - sorted_c|)
        return np.mean(np.abs(C - x[None, :]), axis=1)

    for _it in range(max_iters):
        # assign
        changed = False
        for i in range(M):
            d = _dist_to_centroids(X[i, :], cent)
            j = int(np.argmin(d))
            if labels[i] != j:
                labels[i] = j
                changed = True

        # update centroids via coordinate-wise median (W1 barycenter)
        new_cent = cent.copy()
        for j in range(k):
            idx = np.where(labels == j)[0]
            if idx.size == 0:
                new_cent[j, :] = X[int(rng.integers(0, M)), :]
            else:
                new_cent[j, :] = np.median(X[idx, :], axis=0)

        shift = float(np.mean(np.abs(new_cent - cent))) if np.isfinite(new_cent).all() else np.inf
        cent = new_cent
        if (not changed) or (np.isfinite(shift) and shift < 1e-6):
            break

    # Within-cluster dispersion (W1 distance to centroid)
    counts = {}
    within_mean_by_cluster = {}
    within_sum = 0.0
    within_n = 0
    for j in range(k):
        idx = np.where(labels == j)[0]
        counts[j] = int(idx.size)
        if idx.size:
            d = np.mean(np.abs(X[idx, :] - cent[j : j + 1, :]), axis=1)
            d = np.asarray(d, dtype=float)
            dj = float(np.mean(d)) if d.size else float('nan')
            within_mean_by_cluster[j] = dj
            within_sum += float(np.sum(d)) if d.size else 0.0
            within_n += int(d.size)
        else:
            within_mean_by_cluster[j] = float('nan')

    inertia = float(np.mean([float(v) for v in within_mean_by_cluster.values() if np.isfinite(v)])) if within_mean_by_cluster else float('nan')
    within_weighted = (float(within_sum) / float(within_n)) if within_n > 0 else float('nan')

    # Between-cluster dispersion: centroid-to-centroid W1 distances
    between_centroid_mean = float('nan')
    between_centroid_min = float('nan')
    separation_ratio = float('nan')
    try:
        if int(k) >= 2 and np.isfinite(cent).all():
            dcc = np.mean(np.abs(cent[:, None, :] - cent[None, :, :]), axis=2)
            off = dcc[~np.eye(int(k), dtype=bool)]
            if off.size:
                between_centroid_mean = float(np.mean(off))
                between_centroid_min = float(np.min(off))
                if np.isfinite(within_weighted) and within_weighted > 0:
                    separation_ratio = float(between_centroid_mean / within_weighted)
    except Exception:
        pass

    # Order regimes by mean return in the window (proxy for exuberance vs calm)
    means = {}
    for j in range(k):
        idx = np.where(labels == j)[0]
        if idx.size:
            means[j] = float(np.mean(np.asarray(w_mean, dtype=float)[idx]))
        else:
            means[j] = -np.inf
    ordered = sorted(means, key=means.get)
    weights = np.linspace(0.0, 1.0, num=len(ordered), dtype=float)
    w_by_cluster = {int(reg): float(wi) for wi, reg in zip(weights, ordered)}

    p_win = np.array([100.0 * float(w_by_cluster.get(int(lbl), 0.0)) for lbl in labels], dtype=float)
    p = pd.Series(p_win, index=pd.to_datetime(ends)).sort_index()
    p = p.reindex(returns.index).ffill().rename('wkmeans')

    centroid_quantiles = {}
    try:
        for j in range(k):
            c = np.asarray(cent[j, :], dtype=float)
            centroid_quantiles[int(j)] = {
                'q05': float(np.quantile(c, 0.05)),
                'q50': float(np.quantile(c, 0.50)),
                'q95': float(np.quantile(c, 0.95)),
            }
    except Exception:
        centroid_quantiles = {}

    meta = {
        'k': int(k),
        'window': int(window),
        'step': int(step),
        'n_windows': int(M),
        'cluster_counts': counts,
        'cluster_mean_return': {int(j): float(means[j]) for j in range(k)},
        'inertia_w1': float(inertia),
        'within_w1_mean_by_cluster': {int(j): float(within_mean_by_cluster.get(j, np.nan)) for j in range(k)},
        'within_w1_mean_weighted': float(within_weighted),
        'between_centroid_w1_mean': float(between_centroid_mean),
        'between_centroid_w1_min': float(between_centroid_min),
        'separation_ratio': float(separation_ratio),
        'centroid_quantiles': centroid_quantiles,
        'validate_mmd': bool(validate_mmd),
    }

    if bool(validate_mmd):
        try:
            def _mmd_rbf_1d(xa: np.ndarray, xb: np.ndarray, max_n: int = 400) -> float:
                xa = np.asarray(xa, dtype=float)
                xb = np.asarray(xb, dtype=float)
                xa = xa[np.isfinite(xa)]
                xb = xb[np.isfinite(xb)]
                if xa.size < 10 or xb.size < 10:
                    return float('nan')

                if xa.size > int(max_n):
                    xa = rng.choice(xa, size=int(max_n), replace=False)
                if xb.size > int(max_n):
                    xb = rng.choice(xb, size=int(max_n), replace=False)

                x = np.concatenate([xa, xb], axis=0)
                if x.size < 2:
                    return float('nan')

                pairs = int(min(2000, max(200, x.size * 10)))
                i1 = rng.integers(0, x.size, size=pairs)
                i2 = rng.integers(0, x.size, size=pairs)
                diff = np.abs(x[i1] - x[i2])
                diff = diff[(diff > 0) & np.isfinite(diff)]
                if diff.size:
                    sigma = float(np.median(diff))
                else:
                    sigma = float(np.nanstd(x))

                if not (np.isfinite(sigma) and sigma > 0):
                    sigma = 1.0
                s2 = float(sigma) ** 2
                if not (np.isfinite(s2) and s2 > 0):
                    return float('nan')

                Kxx = np.exp(-((xa[:, None] - xa[None, :]) ** 2) / (2.0 * s2))
                np.fill_diagonal(Kxx, 0.0)
                denom_x = float(xa.size * (xa.size - 1))
                kxx = float(Kxx.sum() / denom_x) if denom_x > 0 else float('nan')

                Kyy = np.exp(-((xb[:, None] - xb[None, :]) ** 2) / (2.0 * s2))
                np.fill_diagonal(Kyy, 0.0)
                denom_y = float(xb.size * (xb.size - 1))
                kyy = float(Kyy.sum() / denom_y) if denom_y > 0 else float('nan')

                Kxy = np.exp(-((xa[:, None] - xb[None, :]) ** 2) / (2.0 * s2))
                kxy = float(np.mean(Kxy)) if Kxy.size else float('nan')

                if not (np.isfinite(kxx) and np.isfinite(kyy) and np.isfinite(kxy)):
                    return float('nan')

                mmd2 = float(kxx + kyy - 2.0 * kxy)
                return float(max(0.0, mmd2))

            mmd_vals: list[float] = []
            for j1 in range(k):
                i1 = np.where(labels == j1)[0]
                if i1.size == 0:
                    continue
                s1 = X[i1, :].reshape(-1)
                for j2 in range(j1 + 1, k):
                    i2 = np.where(labels == j2)[0]
                    if i2.size == 0:
                        continue
                    s2v = X[i2, :].reshape(-1)
                    v = _mmd_rbf_1d(s1, s2v, max_n=400)
                    if np.isfinite(v):
                        mmd_vals.append(float(v))

            if mmd_vals:
                meta['mmd_rbf_mean'] = float(np.mean(mmd_vals))
                meta['mmd_rbf_min'] = float(np.min(mmd_vals))
                meta['mmd_rbf_max'] = float(np.max(mmd_vals))
        except Exception:
            pass

    return p, meta


domain_scores = {}
domain_probs = {}
domain_driver_summary = {}
domain_prob_components = {}
domain_wk_meta = {}


def _smooth_prob(_p: pd.Series) -> pd.Series:
    try:
        if _p is None or _p.dropna().empty:
            return _p
        return _p.clip(lower=0.0, upper=100.0).ewm(span=5, adjust=False).mean()
    except Exception:
        return _p

for domain, tickers in DOMAIN_TICKERS.items():
    s = domain_composite_score(domain, tickers)
    if s.empty:
        continue

    s_clean = s.dropna()
    short_series = len(s_clean) < 100

    # Component models for ensemble/BMA
    p_markov = pd.Series(index=s.index, dtype=float)
    p_log = logistic_bubble_probability(s)
    p_gmm = pd.Series(index=s.index, dtype=float)
    p_wk = pd.Series(index=s.index, dtype=float)
    p_fixed = pd.Series(index=s.index, dtype=float)

    # Optional distributional regimes (WK-means) on sliding windows of returns
    wk_meta = {}
    if bool(ENABLE_WKMEANS_REGIMES):
        try:
            r_wk = None
            try:
                if 'bubble_factor_returns' in globals() and isinstance(bubble_factor_returns, pd.DataFrame) and domain in bubble_factor_returns.columns:
                    r_wk = pd.to_numeric(bubble_factor_returns[domain], errors='coerce').astype(float)
            except Exception:
                r_wk = None
            if r_wk is None or r_wk.dropna().empty:
                # Fallback: use score increments (distribution of changes)
                r_wk = pd.to_numeric(s.diff(), errors='coerce').astype(float)

            p_wk, wk_meta = wkmeans_regime_probability(
                r_wk,
                k=int(WK_K),
                window=int(WK_WINDOW_DAYS),
                step=int(WK_STEP_DAYS),
                max_iters=int(WK_MAX_ITERS),
                seed=int(WK_RANDOM_SEED),
                validate_mmd=bool(WK_VALIDATE_MMD),
            )
            p_wk = p_wk.reindex(s.index)
        except Exception:
            p_wk = pd.Series(index=s.index, dtype=float)
            wk_meta = {}

    try:
        if isinstance(wk_meta, dict) and wk_meta:
            domain_wk_meta[domain] = wk_meta
    except Exception:
        pass

    if short_series:
        # Mitigation: for short series, avoid fitting Markov regimes; prefer a simple 1D GMM when possible.
        params = fit_gmm_1d(s_clean.to_numpy(), n_iter=60)
        if params.get('ok'):
            post = gmm_high_posterior(s_clean.to_numpy(), params)
            p_gmm = pd.Series(post * 100.0, index=s_clean.index).reindex(s.index)
            p_fixed = p_gmm.copy()
        else:
            p_fixed = p_log.copy()
    else:
        # Blend (markov captures regime shifts; logistic-z mapping avoids saturated 0/100 outputs)
        p_markov = markov_bubble_probability(s, k_regimes=MARKOV_K_REGIMES)
        if (not p_markov.dropna().empty) and (not p_log.dropna().empty):
            w = float(MARKOV_BLEND_WEIGHT)
            p_fixed = (w * p_markov + (1.0 - w) * p_log).where(p_markov.notna(), p_log)
        else:
            p_fixed = p_log if not p_log.dropna().empty else p_markov

    # Final fallback: simple 2-component 1D GMM (when everything else fails)
    if p_fixed.dropna().empty:
        params = fit_gmm_1d(s_clean.to_numpy(), n_iter=60)
        if not params.get('ok'):
            continue
        post = gmm_high_posterior(s_clean.to_numpy(), params)
        p_gmm = pd.Series(post * 100.0, index=s_clean.index).reindex(s.index)
        p_fixed = p_gmm.copy()

    # Light smoothing (reduces day-to-day jitter)
    p_fixed = _smooth_prob(p_fixed)
    p_markov = _smooth_prob(p_markov)
    p_log = _smooth_prob(p_log)
    p_gmm = _smooth_prob(p_gmm)
    p_wk = _smooth_prob(p_wk)

    # Save components for BMA (keys match later BMA_MODELS)
    domain_prob_components[domain] = {
        'markov': p_markov,
        'logistic': p_log,
        'gmm': p_gmm,
        'wkmeans': p_wk,
    }

    p = p_fixed

    p = p.rename(f'{domain}__bubble_likelihood')
    domain_scores[domain] = s
    domain_probs[domain] = p


# Combine into a single dashboard table
prob_df = pd.DataFrame(domain_probs).dropna(how='all')
score_df = pd.DataFrame(domain_scores).dropna(how='all')

if prob_df.empty:
    raise RuntimeError('Bubble detection: no domain probabilities could be computed. Check tickers/data availability.')


# -----------------------------
# 12.5 News & sentiment (Alpha Vantage preferred; Yahoo fallback)
# -----------------------------

import os
import json
import time
import hashlib
import requests


# -----------------------------
# 12.4A Ensemble + Bayesian Model Averaging (BMA) (lightweight)
# -----------------------------
# We keep the existing fixed Markov/logistic blend as a fallback, but optionally
# re-weight component models (markov/logistic/gmm/wkmeans) per-domain using historical
# validation against the labeled bubble-events DB.

ENABLE_BMA_ENSEMBLE = True
BMA_TEMPERATURE = 8.0
BMA_MODELS = ['markov', 'logistic', 'gmm', 'wkmeans']
BMA_LEAD_DAYS = 90
BMA_ROC_POINTS = 51

BMA_WEIGHTS_BY_DOMAIN = {}
BMA_MODEL_SCORES_BY_DOMAIN = {}


def _bma_softmax(scores: dict[str, float], temperature: float = 8.0) -> dict[str, float]:
    if not scores:
        return {}
    ks = list(scores.keys())
    x = np.array([float(scores[k]) for k in ks], dtype=float)
    # Center at 0.5 (random baseline)
    z = float(temperature) * (x - 0.5)
    z = z - float(np.max(z))
    w = np.exp(z)
    denom = float(np.sum(w))
    if (not np.isfinite(denom)) or denom <= 0:
        return {k: 1.0 / float(len(ks)) for k in ks}
    return {k: float(v) for k, v in zip(ks, (w / denom))}


def _bma_confusion_metrics(alert: pd.Series, y: pd.Series) -> dict:
    common = alert.index.intersection(y.index)
    ap = alert.reindex(common).fillna(0).astype(int)
    yp = y.reindex(common).fillna(0).astype(int)

    tp = int(((ap == 1) & (yp == 1)).sum())
    fp = int(((ap == 1) & (yp == 0)).sum())
    fn = int(((ap == 0) & (yp == 1)).sum())
    tn = int(((ap == 0) & (yp == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if (np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return {'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fpr, 'tpr': recall}


def _bma_auc_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    mask = np.isfinite(fpr) & np.isfinite(tpr)
    if mask.sum() < 2:
        return np.nan
    fpr = fpr[mask]
    tpr = tpr[mask]
    order = np.argsort(fpr)
    x = fpr[order]
    y = tpr[order]
    if len(x) < 2:
        return np.nan
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def _bma_load_events(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame()
    except Exception:
        df = pd.DataFrame()

    if not df.empty:
        for c in ['start_date', 'peak_date', 'end_date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors='coerce')
    return df


def _bma_events_for_domain(events_db: pd.DataFrame, domain: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if events_db is None or events_db.empty:
        return out

    if 'domains' not in events_db.columns:
        return out

    dom = str(domain or '').strip().lower()
    df = events_db[events_db['domains'].astype(str).str.lower().str.contains(dom, regex=False)]
    for _i, r in df.iterrows():
        start_dt = r.get('start_date', pd.NaT)
        end_dt = r.get('end_date', pd.NaT)
        if pd.isna(start_dt):
            start_dt = r.get('peak_date', pd.NaT)
        if pd.isna(end_dt):
            end_dt = r.get('peak_date', pd.NaT)
        if pd.isna(start_dt) or pd.isna(end_dt):
            continue
        label = r.get('event_name', None) or r.get('event_id', None) or 'event'
        out.append({'event': str(label)[:120], 'start_dt': pd.to_datetime(start_dt), 'end_dt': pd.to_datetime(end_dt)})
    out = sorted(out, key=lambda x: x.get('start_dt', pd.Timestamp.max))
    return out


def _bma_label_series(index: pd.DatetimeIndex, events: list[dict[str, object]], lead_days: int) -> pd.Series:
    y = pd.Series(0, index=index, dtype=int)
    for ev in events or []:
        try:
            start_dt = pd.to_datetime(ev.get('start_dt', pd.NaT), errors='coerce')
            end_dt = pd.to_datetime(ev.get('end_dt', pd.NaT), errors='coerce')
            if pd.isna(start_dt) or pd.isna(end_dt):
                continue
            lead_start = start_dt - pd.Timedelta(days=int(lead_days))
            y.loc[(y.index >= lead_start) & (y.index <= end_dt)] = 1
        except Exception:
            continue
    return y


def _bma_score_model(p: pd.Series, y: pd.Series, roc_points: int = 51) -> dict[str, float]:
    # Return {auc, best_f1} in [0,1] when possible.
    out = {'auc': np.nan, 'best_f1': np.nan}
    p = p.dropna().astype(float)
    if p.empty:
        return out

    thr_grid = np.linspace(0.0, 100.0, int(roc_points))
    best_f1 = -np.inf
    fprs = []
    tprs = []
    for thr in thr_grid:
        alert = (p >= float(thr)).astype(int)
        m = _bma_confusion_metrics(alert, y)
        fprs.append(m['fpr'])
        tprs.append(m['tpr'])
        if np.isfinite(m['f1']) and float(m['f1']) > float(best_f1):
            best_f1 = float(m['f1'])
    auc = _bma_auc_from_roc(np.array(fprs, dtype=float), np.array(tprs, dtype=float))
    out['auc'] = float(auc) if np.isfinite(auc) else np.nan
    out['best_f1'] = float(best_f1) if np.isfinite(best_f1) else np.nan
    return out


# Compute a BMA ensemble probability series (used by alerts + rebalancing)
try:
    if ENABLE_BMA_ENSEMBLE and (not prob_df.empty) and domain_prob_components:
        bma_path = str(globals().get('BUBBLE_EVENTS_PATH', 'bubble_events_database.csv') or 'bubble_events_database.csv')
        _bma_db = _bma_load_events(bma_path)

        # Only do BMA if we have a usable event DB with domains tagging.
        if (not _bma_db.empty) and ('domains' in _bma_db.columns):
            prob_ens = pd.DataFrame(index=prob_df.index)

            for domain in prob_df.columns:
                comps = domain_prob_components.get(domain, {})
                series_by_model = {}
                for m in BMA_MODELS:
                    s = comps.get(m, None)
                    if isinstance(s, pd.Series) and (not s.dropna().empty):
                        series_by_model[m] = s.reindex(prob_df.index)

                if not series_by_model:
                    prob_ens[domain] = prob_df[domain]
                    continue

                events = _bma_events_for_domain(_bma_db, domain)
                scores = {}
                for m, ps in series_by_model.items():
                    # Default score ~0.5 when labels are unavailable.
                    if not events:
                        scores[m] = 0.5
                        continue
                    p = ps.dropna().astype(float)
                    if p.empty:
                        scores[m] = 0.5
                        continue
                    y = _bma_label_series(p.index, events, BMA_LEAD_DAYS)
                    if int(y.sum()) == 0:
                        scores[m] = 0.5
                        continue
                    met = _bma_score_model(p, y, roc_points=BMA_ROC_POINTS)
                    auc = float(met.get('auc', np.nan))
                    f1 = float(met.get('best_f1', np.nan))
                    parts = []
                    if np.isfinite(auc):
                        parts.append(0.7 * auc)
                    if np.isfinite(f1):
                        parts.append(0.3 * f1)
                    scores[m] = float(np.clip(sum(parts), 0.0, 1.0)) if parts else 0.5

                weights = _bma_softmax(scores, temperature=BMA_TEMPERATURE)
                BMA_WEIGHTS_BY_DOMAIN[domain] = weights
                BMA_MODEL_SCORES_BY_DOMAIN[domain] = scores

                # Combine per-date, ignoring missing component values.
                num = None
                den = None
                for m, ps in series_by_model.items():
                    w = float(weights.get(m, 0.0))
                    if (not np.isfinite(w)) or w <= 0:
                        continue
                    part = ps.astype(float)
                    mask = part.notna()
                    if num is None:
                        num = (w * part.where(mask, 0.0)).copy()
                        den = (w * mask.astype(float)).copy()
                    else:
                        num = num.add(w * part.where(mask, 0.0), fill_value=0.0)
                        den = den.add(w * mask.astype(float), fill_value=0.0)

                if (num is None) or (den is None):
                    prob_ens[domain] = prob_df[domain]
                    continue

                p_ens = (num / den.replace(0.0, np.nan)).reindex(prob_df.index)
                p_ens = p_ens.where(p_ens.notna(), prob_df[domain])
                prob_ens[domain] = p_ens.clip(lower=0.0, upper=100.0)

            prob_df = prob_ens.dropna(how='all') if (not prob_ens.empty) else prob_df
except Exception:
    # Never let BMA break the dashboard; fall back to the fixed blend.
    pass

POS_WORDS = {
    'soars', 'surges', 'record', 'rally', 'bull', 'breakout', 'boom', 'explodes', 'beats', 'strong', 'upgrade'
}
NEG_WORDS = {
    'crash', 'plunge', 'fraud', 'lawsuit', 'hack', 'collapse', 'bubble', 'downgrade', 'selloff', 'risk', 'panic'
}

def headline_sentiment(title: str) -> float:
    words = [w.strip(".,:;!?()[]{}\\\"' ").lower() for w in title.split()]
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    return float((pos - neg) / max(len(words), 1))


def fetch_headlines(ticker: str, limit: int = 25) -> pd.DataFrame:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        items = []

    rows = []
    for it in items[:limit]:
        title = it.get('title', '')
        ts = it.get('providerPublishTime', None)
        dt = pd.to_datetime(ts, unit='s', utc=True, errors='coerce') if ts is not None else pd.NaT
        if not title:
            continue
        rows.append({'ticker': ticker, 'time': dt, 'title': title, 'sentiment': headline_sentiment(title)})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('time', ascending=False)
    return df


SENTIMENT_TICKERS = {
    'Crypto bubble': 'BTC-USD',
    'AI bubble': 'QQQ',
    'Private Credit bubble': 'HYG',
}

# Skip headline pulls by default (keeps Section 12 output clean + avoids flaky Yahoo news availability)
def _cache_load(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    return {}


def _cache_save(path: str, cache: dict) -> None:
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _cache_key(params: dict) -> str:
    # Do NOT include the API key in the cache key.
    safe = {k: v for k, v in params.items() if k != 'apikey'}
    raw = json.dumps(safe, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()


def fetch_alpha_vantage_news(params: dict, cache_path: str, ttl_s: int) -> dict:
    # Returns parsed JSON dict; may include 'error' key.
    ck = _cache_key(params)
    cache = _cache_load(cache_path)
    now = int(time.time())
    if ck in cache:
        ent = cache.get(ck) or {}
        ts = int(ent.get('ts', 0) or 0)
        if (now - ts) <= int(ttl_s) and isinstance(ent.get('data'), dict):
            return ent['data']

    url = 'https://www.alphavantage.co/query'
    try:
        r = requests.get(url, params=params, timeout=25)
        data = r.json() if r is not None else {}
    except Exception as e:
        return {'error': f'request_failed: {str(e)[:160]}'}

    # Rate limit / error surfaces
    if isinstance(data, dict):
        if data.get('Error Message'):
            return {'error': str(data.get('Error Message'))[:200]}
        if data.get('Note'):
            return {'error': str(data.get('Note'))[:200]}
        if data.get('Information'):
            return {'error': str(data.get('Information'))[:200]}

    # Save to cache (best-effort)
    try:
        cache[ck] = {'ts': now, 'data': data}
        _cache_save(cache_path, cache)
    except Exception:
        pass

    return data if isinstance(data, dict) else {'error': 'invalid_response'}


def _av_time_from(hours_back: int) -> str:
    # Format: YYYYMMDDTHHMM (UTC)
    try:
        # Round "now" to the hour so reruns within the cache TTL reuse the same cache key.
        now = pd.Timestamp.utcnow().floor('1H')
        dt = now - pd.Timedelta(hours=int(hours_back))
        return dt.strftime('%Y%m%dT%H%M')
    except Exception:
        return ''


def _news_metrics_from_feed(feed: list) -> dict:
    # Produce compact numeric features + a few headline strings.
    rows = []
    for it in feed or []:
        if not isinstance(it, dict):
            continue
        title = str(it.get('title', '') or '')
        score = it.get('overall_sentiment_score', None)
        try:
            score = float(score)
        except Exception:
            score = np.nan
        tpub = str(it.get('time_published', '') or '')
        rows.append({'title': title, 'score': score, 'time_published': tpub})

    if not rows:
        return {
            'articles_24h': 0,
            'sentiment_mean': np.nan,
            'neg_share': np.nan,
            'heat_0_100': 0.0,
            'top_titles': [],
        }

    df = pd.DataFrame(rows)
    s = pd.to_numeric(df['score'], errors='coerce')
    mean = float(s.mean()) if s.notna().any() else np.nan
    neg = float((s < -0.10).mean()) if s.notna().any() else np.nan
    n = int(len(df))

    # Saturating heat score: 0 at 0 articles, ~63 at 20, ~86 at 40, → 100 as n grows.
    heat = float(100.0 * (1.0 - np.exp(-n / 20.0)))

    top = [t for t in df['title'].tolist() if isinstance(t, str) and t.strip()]
    top = [t.strip()[:140] for t in top[:3]]

    return {
        'articles_24h': n,
        'sentiment_mean': mean,
        'neg_share': neg,
        'heat_0_100': heat,
        'top_titles': top,
    }


# Build per-domain news table (Alpha Vantage preferred). We keep Yahoo headline sentiment as a fallback.
news_table = pd.DataFrame([])

if RUN_SENTIMENT:
    def _read_alpha_key_from_file(path: str) -> str:
        # Optional local fallback for environments where env vars don't propagate into the kernel.
        # Expected format: first non-empty, non-comment line is the key.
        try:
            if path and os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in (f.read() or '').splitlines():
                        s = (line or '').strip()
                        if s and (not s.startswith('#')):
                            return s
        except Exception:
            pass
        return ''

    api_key = os.getenv('ALPHAVANTAGE_API_KEY', '').strip()
    api_key_source = 'env' if api_key else 'missing'
    key_path = ''
    if not api_key:
        key_path = os.getenv('ALPHAVANTAGE_API_KEY_FILE', '').strip()
        if not key_path:
            key_path = '.alphavantage_api_key.txt'
            if not os.path.exists(key_path):
                alt = 'alphavantage_api_key.txt'
                if os.path.exists(alt):
                    key_path = alt
        api_key = _read_alpha_key_from_file(key_path).strip()
        if api_key:
            api_key_source = f'file:{key_path}'

    if api_key_source.startswith('file:'):
        print(f"[news] Alpha Vantage API key loaded from local file fallback: {key_path}")
    cache_path = os.getenv('ALPHAVANTAGE_NEWS_CACHE_PATH', '.alphavantage_news_cache.json').strip() or '.alphavantage_news_cache.json'
    cache_minutes = int(os.getenv('ALPHAVANTAGE_NEWS_CACHE_MINUTES', '30') or 30)
    ttl_s = int(max(60, cache_minutes * 60))
    lookback_h = int(os.getenv('ALPHAVANTAGE_NEWS_LOOKBACK_HOURS', '24') or 24)
    limit = int(os.getenv('ALPHAVANTAGE_NEWS_LIMIT', '50') or 50)

    # Domain-specific query hints (tickers + topics) to keep calls stable.
    # Note: Alpha Vantage supports special prefixes like CRYPTO:BTC.
    domain_queries = {
        'AI bubble': {
            'tickers': 'QQQ,ROBT,WTAI',
            'topics': 'technology',
        },
        'Private Credit bubble': {
            'tickers': 'HYG,VPC',
            'topics': 'financial_markets,finance',
        },
        'Crypto bubble': {
            'tickers': 'CRYPTO:BTC',
            'topics': 'blockchain',
        },
    }

    av_rows = []
    if not api_key:
        print('[news] Alpha Vantage API key not set. Set env var ALPHAVANTAGE_API_KEY (or create .alphavantage_api_key.txt / alphavantage_api_key.txt, or set ALPHAVANTAGE_API_KEY_FILE) to enable live market news & sentiment.')
        for domain in domain_queries.keys():
            av_rows.append(
                {
                    'domain': domain,
                    'news_articles_24h': 0,
                    'news_sentiment_mean': np.nan,
                    'news_neg_share': np.nan,
                    'news_heat_0_100': 0.0,
                    'news_top_titles': [],
                    'news_error': 'missing_api_key',
                }
            )
    else:
        for domain, q in domain_queries.items():
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': q.get('tickers', ''),
                'topics': q.get('topics', ''),
                'time_from': _av_time_from(lookback_h),
                'sort': 'LATEST',
                'limit': str(limit),
                'apikey': api_key,
            }

            data = fetch_alpha_vantage_news(params, cache_path=cache_path, ttl_s=ttl_s)
            feed = data.get('feed', []) if isinstance(data, dict) else []
            m = _news_metrics_from_feed(feed if isinstance(feed, list) else [])
            av_rows.append(
                {
                    'domain': domain,
                    'news_articles_24h': int(m['articles_24h']),
                    'news_sentiment_mean': float(m['sentiment_mean']) if np.isfinite(m['sentiment_mean']) else np.nan,
                    'news_neg_share': float(m['neg_share']) if np.isfinite(m['neg_share']) else np.nan,
                    'news_heat_0_100': float(m['heat_0_100']),
                    'news_top_titles': m['top_titles'],
                    'news_error': str(data.get('error', ''))[:200] if isinstance(data, dict) else 'invalid_response',
                }
            )

    news_table = pd.DataFrame(av_rows)
    if (not news_table.empty) and ('domain' in news_table.columns):
        news_table = news_table.set_index('domain')


# Yahoo headline sentiment fallback (best-effort)
# - By default we aggregate across *all* domain tickers to avoid "0 headlines" when a single symbol has no items.
use_yahoo = bool(globals().get('RUN_YAHOO_HEADLINES', True))
use_all_tickers = bool(globals().get('YAHOO_HEADLINES_USE_ALL_DOMAIN_TICKERS', True))

limit_per_ticker = int(globals().get('YAHOO_NEWS_LIMIT_PER_TICKER', 10) or 10)
try:
    limit_per_ticker = int(os.getenv('YAHOO_NEWS_LIMIT_PER_TICKER', str(limit_per_ticker)) or limit_per_ticker)
except Exception:
    pass

if not use_yahoo:
    yahoo_by_domain = {}
else:
    if use_all_tickers and isinstance(globals().get('DOMAIN_TICKERS', None), dict) and bool(DOMAIN_TICKERS):
        yahoo_by_domain = {k: list(v) for k, v in DOMAIN_TICKERS.items()}
    else:
        yahoo_by_domain = {k: [v] for k, v in (SENTIMENT_TICKERS or {}).items()}

sent_rows = []
for domain, tickers in yahoo_by_domain.items():
    t_list = [t for t in (tickers or []) if isinstance(t, str) and t.strip()]
    dfs = []
    for t in t_list:
        df = fetch_headlines(t, limit=limit_per_ticker)
        if df is not None and (not df.empty):
            dfs.append(df)

    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        # Deduplicate headlines that appear across multiple tickers
        try:
            all_df = all_df.dropna(subset=['title'])
            all_df = all_df.drop_duplicates(subset=['title'])
        except Exception:
            pass

        mean_sent = float(all_df['sentiment'].mean()) if all_df.get('sentiment') is not None and all_df['sentiment'].notna().any() else np.nan
        n = int(len(all_df))
    else:
        mean_sent = np.nan
        n = 0

    sent_rows.append(
        {
            'domain': domain,
            'tickers': ','.join(t_list),
            'headline_sentiment_mean': mean_sent,
            'headlines': int(n),
        }
    )

sent_table = pd.DataFrame(sent_rows)
if (not sent_table.empty) and ('domain' in sent_table.columns):
    sent_table = sent_table.set_index('domain')
if use_yahoo and (not SHOW_ONLY_SIMPLE_EWS_TABLE) and (not sent_table.empty):
    display(sent_table)


# -----------------------------
# 12.6 Risk regimes + alerts
# -----------------------------

# We derive Normal/Elevated/High from historical percentiles per domain so thresholds adapt
# to each domain’s own history ("historical thresholds").
domain_thresholds = {}

# Dynamic threshold calibration (institutional-style): default to rolling quantiles.
THRESH_METHOD = 'rolling'          # 'rolling' or 'full_history'
THRESH_WINDOW_DAYS = 252 * 5       # ~5y
THRESH_ELEV_Q = 0.80
THRESH_HIGH_Q = 0.95

# Hoeffding-bound drift early-warning (no new tickers; operates on the probability series itself)
HOEFFDING_ALPHA = 0.01
HOEFFDING_RECENT_DAYS = 21         # ~1 trading month
HOEFFDING_BASELINE_DAYS = 252 * 3  # ~3y baseline (excluding recent window)

# Page-Hinkley drift detector (change-point detection on the probability series)
PH_DELTA = 0.10
PH_LAMBDA = 5.0
PH_MAX_DAYS = 252 * 5

# CUSUM drift detector (standardized; baseline vs recent)
CUSUM_K = 0.50
CUSUM_H = 5.0
CUSUM_RECENT_DAYS = 21
CUSUM_BASELINE_DAYS = 252 * 3

# ADWIN drift detector (adaptive window; simplified ADWIN-style split test)
ADWIN_DELTA = 0.002
ADWIN_MAX_DAYS = 252 * 5
ADWIN_MIN_WINDOW = 40

# Bootstrap confidence interval for the current probability (block bootstrap)
BOOT_CI_ENABLE = True
BOOT_CI_WINDOW_DAYS = 252 * 3
BOOT_CI_N = 150
BOOT_CI_BLOCK = 5
BOOT_CI_Q_LO = 0.05
BOOT_CI_Q_HI = 0.95
BOOT_CI_RANDOM_SEED = 42

# Granger causality validation: does bubble probability lead forward drawdowns of the domain proxy?
# Best-effort only; p-values can be sensitive to overlapping horizons / non-stationarity.
GRANGER_ENABLE = bool(globals().get('GRANGER_ENABLE', True))
GRANGER_HORIZON_DAYS = int(globals().get('GRANGER_HORIZON_DAYS', 63) or 63)
GRANGER_MAX_LAG = int(globals().get('GRANGER_MAX_LAG', 20) or 20)
GRANGER_ALPHA = float(globals().get('GRANGER_ALPHA', 0.05) or 0.05)


def _domain_proxy_price_for_granger(domain: str) -> pd.Series:
    # Prefer domain-level composite bubble factor prices (Section 3), else fall back to the first domain ticker.
    try:
        if 'bubble_factor_prices' in globals() and isinstance(bubble_factor_prices, pd.DataFrame) and domain in bubble_factor_prices.columns:
            return pd.to_numeric(bubble_factor_prices[domain], errors='coerce').astype(float)
    except Exception:
        pass

    try:
        tks = DOMAIN_TICKERS.get(domain, []) if 'DOMAIN_TICKERS' in globals() else []
        t0 = str(tks[0]) if isinstance(tks, (list, tuple)) and len(tks) else ''
        if t0 and ('detect_prices' in globals()) and isinstance(detect_prices, pd.DataFrame) and (t0 in detect_prices.columns):
            return pd.to_numeric(detect_prices[t0], errors='coerce').astype(float)
    except Exception:
        pass

    return pd.Series(dtype=float)


def _forward_min_return(price: pd.Series, horizon: int) -> pd.Series:
    # Forward-looking worst return in the next horizon (min_{t<tau<=t+h} price_tau/price_t - 1).
    p = pd.to_numeric(price, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    H = int(max(1, horizon))
    if len(p) < (H + 60):
        return pd.Series(index=price.index, dtype=float)

    arr = p.to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    n = int(len(arr))
    for i in range(n):
        j0 = i + 1
        j1 = min(n, i + H + 1)
        if j1 <= j0:
            continue
        mn = float(np.nanmin(arr[j0:j1]))
        den = float(arr[i])
        if np.isfinite(mn) and np.isfinite(den) and den != 0.0:
            out[i] = (mn / den) - 1.0

    return pd.Series(out, index=p.index).reindex(price.index)


def _granger_prob_leads_drawdown(prob: pd.Series, proxy_price: pd.Series, horizon_days: int, max_lag: int) -> dict:
    # Tests whether past probability values help predict forward drawdowns (best-effort heuristic).
    out = {
        'ok': False,
        'p_min': np.nan,
        'best_lag': np.nan,
        'n_obs': 0,
        'horizon_days': int(horizon_days),
        'max_lag_used': np.nan,
    }

    try:
        px = pd.to_numeric(proxy_price, errors='coerce').astype(float)
        if px.dropna().empty:
            return out

        y = _forward_min_return(px, horizon=int(horizon_days)).rename('fwd_min_ret')
        x = (pd.to_numeric(prob, errors='coerce').astype(float) / 100.0).rename('prob_0_1')
        df = pd.concat([y, x], axis=1).dropna()
        out['n_obs'] = int(len(df))
        if len(df) < 200:
            return out

        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except Exception:
            return out

        max_lag = int(max(1, max_lag))
        max_lag_use = int(min(max_lag, 30, max(1, len(df) // 10)))
        if len(df) < (max_lag_use + 50):
            max_lag_use = int(max(1, min(max_lag_use, len(df) // 15)))
        if max_lag_use < 1:
            return out

        data = df[['fwd_min_ret', 'prob_0_1']].to_numpy(dtype=float)
        try:
            import io
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                res = grangercausalitytests(data, maxlag=max_lag_use)
        except Exception:
            res = grangercausalitytests(data, maxlag=max_lag_use)

        p_by_lag = {}
        for lag in range(1, max_lag_use + 1):
            try:
                p = float(res[lag][0]['ssr_ftest'][1])
            except Exception:
                p = np.nan
            p_by_lag[int(lag)] = float(p) if np.isfinite(p) else np.nan

        finite = {lag: p for lag, p in p_by_lag.items() if np.isfinite(p)}
        if finite:
            best_lag = min(finite, key=finite.get)
            out['ok'] = True
            out['p_min'] = float(finite[best_lag])
            out['best_lag'] = int(best_lag)
        out['max_lag_used'] = int(max_lag_use)
        out['p_by_lag'] = p_by_lag
    except Exception:
        return out

    return out


def _hoeffding_drift(series: pd.Series):
    s = series.dropna().astype(float)
    if len(s) < (int(HOEFFDING_RECENT_DAYS) + 60):
        return False, np.nan

    recent = s.tail(int(HOEFFDING_RECENT_DAYS))
    baseline = s.iloc[: -int(HOEFFDING_RECENT_DAYS)].tail(int(HOEFFDING_BASELINE_DAYS))
    if baseline.empty or len(recent) < max(10, int(HOEFFDING_RECENT_DAYS * 0.6)):
        return False, np.nan

    # Work in 0..1 range
    mu1 = float(recent.mean()) / 100.0
    mu0 = float(baseline.mean()) / 100.0
    n = float(len(recent))
    eps = float(np.sqrt(np.log(1.0 / float(HOEFFDING_ALPHA)) / (2.0 * n))) if n > 0 else np.nan
    drift = float(mu1 - mu0) if (np.isfinite(mu1) and np.isfinite(mu0)) else np.nan
    score = float(drift / eps) if (np.isfinite(drift) and np.isfinite(eps) and eps > 0) else np.nan
    flag = bool(np.isfinite(drift) and np.isfinite(eps) and (drift > eps))
    return flag, score


def _page_hinkley_drift(series: pd.Series):
    # Simple Page-Hinkley on the probability series.
    s = series.dropna().astype(float).tail(int(PH_MAX_DAYS))
    if len(s) < 120:
        return False, np.nan

    mean = 0.0
    cum = 0.0
    min_cum = 0.0
    best = 0.0
    for i, x in enumerate(s.to_numpy(dtype=float), start=1):
        mean += (float(x) - mean) / float(i)
        cum += (float(x) - mean - float(PH_DELTA))
        min_cum = float(min(min_cum, cum))
        best = float(cum - min_cum)
        if np.isfinite(best) and best > float(PH_LAMBDA):
            return True, best

    return bool(np.isfinite(best) and best > float(PH_LAMBDA)), float(best)


def _cusum_drift(series: pd.Series):
    s = series.dropna().astype(float)
    if len(s) < (int(CUSUM_RECENT_DAYS) + 60):
        return False, np.nan

    recent = s.tail(int(CUSUM_RECENT_DAYS))
    baseline = s.iloc[: -int(CUSUM_RECENT_DAYS)].tail(int(CUSUM_BASELINE_DAYS))
    if baseline.empty or len(recent) < max(10, int(CUSUM_RECENT_DAYS * 0.6)):
        return False, np.nan

    mu0 = float(baseline.mean())
    sd0 = float(baseline.std())
    if (not np.isfinite(mu0)) or (not np.isfinite(sd0)) or sd0 <= 1e-9:
        return False, np.nan

    # One-sided upward-drift CUSUM
    pos = 0.0
    for x in recent.to_numpy(dtype=float):
        z = (float(x) - mu0) / sd0
        pos = max(0.0, float(pos + z - float(CUSUM_K)))

    flag = bool(np.isfinite(pos) and pos > float(CUSUM_H))
    return flag, float(pos)


def _adwin_drift(series: pd.Series):
    # ADWIN-style detector: scan a recent window and look for an upward mean shift that
    # exceeds a Bernstein-like bound. Returns (flag, score).
    s = series.dropna().astype(float).tail(int(ADWIN_MAX_DAYS))
    if len(s) < (2 * int(ADWIN_MIN_WINDOW)):
        return False, np.nan

    x = s.to_numpy(dtype=float)
    n = int(len(x))
    if n < (2 * int(ADWIN_MIN_WINDOW)):
        return False, np.nan

    # Cumulative sums for O(1) split statistics.
    csum = np.cumsum(x)
    csum2 = np.cumsum(x * x)

    best_margin = -np.inf
    best_diff = np.nan
    best_eps = np.nan

    delta = float(ADWIN_DELTA)
    if (not np.isfinite(delta)) or delta <= 0.0 or delta >= 1.0:
        delta = 0.002
    log_term = float(np.log(2.0 / delta))

    minw = int(ADWIN_MIN_WINDOW)
    for cut in range(minw, n - minw + 1):
        n0 = int(cut)
        n1 = int(n - cut)
        if n0 <= 1 or n1 <= 1:
            continue

        sum0 = float(csum[cut - 1])
        sum1 = float(csum[-1] - csum[cut - 1])
        mu0 = sum0 / float(n0)
        mu1 = sum1 / float(n1)

        diff = float(mu1 - mu0)  # upward shift only

        ss0 = float(csum2[cut - 1])
        ss1 = float(csum2[-1] - csum2[cut - 1])
        var0 = max(0.0, (ss0 / float(n0)) - mu0 * mu0)
        var1 = max(0.0, (ss1 / float(n1)) - mu1 * mu1)
        var = float(max(var0, var1))

        inv = (1.0 / float(n0)) + (1.0 / float(n1))
        eps = float(np.sqrt(2.0 * var * log_term * inv) + (2.0 / 3.0) * log_term * inv)

        margin = float(diff - eps) if (np.isfinite(diff) and np.isfinite(eps)) else -np.inf
        if margin > best_margin:
            best_margin = margin
            best_diff = diff
            best_eps = eps

    flag = bool(np.isfinite(best_margin) and best_margin > 0.0)
    score = float(best_diff / best_eps) if (np.isfinite(best_diff) and np.isfinite(best_eps) and best_eps > 0) else float(best_margin)
    return flag, score


def _bootstrap_prob_ci(series: pd.Series):
    # Block bootstrap CI for the smoothed probability estimate.
    s = series.dropna().astype(float).tail(int(BOOT_CI_WINDOW_DAYS))
    if len(s) < 120:
        return np.nan, np.nan

    arr = s.to_numpy(dtype=float)
    n = int(len(arr))
    blk = max(2, int(BOOT_CI_BLOCK))
    n_boot = max(50, int(BOOT_CI_N))
    n_blocks = int(np.ceil(n / float(blk)))

    rng = np.random.default_rng(int(BOOT_CI_RANDOM_SEED))
    boots = []
    for _ in range(n_boot):
        chunks = []
        for _b in range(n_blocks):
            start = int(rng.integers(0, n))
            block = arr[start : start + blk]
            if len(block) < blk:
                block = np.concatenate([block, arr[: blk - len(block)]])
            chunks.append(block)
        sample = np.concatenate(chunks)[:n]
        # mimic the main smoothing (span=5)
        val = float(pd.Series(sample).ewm(span=5, adjust=False).mean().iloc[-1])
        if np.isfinite(val):
            boots.append(val)

    if not boots:
        return np.nan, np.nan

    qlo = float(BOOT_CI_Q_LO)
    qhi = float(BOOT_CI_Q_HI)
    lo, hi = np.quantile(np.asarray(boots, dtype=float), [qlo, qhi])
    return float(lo), float(hi)


def _quantile_threshold(series: pd.Series, q: float) -> float:
    s = series.dropna().astype(float)
    if len(s) < 60:
        return np.nan

    if THRESH_METHOD == 'rolling':
        w = int(THRESH_WINDOW_DAYS)
        minp = int(min(w, max(252, int(w * 0.6))))
        rq = s.rolling(w, min_periods=minp).quantile(float(q))
        if rq.notna().any():
            v = float(rq.iloc[-1])
            if np.isfinite(v):
                return float(v)

    return float(np.percentile(s.to_numpy(), float(q) * 100.0))

granger_by_domain = {}
alert_rows = []
for domain in prob_df.columns:
    series = prob_df[domain].dropna()
    cur = float(series.iloc[-1])
    thr_elev = _quantile_threshold(series, THRESH_ELEV_Q)
    thr_high = _quantile_threshold(series, THRESH_HIGH_Q)

    # Use EWMA slope for smoother early-warning (less noisy than raw diffs)
    if len(series) >= 60:
        ew = series.ewm(span=30, adjust=False).mean()
        slope = float(ew.diff().tail(30).mean())
    else:
        slope = float(series.diff().mean())

    if not np.isfinite(cur):
        status = 'Unknown'
    elif cur >= thr_high:
        status = 'High'
    elif cur >= thr_elev:
        status = 'Elevated'
    else:
        status = 'Normal'

    hoeff_flag, hoeff_score = _hoeffding_drift(series)

    ph_flag, ph_score = _page_hinkley_drift(series)
    cusum_flag, cusum_score = _cusum_drift(series)
    adwin_flag, adwin_score = _adwin_drift(series)
    drift_any = bool(hoeff_flag) or bool(ph_flag) or bool(cusum_flag) or bool(adwin_flag)

    ci_lo, ci_hi = np.nan, np.nan
    if bool(BOOT_CI_ENABLE):
        ci_lo, ci_hi = _bootstrap_prob_ci(series)

    gr_ok = False
    gr_p = np.nan
    gr_lag = np.nan
    gr_n = np.nan
    gr_maxlag_used = np.nan
    gr_sig = False
    if bool(GRANGER_ENABLE):
        try:
            px = _domain_proxy_price_for_granger(str(domain))
            g = _granger_prob_leads_drawdown(series, px, horizon_days=int(GRANGER_HORIZON_DAYS), max_lag=int(GRANGER_MAX_LAG))
            if isinstance(g, dict):
                granger_by_domain[str(domain)] = g
                gr_ok = bool(g.get('ok', False))
                gr_p = float(g.get('p_min', np.nan))
                gr_lag = float(g.get('best_lag', np.nan))
                gr_n = float(g.get('n_obs', np.nan))
                gr_maxlag_used = float(g.get('max_lag_used', np.nan))
                gr_sig = bool(np.isfinite(gr_p) and (gr_p < float(GRANGER_ALPHA)))
        except Exception:
            pass

    early_warning = bool((cur >= thr_elev) and (slope > 0)) or bool(drift_any)
    domain_thresholds[domain] = {'elev': thr_elev, 'high': thr_high, 'method': THRESH_METHOD}

    alert_rows.append(
        {
            'domain': domain,
            'asof': series.index[-1],
            'bubble_likelihood_now': cur,
            'hist_elev_threshold': thr_elev,
            'hist_high_threshold': thr_high,
            'recent_slope': slope,
            'risk_regime': status,
            'early_warning_flag': bool(early_warning),
            'hoeffding_drift_flag': bool(hoeff_flag),
            'hoeffding_drift_score': float(hoeff_score) if np.isfinite(hoeff_score) else np.nan,
            'page_hinkley_flag': bool(ph_flag),
            'page_hinkley_score': float(ph_score) if np.isfinite(ph_score) else np.nan,
            'cusum_flag': bool(cusum_flag),
            'cusum_score': float(cusum_score) if np.isfinite(cusum_score) else np.nan,
            'adwin_flag': bool(adwin_flag),
            'adwin_score': float(adwin_score) if np.isfinite(adwin_score) else np.nan,
            'drift_flag': bool(drift_any),
            'prob_ci_lo': float(ci_lo) if np.isfinite(ci_lo) else np.nan,
            'prob_ci_hi': float(ci_hi) if np.isfinite(ci_hi) else np.nan,
            'granger_ok': bool(gr_ok),
            'granger_p_min': float(gr_p) if np.isfinite(gr_p) else np.nan,
            'granger_best_lag': float(gr_lag) if np.isfinite(gr_lag) else np.nan,
            'granger_n_obs': float(gr_n) if np.isfinite(gr_n) else np.nan,
            'granger_horizon_days': int(GRANGER_HORIZON_DAYS),
            'granger_max_lag_used': float(gr_maxlag_used) if np.isfinite(gr_maxlag_used) else np.nan,
            'granger_alpha': float(GRANGER_ALPHA),
            'granger_sig': bool(gr_sig),
        }
    )

alerts = pd.DataFrame(alert_rows).set_index('domain').sort_index()

# Optional: Granger validation summary table
if bool(GRANGER_ENABLE):
    try:
        gr_tbl = pd.DataFrame(granger_by_domain).T
        if (gr_tbl is not None) and isinstance(gr_tbl, pd.DataFrame) and (not gr_tbl.empty):
            keep = [c for c in ['p_min', 'best_lag', 'n_obs', 'horizon_days', 'max_lag_used'] if c in gr_tbl.columns]
            view = gr_tbl[keep].copy() if keep else gr_tbl.copy()
            try:
                view = view.sort_values('p_min') if 'p_min' in view.columns else view
            except Exception:
                pass

            if not bool(SHOW_ONLY_SIMPLE_EWS_TABLE):
                try:
                    display(view.style.format({'p_min': '{:.3g}'}))
                except Exception:
                    display(view)
    except Exception:
        pass

# Merge news signals into alerts (so downstream sections can use them)
# - Alpha Vantage: news_heat_0_100 + news_sentiment_mean
# - Yahoo headlines: headline_sentiment_mean + headlines
if ('news_table' in globals()) and isinstance(news_table, pd.DataFrame) and (not news_table.empty):
    try:
        alerts['news_heat_0_100'] = pd.to_numeric(news_table.get('news_heat_0_100'), errors='coerce').reindex(alerts.index)
    except Exception:
        alerts['news_heat_0_100'] = np.nan
    try:
        alerts['news_sentiment_mean'] = pd.to_numeric(news_table.get('news_sentiment_mean'), errors='coerce').reindex(alerts.index)
    except Exception:
        alerts['news_sentiment_mean'] = np.nan
else:
    alerts['news_heat_0_100'] = np.nan
    alerts['news_sentiment_mean'] = np.nan

if ('sent_table' in globals()) and isinstance(sent_table, pd.DataFrame) and (not sent_table.empty):
    try:
        alerts['headline_sentiment_mean'] = pd.to_numeric(sent_table.get('headline_sentiment_mean'), errors='coerce').reindex(alerts.index)
    except Exception:
        alerts['headline_sentiment_mean'] = np.nan
    try:
        alerts['headlines'] = pd.to_numeric(sent_table.get('headlines'), errors='coerce').reindex(alerts.index)
    except Exception:
        alerts['headlines'] = np.nan
else:
    alerts['headline_sentiment_mean'] = np.nan
    alerts['headlines'] = np.nan


# -----------------------------
# 12.6a LPPL endgame overlay (Sornette-style, best-effort)
# -----------------------------
# LPPL is a *timing* model: it tries to estimate whether a bubble-like acceleration could be approaching
# a critical time tc. We treat this as supplementary evidence, not as the core probability model.
RUN_LPPL = bool(globals().get('RUN_LPPL', True))
LPPL_WINDOW_DAYS = int(globals().get('LPPL_WINDOW_DAYS', 252) or 252)          # 1-year lookback (was 2y)
LPPL_TC_MAX_DAYS = int(globals().get('LPPL_TC_MAX_DAYS', 365) or 365)          # 1-year horizon  (was 180d)
LPPL_RANDOM_STARTS = int(globals().get('LPPL_RANDOM_STARTS', 40) or 40)        # more restarts for stability
LPPL_QUALITY_DECAY = float(globals().get('LPPL_QUALITY_DECAY', 4.0) or 4.0)    # softer RMSE penalty (was 8.0)


def _lppl_domain_index_price(tickers: list[str]) -> pd.Series:
    cols = [t for t in (tickers or []) if isinstance(t, str) and t.strip() and (t in detect_prices.columns)]
    if not cols:
        return pd.Series(dtype=float)

    px = detect_prices[cols].astype(float).replace(0.0, np.nan).ffill()
    if px.dropna(how='all').empty:
        return pd.Series(dtype=float)

    base = px.apply(lambda s: float(s.dropna().iloc[0]) if len(s.dropna()) else np.nan)
    idx = (px / base).mean(axis=1, skipna=True).rename('idx').dropna()
    return idx


def fit_lppl_endgame(price: pd.Series) -> dict[str, float]:
    # Returns: tc_days (ahead), endgame_0_100, rmse
    out = {'tc_days': np.nan, 'endgame_0_100': np.nan, 'rmse': np.nan}

    p = pd.to_numeric(price, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(p) < int(LPPL_WINDOW_DAYS):
        return out

    sub = p.tail(int(LPPL_WINDOW_DAYS))
    try:
        if float(sub.iloc[-1] / sub.iloc[0]) <= 1.0:
            out['tc_days'] = np.nan
            out['endgame_0_100'] = 0.0
            out['rmse'] = np.nan
            return out
    except Exception:
        return out

    try:
        from scipy.optimize import curve_fit
    except Exception:
        return out

    y = np.log(np.clip(sub.to_numpy(dtype=float), 1e-12, None))
    n = int(len(y))
    t = np.arange(n, dtype=float)

    def lppl(t, A, B, C, tc, m, omega, phi):
        dt = np.clip(tc - t, 1e-6, None)
        return A + B * (dt**m) + C * (dt**m) * np.cos(omega * np.log(dt) + phi)

    tc_lo = float(n + 5)
    tc_hi = float(n + int(LPPL_TC_MAX_DAYS))

    bounds_lo = [float(np.min(y) - 1.0), -10.0, -2.0, tc_lo, 0.10, 6.0, -np.pi]
    bounds_hi = [float(np.max(y) + 1.0), 10.0, 2.0, tc_hi, 0.90, 13.0, np.pi]

    rng_lppl = np.random.default_rng(7)
    best = {'rmse': np.inf, 'tc': np.nan, 'm': np.nan, 'omega': np.nan, 'B': np.nan}

    for _ in range(int(LPPL_RANDOM_STARTS)):
        A0 = float(y[-1])
        B0 = float(-1.0 * abs(rng_lppl.normal(0.5, 0.5)))
        C0 = float(rng_lppl.normal(0.0, 0.2))
        tc0 = float(rng_lppl.uniform(tc_lo, tc_hi))
        m0 = float(rng_lppl.uniform(0.15, 0.85))
        omega0 = float(rng_lppl.uniform(6.5, 12.5))
        phi0 = float(rng_lppl.uniform(-np.pi, np.pi))
        p0 = [A0, B0, C0, tc0, m0, omega0, phi0]

        try:
            popt, _pcov = curve_fit(
                lppl,
                t,
                y,
                p0=p0,
                bounds=(bounds_lo, bounds_hi),
                maxfev=8000,
            )
        except Exception:
            continue

        yhat = lppl(t, *popt)
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        if not np.isfinite(rmse) or rmse <= 0:
            continue

        _A, B, _C, tc, m, omega, _phi = [float(v) for v in popt]
        if rmse < best['rmse']:
            best = {'rmse': rmse, 'tc': tc, 'm': m, 'omega': omega, 'B': B}

    if not np.isfinite(best['rmse']) or not np.isfinite(best['tc']):
        return out

    tc_days = float(best['tc'] - (n - 1))
    out['tc_days'] = tc_days
    out['rmse'] = float(best['rmse'])

    # Endgame score: closer tc -> higher score, penalize poor fit; require bubble-like curvature (B < 0)
    score = 0.0
    if (
        (tc_days > 0.0)
        and (tc_days <= float(LPPL_TC_MAX_DAYS))
        and (0.10 <= float(best.get('m', np.nan)) <= 0.90)
        and (6.0 <= float(best.get('omega', np.nan)) <= 13.0)
        and np.isfinite(float(best.get('B', np.nan)))
        and (float(best.get('B', 0.0)) < 0.0)
    ):
        closeness = float(1.0 - (tc_days / float(LPPL_TC_MAX_DAYS)))
        quality = float(np.exp(-float(best['rmse']) * float(LPPL_QUALITY_DECAY)))
        score = float(np.clip(closeness * quality, 0.0, 1.0))

    out['endgame_0_100'] = float(score * 100.0)
    return out


lppl_by_domain: dict[str, dict[str, float]] = {}
if RUN_LPPL:
    for domain, tickers in DOMAIN_TICKERS.items():
        idx = _lppl_domain_index_price(tickers)
        lppl_by_domain[domain] = fit_lppl_endgame(idx)

alerts['lppl_tc_days'] = pd.Series({k: v.get('tc_days', np.nan) for k, v in lppl_by_domain.items()}).reindex(alerts.index)
alerts['lppl_endgame_0_100'] = pd.Series({k: v.get('endgame_0_100', np.nan) for k, v in lppl_by_domain.items()}).reindex(alerts.index)


def _lppl_nudge_buckets(row: pd.Series) -> int:
    # Return +0 or +1 bucket nudge for strong LPPL endgame evidence.
    s = row.get('lppl_endgame_0_100', np.nan)
    tc = row.get('lppl_tc_days', np.nan)
    if np.isfinite(s) and np.isfinite(tc):
        if (float(s) >= 50.0) and (float(tc) <= float(LPPL_TC_MAX_DAYS)):
            return 1
    return 0


alerts['lppl_nudge_buckets'] = alerts.apply(_lppl_nudge_buckets, axis=1)


def _news_nudge_buckets(row: pd.Series) -> int:
    # Return +0 or +1 bucket nudge when news is extremely hot and very negative.
    # This is an additive adjustment on top of the Yahoo-price-driven probability + 30D change.

    heat = row.get('news_heat_0_100', np.nan)
    nsent = row.get('news_sentiment_mean', np.nan)
    ysent = row.get('headline_sentiment_mean', np.nan)
    ycnt = row.get('headlines', np.nan)

    nudge = 0
    # Alpha Vantage trigger (primary)
    if np.isfinite(heat) and np.isfinite(nsent):
        if (float(heat) >= 80.0) and (float(nsent) <= -0.15):
            nudge = 1

    # Yahoo headline confirmation / backup
    if np.isfinite(ycnt) and np.isfinite(ysent):
        if (float(ycnt) >= 5.0) and (float(ysent) <= -0.05):
            nudge = max(nudge, 1)

    return int(nudge)


alerts['news_nudge_buckets'] = alerts.apply(_news_nudge_buckets, axis=1)


# -----------------------------
# 12.6a SEC EDGAR filings NLP (10-K/10-Q) (best-effort)
# -----------------------------
# This adds a disclosure/tone signal from filings text (complements price-based features).
# Disabled by default; requires a proper SEC User-Agent.
RUN_SEC_FILINGS_NLP = bool(globals().get('RUN_SEC_FILINGS_NLP', False))

# Defaults (so downstream table logic can always reference these columns safely)
alerts['sec_heat_0_100'] = np.nan
alerts['sec_keyword_rate_per_10k'] = np.nan
alerts['sec_risk_rate_per_10k'] = np.nan
alerts['sec_nudge_buckets'] = 0
alerts['sec_tickers_used'] = ''

if RUN_SEC_FILINGS_NLP:
    import os
    import json
    import time
    import re
    import html as _html
    import urllib.request

    sec_ua = str(os.getenv('EWS_SEC_USER_AGENT', '') or '').strip()
    if not sec_ua:
        print('[sec] Missing EWS_SEC_USER_AGENT; skipping SEC filings NLP.')
    else:
        cache_dir = (os.getenv('EWS_SEC_CACHE_DIR', '') or '').strip() or '.sec_edgar_cache'
        os.makedirs(cache_dir, exist_ok=True)

        SEC_MAX_TICKERS = int(os.getenv('EWS_SEC_MAX_TICKERS', '12') or 12)
        SEC_MAX_FILINGS_PER_TICKER = int(os.getenv('EWS_SEC_MAX_FILINGS_PER_TICKER', '2') or 2)
        SEC_TOP_TICKERS_PER_DOMAIN = int(os.getenv('EWS_SEC_TOP_TICKERS_PER_DOMAIN', '3') or 3)
        SEC_FORMS = {'10-K', '10-Q'}

        SEC_DOMAIN_KEYWORDS = {
            'AI bubble': [
                'artificial intelligence',
                'generative ai',
                'machine learning',
                'data center',
                'gpu',
                'large language model',
            ],
            'Private Credit bubble': [
                'private credit',
                'direct lending',
                'leveraged loan',
                'collateralized loan obligation',
                'clo',
                'structured credit',
                'middle market',
            ],
            'Crypto bubble': [
                'crypto',
                'cryptocurrency',
                'bitcoin',
                'ethereum',
                'blockchain',
                'digital asset',
                'token',
                'stablecoin',
            ],
        }

        # A small, conservative risk/uncertainty proxy (not full LM dictionary).
        SEC_RISK_WORDS = [
            'risk',
            'uncertain',
            'uncertainty',
            'volatility',
            'liquidity',
            'impairment',
            'default',
            'downgrade',
            'material weakness',
            'going concern',
        ]

        def _sec_headers():
            return {
                'User-Agent': sec_ua,
                'Accept': 'application/json,text/html,text/plain',
            }

        def _is_fresh(path: str, max_age_days: int) -> bool:
            try:
                return os.path.exists(path) and (time.time() - os.path.getmtime(path) < float(max_age_days) * 86400.0)
            except Exception:
                return False

        def _fetch(url: str, timeout_s: int = 30) -> bytes:
            req = urllib.request.Request(url, headers=_sec_headers())
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.read()

        def _fetch_json_cached(url: str, cache_path: str, max_age_days: int) -> object:
            if _is_fresh(cache_path, max_age_days):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    pass

            try:
                b = _fetch(url)
                obj = json.loads(b.decode('utf-8', errors='ignore'))
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(obj, f)
                time.sleep(0.2)
                return obj
            except Exception as e:
                print(f'[sec] Fetch failed: {url} ({e})')
                return None

        def _fetch_text_cached(url: str, cache_path: str, max_age_days: int) -> str:
            if _is_fresh(cache_path, max_age_days):
                try:
                    with open(cache_path, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
                except Exception:
                    pass

            try:
                b = _fetch(url, timeout_s=45)
                txt = b.decode('utf-8', errors='ignore')
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(txt)
                time.sleep(0.2)
                return txt
            except Exception as e:
                print(f'[sec] Fetch failed: {url} ({e})')
                return ''

        def _build_ticker_cik_map() -> dict:
            # Returns ticker -> 10-digit CIK string.
            urls = [
                'https://www.sec.gov/files/company_tickers.json',
                'https://www.sec.gov/files/company_tickers_exchange.json',
            ]
            for u in urls:
                obj = _fetch_json_cached(u, os.path.join(cache_dir, os.path.basename(u)), max_age_days=30)
                if not obj:
                    continue
                if isinstance(obj, dict) and obj:
                    vals = list(obj.values())
                elif isinstance(obj, list):
                    vals = obj
                else:
                    vals = []

                out = {}
                for r in vals:
                    try:
                        t = str(r.get('ticker', '') or '').upper().strip()
                        cik = r.get('cik_str', r.get('cik', None))
                        if t and cik is not None:
                            out[t] = str(int(cik)).zfill(10)
                    except Exception:
                        continue

                if out:
                    return out
            return {}

        ticker_to_cik = _build_ticker_cik_map()
        if not ticker_to_cik:
            print('[sec] Could not load ticker->CIK map; skipping SEC filings NLP.')
        else:
            # Candidate tickers: user can override via env, else we use the roster (user-provided symbols).
            tickers_env = str(os.getenv('EWS_SEC_TICKERS', '') or '').strip()
            cand = []
            if tickers_env:
                cand = [t.strip().upper() for t in tickers_env.split(',') if t.strip()]
            elif ('roster' in globals()) and isinstance(roster, pd.DataFrame) and ('Ticker' in roster.columns):
                tmp = roster[['Ticker']].copy()
                if 'RiskLimitPercent' in roster.columns:
                    tmp['RiskLimitPercent'] = pd.to_numeric(roster['RiskLimitPercent'], errors='coerce').fillna(0.0)
                    tmp = tmp.sort_values('RiskLimitPercent', ascending=False)
                cand = [str(t).strip().upper() for t in tmp['Ticker'].tolist() if isinstance(t, str) and t.strip()]

            cand = [t for t in cand if t and (t in ticker_to_cik)]
            cand = cand[: int(max(1, SEC_MAX_TICKERS))]

            if not cand:
                print('[sec] No candidate tickers with CIK mapping; skipping SEC filings NLP.')
            else:
                def _strip_html(text: str) -> str:
                    try:
                        text = re.sub(r'(?is)<(script|style).*?>.*?</\\1>', ' ', text)
                        text = re.sub(r'(?is)<[^>]+>', ' ', text)
                        text = _html.unescape(text)
                        text = re.sub(r'\\s+', ' ', text)
                        return text
                    except Exception:
                        return text

                def _score_text(text: str) -> dict:
                    t = _strip_html(text)
                    t = t[:500_000]  # cap work/time (still enough for keyword + tone proxies)
                    t_l = t.lower()
                    n_words = len(re.findall(r"[a-z]{2,}", t_l))
                    out = {'n_words': float(n_words)}

                    for dom, kws in SEC_DOMAIN_KEYWORDS.items():
                        c = 0
                        for kw in kws:
                            if kw:
                                c += int(t_l.count(kw))
                        out[f'{dom}__kw_count'] = float(c)
                        out[f'{dom}__kw_per_10k'] = (10000.0 * float(c) / float(n_words)) if n_words > 0 else np.nan

                    risk_c = 0
                    for w in SEC_RISK_WORDS:
                        if w:
                            risk_c += int(t_l.count(w))
                    out['risk_count'] = float(risk_c)
                    out['risk_per_10k'] = (10000.0 * float(risk_c) / float(n_words)) if n_words > 0 else np.nan
                    return out

                ticker_rows = []
                for tkr in cand:
                    cik10 = ticker_to_cik.get(tkr, '')
                    if not cik10:
                        continue

                    sub_url = f'https://data.sec.gov/submissions/CIK{cik10}.json'
                    sub_path = os.path.join(cache_dir, f'sub_{cik10}.json')
                    sub = _fetch_json_cached(sub_url, sub_path, max_age_days=7)
                    if not isinstance(sub, dict):
                        continue

                    recent = ((sub.get('filings') or {}).get('recent') or {})
                    forms = list(recent.get('form', []) or [])
                    accs = list(recent.get('accessionNumber', []) or [])
                    prim = list(recent.get('primaryDocument', []) or [])
                    fdates = list(recent.get('filingDate', []) or [])

                    used = 0
                    agg = None
                    for i in range(min(len(forms), len(accs), len(prim), len(fdates))):
                        form = str(forms[i] or '').strip().upper()
                        if form not in SEC_FORMS:
                            continue
                        acc = str(accs[i] or '').strip()
                        doc = str(prim[i] or '').strip()
                        fdt = str(fdates[i] or '').strip()
                        if (not acc) or (not doc):
                            continue

                        acc_nodash = acc.replace('-', '')
                        cik_int = str(int(cik10))

                        safe_doc = doc.replace('/', '_')
                        safe_doc = re.sub(r'[^A-Za-z0-9._-]', '_', safe_doc)
                        filing_url = f'https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}'
                        filing_path = os.path.join(cache_dir, f'filing_{cik10}_{acc_nodash}_{safe_doc}')

                        raw = _fetch_text_cached(filing_url, filing_path, max_age_days=365)
                        if not raw:
                            continue

                        met = _score_text(raw)
                        met['ticker'] = tkr
                        met['cik10'] = cik10
                        met['form'] = form
                        met['filing_date'] = fdt

                        if agg is None:
                            agg = met.copy()
                            agg['filings_used'] = 1
                        else:
                            for k, v in met.items():
                                if k in ('ticker', 'cik10', 'form', 'filing_date'):
                                    continue
                                try:
                                    agg[k] = float(agg.get(k, 0.0)) + float(v)
                                except Exception:
                                    pass
                            agg['filings_used'] = int(agg.get('filings_used', 0)) + 1

                        used += 1
                        if used >= int(SEC_MAX_FILINGS_PER_TICKER):
                            break

                    if agg is None:
                        continue

                    n_used = int(agg.get('filings_used', 1))
                    for k, v in list(agg.items()):
                        if k in ('ticker', 'cik10', 'form', 'filing_date', 'filings_used'):
                            continue
                        try:
                            agg[k] = float(v) / float(n_used)
                        except Exception:
                            pass

                    ticker_rows.append(agg)

                sec_by_ticker = pd.DataFrame(ticker_rows)
                if sec_by_ticker.empty:
                    print('[sec] No usable 10-K/10-Q filings found for candidate tickers; skipping.')
                else:
                    for dom in list(alerts.index):
                        kw_col = f'{dom}__kw_per_10k'
                        if kw_col not in sec_by_ticker.columns:
                            continue

                        top = sec_by_ticker.sort_values(kw_col, ascending=False).head(int(max(1, SEC_TOP_TICKERS_PER_DOMAIN)))
                        kw_rate = float(top[kw_col].mean()) if top[kw_col].notna().any() else np.nan
                        risk_rate = float(top['risk_per_10k'].mean()) if ('risk_per_10k' in top.columns and top['risk_per_10k'].notna().any()) else np.nan

                        # Heat: saturating transform (0..100). Interpretable: higher = more domain talk + more risk language.
                        heat = np.nan
                        try:
                            heat = 30.0 * float(np.log1p(max(0.0, kw_rate))) if np.isfinite(kw_rate) else np.nan
                            if np.isfinite(risk_rate):
                                heat += 10.0 * float(np.tanh((float(risk_rate) - 60.0) / 40.0))
                            heat = float(np.clip(heat, 0.0, 100.0)) if np.isfinite(heat) else np.nan
                        except Exception:
                            heat = np.nan

                        nudge = 0
                        try:
                            if np.isfinite(heat):
                                if heat >= 85.0:
                                    nudge = 2
                                elif heat >= 70.0:
                                    nudge = 1
                        except Exception:
                            nudge = 0

                        alerts.loc[dom, 'sec_keyword_rate_per_10k'] = kw_rate
                        alerts.loc[dom, 'sec_risk_rate_per_10k'] = risk_rate
                        alerts.loc[dom, 'sec_heat_0_100'] = heat
                        alerts.loc[dom, 'sec_nudge_buckets'] = int(nudge)
                        try:
                            alerts.loc[dom, 'sec_tickers_used'] = ','.join([str(x) for x in top.get('ticker', pd.Series(dtype=str)).tolist()])
                        except Exception:
                            alerts.loc[dom, 'sec_tickers_used'] = ''

                    # Optional debug table
                    if not bool(globals().get('SHOW_ONLY_SIMPLE_EWS_TABLE', True)):
                        try:
                            cols = ['ticker', 'filings_used', 'risk_per_10k']
                            for dom in list(alerts.index):
                                c = f'{dom}__kw_per_10k'
                                if c in sec_by_ticker.columns:
                                    cols.append(c)
                            cols = [c for c in cols if c in sec_by_ticker.columns]
                            if cols:
                                display(sec_by_ticker[cols].sort_values('risk_per_10k', ascending=False).head(20))
                        except Exception:
                            pass

# -----------------------------
# 12.6b Earnings-call transcripts (local folder) (best-effort)
# -----------------------------
# Local-only scaffold: score transcript text files from a folder (no scraping) using a simple
# keyword + risk-language proxy. Disabled by default.
RUN_TRANSCRIPTS_NLP = bool(globals().get('RUN_TRANSCRIPTS_NLP', False))

# Defaults (so downstream table logic can always reference these columns safely)
alerts['transcript_heat_0_100'] = np.nan
alerts['transcript_keyword_rate_per_10k'] = np.nan
alerts['transcript_risk_rate_per_10k'] = np.nan
alerts['transcript_nudge_buckets'] = 0
alerts['transcript_tickers_used'] = ''
alerts['transcript_docs_used'] = ''

if RUN_TRANSCRIPTS_NLP:
    import os
    import re
    import json
    import glob

    transcripts_dir = (os.getenv('EWS_TRANSCRIPTS_DIR', '') or '').strip() or 'transcripts'
    if not os.path.isdir(transcripts_dir):
        print(f'[transcripts] Missing transcripts directory: {transcripts_dir}; skipping transcripts NLP.')
    else:
        TRANSCRIPT_MAX_FILES_PER_TICKER = int(os.getenv('EWS_TRANSCRIPTS_MAX_FILES_PER_TICKER', '2') or 2)
        TRANSCRIPT_TOP_TICKERS_PER_DOMAIN = int(os.getenv('EWS_TRANSCRIPTS_TOP_TICKERS_PER_DOMAIN', '3') or 3)
        TRANSCRIPT_MAX_CHARS_PER_DOC = int(os.getenv('EWS_TRANSCRIPTS_MAX_CHARS_PER_DOC', '250000') or 250000)

        TRANSCRIPT_DOMAIN_KEYWORDS = {
            'AI bubble': [
                'artificial intelligence',
                'generative ai',
                'machine learning',
                'data center',
                'gpu',
                'large language model',
            ],
            'Private Credit bubble': [
                'private credit',
                'direct lending',
                'leveraged loan',
                'collateralized loan obligation',
                'clo',
                'structured credit',
                'middle market',
            ],
            'Crypto bubble': [
                'crypto',
                'cryptocurrency',
                'bitcoin',
                'ethereum',
                'blockchain',
                'digital asset',
                'token',
                'stablecoin',
            ],
        }

        # Conservative risk/uncertainty proxy (not a full LM dictionary).
        TRANSCRIPT_RISK_WORDS = [
            'risk',
            'uncertain',
            'uncertainty',
            'volatility',
            'liquidity',
            'impairment',
            'default',
            'downgrade',
            'going concern',
        ]

        domain_ticker_set: set[str] = set()
        try:
            if 'DOMAIN_TICKERS' in globals() and isinstance(DOMAIN_TICKERS, dict):
                for _, tks in DOMAIN_TICKERS.items():
                    if isinstance(tks, (list, tuple, set)):
                        for t in tks:
                            if isinstance(t, str) and t.strip():
                                domain_ticker_set.add(str(t).upper().strip())
        except Exception:
            domain_ticker_set = set()

        def _infer_ticker_from_name(path: str) -> str:
            try:
                stem = os.path.splitext(os.path.basename(path))[0]
                m = re.match(r'^([A-Za-z0-9.^=-]{1,15})', stem)
                tok = (m.group(1) if m else '').upper().strip()
                return tok
            except Exception:
                return ''

        def _read_json_text(path: str) -> tuple[str, str]:
            # Returns (ticker, text). Supports simple dict schema or a list of {text: ...} segments.
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    obj = json.load(f)

                if isinstance(obj, dict):
                    ticker = str(obj.get('ticker', '') or '').upper().strip()
                    text = ''
                    for k in ('text', 'transcript', 'content', 'body'):
                        v = obj.get(k, None)
                        if isinstance(v, str) and v.strip():
                            text = v
                            break
                    if (not text) and isinstance(obj.get('segments'), list):
                        parts = []
                        for seg in obj.get('segments'):
                            if isinstance(seg, dict):
                                t = seg.get('text', None)
                                if isinstance(t, str) and t.strip():
                                    parts.append(t)
                        text = '\n'.join(parts)
                    return ticker, text

                if isinstance(obj, list):
                    parts = []
                    for seg in obj:
                        if isinstance(seg, dict):
                            t = seg.get('text', None)
                            if isinstance(t, str) and t.strip():
                                parts.append(t)
                    return '', '\n'.join(parts)

                return '', ''
            except Exception:
                return '', ''

        def _read_text_file(path: str) -> str:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception:
                return ''

        def _normalize_text(text: str) -> str:
            try:
                t = re.sub(r'(?is)<(script|style).*?>.*?</\\1>', ' ', text)
                t = re.sub(r'(?is)<[^>]+>', ' ', t)
                t = re.sub(r'\\s+', ' ', t)
                return t
            except Exception:
                return text

        def _score_text(text: str) -> dict:
            t = _normalize_text(text)
            t = t[:800_000]
            t_l = t.lower()
            n_words = len(re.findall(r"[a-z]{2,}", t_l))
            out = {'n_words': float(n_words)}

            for dom, kws in TRANSCRIPT_DOMAIN_KEYWORDS.items():
                c = 0
                for kw in kws:
                    if kw:
                        c += int(t_l.count(str(kw)))
                out[f'{dom}__kw_count'] = float(c)
                out[f'{dom}__kw_per_10k'] = (10000.0 * float(c) / float(n_words)) if n_words > 0 else np.nan

            risk_c = 0
            for w in TRANSCRIPT_RISK_WORDS:
                if w:
                    risk_c += int(t_l.count(str(w)))
            out['risk_count'] = float(risk_c)
            out['risk_per_10k'] = (10000.0 * float(risk_c) / float(n_words)) if n_words > 0 else np.nan
            return out

        allowed_ext = {'.txt', '.md', '.json'}
        transcript_paths = [
            p for p in glob.glob(os.path.join(transcripts_dir, '**', '*'), recursive=True) if os.path.isfile(p)
        ]
        transcript_file_rows: list[dict] = []
        for p in transcript_paths:
            ext = str(os.path.splitext(p)[1] or '').lower()
            if ext not in allowed_ext:
                continue

            tkr = ''
            pre_txt = ''
            if ext == '.json':
                tkr, pre_txt = _read_json_text(p)
            if not tkr:
                tkr = _infer_ticker_from_name(p)
            tkr = str(tkr or '').upper().strip()
            if not tkr:
                continue
            if domain_ticker_set and (tkr not in domain_ticker_set):
                continue

            try:
                mtime = float(os.path.getmtime(p))
            except Exception:
                mtime = 0.0
            transcript_file_rows.append({'ticker': tkr, 'path': p, 'ext': ext, 'mtime': mtime, 'pre_txt': pre_txt})

        if not transcript_file_rows:
            print(f'[transcripts] No transcript files found under: {transcripts_dir}')
        else:
            transcript_by_ticker: dict[str, list[dict]] = {}
            for r in transcript_file_rows:
                transcript_by_ticker.setdefault(str(r.get('ticker', '')), []).append(r)

            transcript_ticker_rows: list[dict] = []
            for tkr, items in transcript_by_ticker.items():
                items = sorted(items, key=lambda x: float(x.get('mtime', 0.0)), reverse=True)
                items = items[: int(max(1, TRANSCRIPT_MAX_FILES_PER_TICKER))]

                texts: list[str] = []
                basenames: list[str] = []
                for it in items:
                    path = str(it.get('path', '') or '')
                    ext = str(it.get('ext', '') or '').lower()
                    txt = str(it.get('pre_txt', '') or '')
                    if (not txt) and path:
                        if ext == '.json':
                            _, txt = _read_json_text(path)
                        else:
                            txt = _read_text_file(path)
                    if txt:
                        txt = txt[: int(max(10_000, TRANSCRIPT_MAX_CHARS_PER_DOC))]
                        texts.append(txt)
                        basenames.append(os.path.basename(path))

                if not texts:
                    continue

                combined = '\n'.join(texts)
                met = _score_text(combined)
                met['ticker'] = str(tkr)
                met['docs_used'] = float(len(basenames))
                met['docs_list'] = ';'.join([str(x) for x in basenames[:12]])
                transcript_ticker_rows.append(met)

            tr_by_ticker = pd.DataFrame(transcript_ticker_rows)
            if tr_by_ticker.empty:
                print('[transcripts] No usable transcript text found after filtering/caps; skipping.')
            else:
                try:
                    tr_by_ticker['ticker'] = tr_by_ticker['ticker'].astype(str).str.upper().str.strip()
                except Exception:
                    pass

                for dom in list(alerts.index):
                    tickers = []
                    try:
                        if 'DOMAIN_TICKERS' in globals() and isinstance(DOMAIN_TICKERS, dict):
                            tickers = DOMAIN_TICKERS.get(dom, [])
                    except Exception:
                        tickers = []
                    tickers = [str(t).upper().strip() for t in tickers if isinstance(t, str) and t.strip()]
                    if not tickers:
                        continue

                    kw_col = f'{dom}__kw_per_10k'
                    if kw_col not in tr_by_ticker.columns:
                        continue

                    sub = tr_by_ticker[tr_by_ticker['ticker'].isin(tickers)].copy()
                    if sub.empty:
                        continue

                    top = sub.sort_values(kw_col, ascending=False).head(int(max(1, TRANSCRIPT_TOP_TICKERS_PER_DOMAIN)))
                    kw_rate = float(top[kw_col].mean()) if top[kw_col].notna().any() else np.nan
                    risk_rate = (
                        float(top['risk_per_10k'].mean())
                        if ('risk_per_10k' in top.columns and top['risk_per_10k'].notna().any())
                        else np.nan
                    )

                    heat = np.nan
                    try:
                        heat = 30.0 * float(np.log1p(max(0.0, kw_rate))) if np.isfinite(kw_rate) else np.nan
                        if np.isfinite(risk_rate):
                            heat += 10.0 * float(np.tanh((float(risk_rate) - 60.0) / 40.0))
                        heat = float(np.clip(heat, 0.0, 100.0)) if np.isfinite(heat) else np.nan
                    except Exception:
                        heat = np.nan

                    nudge = 0
                    try:
                        if np.isfinite(heat):
                            if heat >= 85.0:
                                nudge = 2
                            elif heat >= 70.0:
                                nudge = 1
                    except Exception:
                        nudge = 0

                    alerts.loc[dom, 'transcript_keyword_rate_per_10k'] = kw_rate
                    alerts.loc[dom, 'transcript_risk_rate_per_10k'] = risk_rate
                    alerts.loc[dom, 'transcript_heat_0_100'] = heat
                    alerts.loc[dom, 'transcript_nudge_buckets'] = int(nudge)
                    try:
                        alerts.loc[dom, 'transcript_tickers_used'] = ','.join(
                            [str(x) for x in top.get('ticker', pd.Series(dtype=str)).tolist()]
                        )
                    except Exception:
                        alerts.loc[dom, 'transcript_tickers_used'] = ''
                    try:
                        docs = []
                        for s in top.get('docs_list', pd.Series(dtype=str)).tolist():
                            for part in str(s or '').split(';'):
                                part = part.strip()
                                if part and part not in docs:
                                    docs.append(part)
                        alerts.loc[dom, 'transcript_docs_used'] = ';'.join(docs[:12])
                    except Exception:
                        alerts.loc[dom, 'transcript_docs_used'] = ''

                # Optional debug view
                if not bool(globals().get('SHOW_ONLY_SIMPLE_EWS_TABLE', True)):
                    try:
                        cols = ['ticker', 'docs_used', 'risk_per_10k']
                        for dom in list(alerts.index):
                            c = f'{dom}__kw_per_10k'
                            if c in tr_by_ticker.columns:
                                cols.append(c)
                        cols = [c for c in cols if c in tr_by_ticker.columns]
                        if cols:
                            display(tr_by_ticker[cols].sort_values('risk_per_10k', ascending=False).head(20))
                    except Exception:
                        pass

# -----------------------------
# Simple Early Warning Summary (single-table output)
# -----------------------------

def _trend_label(delta_pp: float) -> str:
    if not np.isfinite(delta_pp):
        return 'Unknown'
    if delta_pp >= 10.0:
        return 'Rising fast'
    if delta_pp >= 5.0:
        return 'Rising'
    if delta_pp <= -10.0:
        return 'Falling fast'
    if delta_pp <= -5.0:
        return 'Falling'
    return 'Flat'


def _bucket_from(p_now: float, delta_pp: float) -> str:
    # Base bucket on probability, then bump up/down on strong 30D move.
    if not np.isfinite(p_now):
        return 'Unknown'
    if p_now >= 80.0:
        bucket = 'Red'
    elif p_now >= 60.0:
        bucket = 'Orange'
    elif p_now >= 40.0:
        bucket = 'Yellow'
    else:
        bucket = 'Green'

    if np.isfinite(delta_pp):
        if delta_pp >= 5.0:
            bucket = {'Green': 'Yellow', 'Yellow': 'Orange', 'Orange': 'Red'}.get(bucket, bucket)
        elif delta_pp <= -5.0:
            bucket = {'Red': 'Orange', 'Orange': 'Yellow', 'Yellow': 'Green'}.get(bucket, bucket)

    return bucket


def _bump_bucket(bucket: str, steps: int) -> str:
    steps = int(steps or 0)
    if steps <= 0:
        return bucket
    order = ['Green', 'Yellow', 'Orange', 'Red']
    if bucket not in order:
        return bucket
    i = order.index(bucket)
    j = min(i + steps, len(order) - 1)
    return order[j]


BUCKET_LABEL = {
    'Red': 'Red — Very risky',
    'Orange': 'Orange — Growing',
    'Yellow': 'Yellow — Small moves',
    'Green': 'Green — Negligible/No changes',
    'Unknown': 'Unknown',
}

BUCKET_COLOR = {
    'Red': '#FFC7CE',
    'Orange': '#FFD966',
    'Yellow': '#FFEB9C',
    'Green': '#C6EFCE',
}


def _confidence_score(series: pd.Series, p_now: float) -> float:
    # Statistical confidence (0-100) in the model's current probability.
    # Not a guarantee of correctness; it measures decisiveness/stability and penalizes short/stale data.
    s = series.dropna()
    if s.empty or (not np.isfinite(p_now)):
        return np.nan

    # 1) Decisiveness via probability entropy (max uncertainty at 50%, min at 0/100)
    p = float(np.clip(p_now / 100.0, 1e-6, 1.0 - 1e-6))
    entropy_bits = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))  # in [0, ~1]
    decisiveness = float(1.0 - np.clip(entropy_bits / 1.0, 0.0, 1.0))

    # 2) Data sufficiency (saturate around ~180 observations ≈ 6 months trading days)
    n = int(len(s))
    sufficiency = float(np.clip(n / 180.0, 0.0, 1.0))

    # 3) Recency vs today (stale data should reduce confidence)
    try:
        asof = pd.to_datetime(s.index.max()).normalize()
        age_days = float((pd.Timestamp.today().normalize() - asof).days)
    except Exception:
        age_days = np.nan

    if np.isfinite(age_days):
        if age_days <= 3.0:
            recency = 1.0
        else:
            recency = float(np.clip(1.0 - (age_days / 60.0), 0.0, 1.0))
    else:
        recency = 0.5

    # 4) Stability: lower recent volatility of the probability => slightly higher confidence
    recent = s.tail(180)
    std = float(recent.std()) if len(recent) else np.nan
    stability = float(1.0 / (1.0 + (std / 25.0))) if np.isfinite(std) else 0.5

    conf = 0.45 * decisiveness + 0.35 * sufficiency + 0.10 * recency + 0.10 * stability
    return float(np.clip(conf, 0.0, 1.0) * 100.0)


def _plain_english(bucket: str, trend: str) -> str:
    if bucket == 'Red':
        return f"Very risky. Bubble risk is {trend.lower()}."
    if bucket == 'Orange':
        return f"Growing risk. Bubble risk is {trend.lower()}."
    if bucket == 'Yellow':
        return f"Small moves. Bubble risk is {trend.lower()}."
    if bucket == 'Green':
        return f"Negligible/no change. Bubble risk is {trend.lower()}."
    return 'Unknown.'


rows = []
for domain in alerts.index:
    series = prob_df[domain].dropna() if domain in prob_df.columns else pd.Series(dtype=float)
    p_now = float(alerts.loc[domain, 'bubble_likelihood_now'])
    p_now_for_delta = p_now
    prev = np.nan
    if len(series) >= 2:
        asof = series.index.max()
        ew = series.ewm(span=30, adjust=False).mean()
        p_now_for_delta = float(ew.iloc[-1])

        prev_dt = asof - pd.Timedelta(days=30)
        prev_slice = ew[ew.index <= prev_dt]
        if len(prev_slice):
            prev = float(prev_slice.iloc[-1])
            prev_date = prev_slice.index[-1]
            # If the last available observation before ~30D is too far away, don't report a misleading "30D" change.
            if (asof - prev_date).days > 45:
                prev = np.nan

    delta_pp = float(p_now_for_delta - prev) if (np.isfinite(p_now_for_delta) and np.isfinite(prev)) else np.nan
    trend = _trend_label(delta_pp)

    # Base bucket from live price-derived probability + 30D move
    bucket_base = _bucket_from(p_now, delta_pp)
    news_nudge = 0
    lppl_nudge = 0
    sec_nudge = 0
    sec_heat = np.nan
    transcript_nudge = 0
    transcript_heat = np.nan
    try:
        news_nudge = int(float(alerts.loc[domain, 'news_nudge_buckets'])) if ('news_nudge_buckets' in alerts.columns) else 0
    except Exception:
        news_nudge = 0
    try:
        lppl_nudge = int(float(alerts.loc[domain, 'lppl_nudge_buckets'])) if ('lppl_nudge_buckets' in alerts.columns) else 0
    except Exception:
        lppl_nudge = 0
    try:
        sec_nudge = int(float(alerts.loc[domain, 'sec_nudge_buckets'])) if ('sec_nudge_buckets' in alerts.columns) else 0
    except Exception:
        sec_nudge = 0
    try:
        sec_heat = float(alerts.loc[domain, 'sec_heat_0_100']) if ('sec_heat_0_100' in alerts.columns) else np.nan
    except Exception:
        sec_heat = np.nan

    try:
        transcript_nudge = int(float(alerts.loc[domain, 'transcript_nudge_buckets'])) if ('transcript_nudge_buckets' in alerts.columns) else 0
    except Exception:
        transcript_nudge = 0
    try:
        transcript_heat = float(alerts.loc[domain, 'transcript_heat_0_100']) if ('transcript_heat_0_100' in alerts.columns) else np.nan
    except Exception:
        transcript_heat = np.nan

    nudge = int(max(news_nudge, lppl_nudge, sec_nudge, transcript_nudge))
    bucket = _bump_bucket(bucket_base, nudge)
    conf = _confidence_score(series, p_now)
    # Backtest confidence is computed later (after validation runs). Placeholder here, then backfilled.
    backtest_conf = np.nan
    drift = bool(alerts.loc[domain, 'drift_flag']) if ('drift_flag' in alerts.columns) else (
        bool(alerts.loc[domain, 'hoeffding_drift_flag']) if ('hoeffding_drift_flag' in alerts.columns) else False
    )

    prob_ci = ''
    try:
        ci_lo = float(alerts.loc[domain, 'prob_ci_lo']) if ('prob_ci_lo' in alerts.columns) else np.nan
        ci_hi = float(alerts.loc[domain, 'prob_ci_hi']) if ('prob_ci_hi' in alerts.columns) else np.nan
        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            prob_ci = f"{ci_lo:.1f}–{ci_hi:.1f}"
    except Exception:
        prob_ci = ''
    rows.append(
        {
            'Bubble': domain,
            'Severity': BUCKET_LABEL.get(bucket, bucket),
            'Probability Now (%)': p_now,
            'Prob CI (90%)': prob_ci,
            'Confidence (0-100)': conf,
            'Backtest Confidence (0-100)': backtest_conf,
            'LPPL endgame (0-100)': float(alerts.loc[domain, 'lppl_endgame_0_100']) if ('lppl_endgame_0_100' in alerts.columns) else np.nan,
            'LPPL tc (days)': float(alerts.loc[domain, 'lppl_tc_days']) if ('lppl_tc_days' in alerts.columns) else np.nan,
            'SEC heat (0-100)': sec_heat,
            'Transcript heat (0-100)': transcript_heat,
            '30D Change (pp)': delta_pp,
            'Trend': trend,
            'Drift Flag': ('Yes' if drift else 'No'),
            'Simple English': _plain_english(bucket, trend),
            'Nudge (+buckets)': int(nudge),
            '__bucket__': bucket,
        }
    )

simple_tbl = pd.DataFrame(rows).set_index('Bubble').sort_index()

# Keep the default/simple view clean when SEC NLP is disabled.
if not bool(globals().get('RUN_SEC_FILINGS_NLP', False)):
    try:
        simple_tbl = simple_tbl.drop(columns=['SEC heat (0-100)'], errors='ignore')
    except Exception:
        pass

# Keep the default/simple view clean when transcripts NLP is disabled.
if not bool(globals().get('RUN_TRANSCRIPTS_NLP', False)):
    try:
        simple_tbl = simple_tbl.drop(columns=['Transcript heat (0-100)'], errors='ignore')
    except Exception:
        pass

# Merge Alpha Vantage news metrics (if available)
if ('news_table' in globals()) and isinstance(news_table, pd.DataFrame) and (not news_table.empty):
    try:
        simple_tbl['News articles (24h)'] = pd.to_numeric(news_table.get('news_articles_24h'), errors='coerce').reindex(simple_tbl.index)
    except Exception:
        pass
    try:
        simple_tbl['News sentiment (avg)'] = pd.to_numeric(news_table.get('news_sentiment_mean'), errors='coerce').reindex(simple_tbl.index)
    except Exception:
        pass
    try:
        simple_tbl['News heat (0-100)'] = pd.to_numeric(news_table.get('news_heat_0_100'), errors='coerce').reindex(simple_tbl.index)
    except Exception:
        pass
    try:
        # Human-readable status for debugging (kept short).
        err = news_table.get('news_error')
        if err is not None:
            s = err.reindex(simple_tbl.index)
            s = s.fillna('')
            s = s.apply(lambda x: str(x)[:32])
            s = s.replace({'': 'ok'})
            simple_tbl['Alpha news status'] = s
    except Exception:
        pass

# Merge Yahoo headline sentiment signals (if available)
if ('sent_table' in globals()) and isinstance(sent_table, pd.DataFrame) and (not sent_table.empty):
    try:
        simple_tbl['Yahoo headline sentiment (avg)'] = pd.to_numeric(sent_table.get('headline_sentiment_mean'), errors='coerce').reindex(simple_tbl.index)
    except Exception:
        pass
    try:
        simple_tbl['Yahoo headlines (count)'] = pd.to_numeric(sent_table.get('headlines'), errors='coerce').reindex(simple_tbl.index)
    except Exception:
        pass


def _style_bucket(row: pd.Series) -> list[str]:
    b = str(row.get('__bucket__', ''))
    color = BUCKET_COLOR.get(b, '')
    out = []
    for col in row.index:
        if col == 'Severity' and color:
            out.append(f'background-color: {color}')
        else:
            out.append('')
    return out


# NOTE: We display this table once, later (after optional 2026–2030 projection columns are appended).


# -----------------------------
# 12.6b Optional alert delivery via Email (SMTP)
# -----------------------------

import os
import json
import hashlib
import smtplib
from email.message import EmailMessage


def _env(name: str, default: str = '') -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else str(v)


def send_smtp_email(subject: str, body: str) -> bool:
    host = _env('EWS_SMTP_HOST')
    port = int(_env('EWS_SMTP_PORT', '587') or 587)
    user = _env('EWS_SMTP_USER')
    password = os.getenv('EWS_SMTP_PASSWORD', '')
    mail_from = _env('EWS_EMAIL_FROM', user)
    mail_to = _env('EWS_EMAIL_TO', 'yellowmarinecapital@outlook.com')

    if not (host and user and password and mail_from and mail_to):
        print('[email] SMTP not configured (set EWS_SMTP_HOST/PORT/USER/PASSWORD and EWS_EMAIL_FROM/TO). Skipping email send.')
        return False

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = mail_from
    msg['To'] = mail_to
    msg.set_content(body)

    use_ssl = _env('EWS_SMTP_SSL', '0').lower() in ('1', 'true', 'yes')
    use_starttls = _env('EWS_SMTP_STARTTLS', '1').lower() in ('1', 'true', 'yes')

    try:
        if use_ssl:
            server = smtplib.SMTP_SSL(host, port, timeout=20)
        else:
            server = smtplib.SMTP(host, port, timeout=20)
        with server:
            server.ehlo()
            if (not use_ssl) and use_starttls:
                server.starttls()
                server.ehlo()
            server.login(user, password)
            server.send_message(msg)
        print('[email] Sent alert email.')
        return True
    except Exception as e:
        print(f'[email] Send failed: {e}')
        return False


if ENABLE_EMAIL_ALERTS:
    force_test = _env('EWS_EMAIL_TEST', '0').lower() in ('1', 'true', 'yes')
    triggered = alerts[(alerts['risk_regime'].isin(['Elevated', 'High'])) | (alerts['early_warning_flag'] == True)]

    if triggered.empty and not force_test:
        print('[email] No triggered domains; not sending email.')
    else:
        nl = chr(10)
        asof_str = str(alerts['asof'].max())
        lines = []
        lines.append('[EWS] Bubble Risk Alert' if not triggered.empty else '[EWS] SMTP Test Alert')
        lines.append('asof=' + asof_str)
        lines.append('')
        if triggered.empty:
            lines.append('No triggered domains (test mode).')
        else:
            lines.append('Triggered domains:')
            for domain, row in triggered.iterrows():
                lines.append(
                    f"- {domain}: {row['risk_regime']} (now={row['bubble_likelihood_now']:.1f}%, elev={row['hist_elev_threshold']:.1f}%, high={row['hist_high_threshold']:.1f}%)"
                )
        body = nl.join(lines)

        # De-duplicate (avoid spamming on repeated reruns)
        state_path = '.ews_last_email_hash.json'
        new_hash = hashlib.sha256(body.encode('utf-8')).hexdigest()
        prev_hash = None
        try:
            if os.path.exists(state_path):
                with open(state_path, 'r', encoding='utf-8') as f:
                    prev_hash = json.load(f).get('hash')
        except Exception:
            prev_hash = None

        if new_hash == prev_hash:
            print('[email] Alert unchanged vs last send; not resending.')
        else:
            subject = ('[EWS] Bubble Risk Alert (' + pd.Timestamp.utcnow().strftime('%Y-%m-%d') + ')') if not triggered.empty else ('[EWS] SMTP Test (' + pd.Timestamp.utcnow().strftime('%Y-%m-%d') + ')')
            if send_smtp_email(subject, body):
                try:
                    with open(state_path, 'w', encoding='utf-8') as f:
                        json.dump({'hash': new_hash, 'sent_at_utc': pd.Timestamp.utcnow().isoformat()}, f)
                except Exception as e:
                    print(f'[email] Warning: could not persist dedupe state: {e}')


# -----------------------------
# 12.6c Optional alert delivery via Slack (Incoming Webhook)
# -----------------------------

import urllib.request


def send_slack_webhook(body: str) -> bool:
    url = _env('EWS_SLACK_WEBHOOK_URL')
    if not url:
        print('[slack] Webhook not configured (set EWS_SLACK_WEBHOOK_URL). Skipping slack send.')
        return False

    payload = {'text': body}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            code = int(getattr(resp, 'status', 200) or 200)
            if code >= 400:
                print(f'[slack] Send failed: HTTP {code}')
                return False
        print('[slack] Sent alert webhook.')
        return True
    except Exception as e:
        print(f'[slack] Send failed: {e}')
        return False


if ENABLE_SLACK_ALERTS:
    force_test = _env('EWS_SLACK_TEST', '0').lower() in ('1', 'true', 'yes')
    triggered = alerts[(alerts['risk_regime'].isin(['Elevated', 'High'])) | (alerts['early_warning_flag'] == True)]

    if triggered.empty and not force_test:
        print('[slack] No triggered domains; not sending webhook.')
    else:
        nl = chr(10)
        asof_str = str(alerts['asof'].max())
        lines = []
        lines.append('[EWS] Bubble Risk Alert' if not triggered.empty else '[EWS] Slack Test Alert')
        lines.append('asof=' + asof_str)
        lines.append('')
        if triggered.empty:
            lines.append('No triggered domains (test mode).')
        else:
            lines.append('Triggered domains:')
            for domain, row in triggered.iterrows():
                lines.append(
                    f"- {domain}: {row['risk_regime']} (now={row['bubble_likelihood_now']:.1f}%, elev={row['hist_elev_threshold']:.1f}%, high={row['hist_high_threshold']:.1f}%)"
                )
        body = nl.join(lines)

        # De-duplicate (avoid spamming on repeated reruns)
        state_path = '.ews_last_slack_hash.json'
        new_hash = hashlib.sha256(body.encode('utf-8')).hexdigest()
        prev_hash = None
        try:
            if os.path.exists(state_path):
                with open(state_path, 'r', encoding='utf-8') as f:
                    prev_hash = json.load(f).get('hash')
        except Exception:
            prev_hash = None

        if new_hash == prev_hash:
            print('[slack] Alert unchanged vs last send; not resending.')
        else:
            if send_slack_webhook(body):
                try:
                    with open(state_path, 'w', encoding='utf-8') as f:
                        json.dump({'hash': new_hash, 'sent_at_utc': pd.Timestamp.utcnow().isoformat()}, f)
                except Exception as e:
                    print(f'[slack] Warning: could not persist dedupe state: {e}')


# -----------------------------
# 12.7 Lightweight historical validation (lead time + precision/recall)
# -----------------------------

VALIDATION_LEAD_DAYS = 90
VALIDATION_TRAIN_YEARS = 10
VALIDATION_TRAIN_Q = 0.80  # walk-forward threshold uses this quantile of prior history
VALIDATION_ROC_POINTS = 51

# Validation labeling upgrade:
#
# Industrial constraint we're addressing:
# - The original prototype only labeled ~3 hardcoded windows.
#
# Upgrade:
# - Load a versioned bubble-events database (CSV) with 20+ events across asset classes.
# - For each domain, use all events tagged to that domain to label "event + lead" periods.
# - Fall back to the original hardcoded windows if the database is missing/empty.
#
# NOTE: Dates in the CSV are editable anchors; this is a research notebook, not an official chronology.

BUBBLE_EVENTS_PATH = str(globals().get('BUBBLE_EVENTS_PATH', 'bubble_events_database.csv') or 'bubble_events_database.csv')

_bubble_events_db = pd.DataFrame()
try:
    if os.path.exists(BUBBLE_EVENTS_PATH):
        _bubble_events_db = pd.read_csv(BUBBLE_EVENTS_PATH)
except Exception:
    _bubble_events_db = pd.DataFrame()

if not _bubble_events_db.empty:
    # Normalize + parse dates
    for c in ['start_date', 'peak_date', 'end_date']:
        if c in _bubble_events_db.columns:
            _bubble_events_db[c] = pd.to_datetime(_bubble_events_db[c], errors='coerce')


# Fallback mapping (kept for backward compatibility / when DB is absent)
FALLBACK_DOMAIN_VALIDATION_WINDOWS = {
    'AI bubble': ['Dot-com crash (AI bubble stress)'],
    'Private Credit bubble': ['GFC (Private Credit bubble stress)'],
    'Crypto bubble': ['Crypto winter (Crypto bubble stress)'],
}


def _events_for_domain(domain: str) -> list[dict[str, object]]:
    # Return a list of {event, start_dt, end_dt} dicts.
    out: list[dict[str, object]] = []

    if isinstance(_bubble_events_db, pd.DataFrame) and (not _bubble_events_db.empty):
        try:
            dom = str(domain or '').strip().lower()
            df = _bubble_events_db.copy()
            if 'domains' in df.columns:
                mask = df['domains'].astype(str).str.lower().str.contains(dom, regex=False)
                df = df[mask]

            for _i, r in df.iterrows():
                start_dt = r.get('start_date', pd.NaT)
                end_dt = r.get('end_date', pd.NaT)
                if pd.isna(start_dt):
                    start_dt = r.get('peak_date', pd.NaT)
                if pd.isna(end_dt):
                    end_dt = r.get('peak_date', pd.NaT)
                if pd.isna(start_dt) or pd.isna(end_dt):
                    continue

                label = r.get('event_name', None) or r.get('event_id', None) or 'event'
                label = str(label)[:120]
                out.append({'event': label, 'start_dt': pd.to_datetime(start_dt), 'end_dt': pd.to_datetime(end_dt)})
        except Exception:
            out = []

    if out:
        out = sorted(out, key=lambda x: x.get('start_dt', pd.Timestamp.max))
        return out

    # Fallback to the legacy EVENT_WINDOWS keys
    wins = FALLBACK_DOMAIN_VALIDATION_WINDOWS.get(domain, [])
    if 'EVENT_WINDOWS' in globals():
        for win in wins:
            if win not in EVENT_WINDOWS:
                continue
            start, end = EVENT_WINDOWS[win]
            out.append({'event': win, 'start_dt': pd.to_datetime(start), 'end_dt': pd.to_datetime(end)})
    return out


def _label_series_for_windows(index: pd.DatetimeIndex, windows: list[str], lead_days: int) -> pd.Series:
    y = pd.Series(0, index=index, dtype=int)
    if 'EVENT_WINDOWS' not in globals():
        return y
    for win in windows:
        if win not in EVENT_WINDOWS:
            continue
        start, end = EVENT_WINDOWS[win]
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        lead_start = start_dt - pd.Timedelta(days=int(lead_days))
        y.loc[(y.index >= lead_start) & (y.index <= end_dt)] = 1
    return y


def _label_series_for_events(index: pd.DatetimeIndex, events: list[dict[str, object]], lead_days: int) -> pd.Series:
    y = pd.Series(0, index=index, dtype=int)
    for ev in events or []:
        try:
            start_dt = pd.to_datetime(ev.get('start_dt', pd.NaT), errors='coerce')
            end_dt = pd.to_datetime(ev.get('end_dt', pd.NaT), errors='coerce')
            if pd.isna(start_dt) or pd.isna(end_dt):
                continue
            lead_start = start_dt - pd.Timedelta(days=int(lead_days))
            y.loc[(y.index >= lead_start) & (y.index <= end_dt)] = 1
        except Exception:
            continue
    return y


def _confusion_metrics(alert: pd.Series, y: pd.Series) -> dict:
    common = alert.index.intersection(y.index)
    ap = alert.reindex(common).fillna(0).astype(int)
    yp = y.reindex(common).fillna(0).astype(int)

    tp = int(((ap == 1) & (yp == 1)).sum())
    fp = int(((ap == 1) & (yp == 0)).sum())
    fn = int(((ap == 0) & (yp == 1)).sum())
    tn = int(((ap == 0) & (yp == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if (np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tpr': recall,
    }


def _auc_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    mask = np.isfinite(fpr) & np.isfinite(tpr)
    if mask.sum() < 2:
        return np.nan
    fpr = fpr[mask]
    tpr = tpr[mask]
    order = np.argsort(fpr)
    x = fpr[order]
    y = tpr[order]
    try:
        if hasattr(np, 'trapezoid'):
            return float(np.trapezoid(y, x))
        elif hasattr(np, 'trapz'):
            return float(np.trapz(y, x))
    except Exception:
        pass
    # Manual trapezoid fallback
    if len(x) < 2:
        return np.nan
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def _lead_time_days(p: pd.Series, start_dt: pd.Timestamp, lead_days: int, thr: float) -> float:
    lead_start = start_dt - pd.Timedelta(days=int(lead_days))
    pre = p.loc[(p.index >= lead_start) & (p.index < start_dt)]
    if pre.empty:
        return np.nan
    first = pre[pre >= thr].index.min()
    if pd.isna(first):
        return np.nan
    return float((start_dt - first).days)


val_summary_rows = []
val_lead_rows = []

thr_grid = np.linspace(0.0, 100.0, int(VALIDATION_ROC_POINTS))

for domain in prob_df.columns:
    p = prob_df[domain].dropna().astype(float)
    if p.empty:
        continue

    events = _events_for_domain(domain)
    y = _label_series_for_events(p.index, events, VALIDATION_LEAD_DAYS)
    if int(y.sum()) == 0:
        continue

    # ROC + best-F1 threshold
    best = {'thr': np.nan, 'f1': -np.inf, 'precision': np.nan, 'recall': np.nan, 'fpr': np.nan}
    fprs = []
    tprs = []
    for thr in thr_grid:
        alert = (p >= float(thr)).astype(int)
        m = _confusion_metrics(alert, y)
        fprs.append(m['fpr'])
        tprs.append(m['tpr'])
        if np.isfinite(m['f1']) and (m['f1'] > best['f1']):
            best = {'thr': float(thr), 'f1': float(m['f1']), 'precision': float(m['precision']), 'recall': float(m['recall']), 'fpr': float(m['fpr'])}

    auc = _auc_from_roc(np.array(fprs, dtype=float), np.array(tprs, dtype=float))

    # Metrics at the current elevated threshold used in the live alerts table
    thr_elev = float(domain_thresholds.get(domain, {}).get('elev', np.nan))
    if not np.isfinite(thr_elev):
        thr_elev = float(np.percentile(p.to_numpy(), 80))
    m_elev = _confusion_metrics((p >= thr_elev).astype(int), y)

    # Walk-forward lead time: compute threshold using ONLY history prior to each event
    lead_times = []
    for ev in events:
        start_dt = pd.to_datetime(ev.get('start_dt', pd.NaT), errors='coerce')
        if pd.isna(start_dt):
            continue

        train_end = start_dt - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(years=int(VALIDATION_TRAIN_YEARS))
        train = p.loc[(p.index >= train_start) & (p.index <= train_end)].dropna()
        thr_train = float(np.percentile(train.to_numpy(), VALIDATION_TRAIN_Q * 100.0)) if len(train) else float(best['thr'])

        lt = _lead_time_days(p, start_dt, VALIDATION_LEAD_DAYS, thr_train)
        lead_times.append(lt)
        val_lead_rows.append(
            {
                'domain': domain,
                'window': str(ev.get('event', 'event'))[:120],
                'train_years': VALIDATION_TRAIN_YEARS,
                'train_q': VALIDATION_TRAIN_Q,
                'thr_train_%': thr_train,
                'lead_days': VALIDATION_LEAD_DAYS,
                'lead_time_days': lt,
            }
        )

    lead_time_mean = float(np.nanmean(lead_times)) if len(lead_times) else np.nan
    alert_burden_best = float(((p >= float(best['thr'])).astype(int)).mean()) if np.isfinite(best['thr']) else np.nan

    val_summary_rows.append(
        {
            'domain': domain,
            'auc': auc,
            'best_thr_%': best['thr'],
            'best_f1': best['f1'],
            'best_precision': best['precision'],
            'best_recall': best['recall'],
            'best_fpr': best['fpr'],
            'alert_burden_best_%days': alert_burden_best,
            'lead_time_walkfwd_days': lead_time_mean,
            'elev_thr_%': thr_elev,
            'elev_precision': float(m_elev['precision']),
            'elev_recall': float(m_elev['recall']),
        }
    )

if RUN_DETAILED_VALIDATION and val_summary_rows:
    val = pd.DataFrame(val_summary_rows).set_index('domain').sort_index()
    display(
        val.style.format(
            {
                'auc': '{:.3f}',
                'best_thr_%': '{:.1f}%',
                'best_f1': '{:.3f}',
                'best_precision': '{:.3f}',
                'best_recall': '{:.3f}',
                'best_fpr': '{:.3f}',
                'alert_burden_best_%days': '{:.2%}',
                'lead_time_walkfwd_days': '{:.0f}',
                'elev_thr_%': '{:.1f}%',
                'elev_precision': '{:.3f}',
                'elev_recall': '{:.3f}',
            }
        )
    )

# Backtest confidence (0-100) by domain, derived from validation metrics (AUC + best F1).
# - AUC is scaled so 0.50 -> 0, 1.00 -> 100, clipped at [0, 100].
# - Best F1 is used as-is (0..1), blended with AUC when available.

# Additional skill metric to mitigate limited labeled bubble windows:
# Does a higher bubble probability correlate with worse forward returns on a domain index?
forward_skill: dict[str, float] = {}
try:
    from scipy.stats import spearmanr

    FWD_H = 63  # ~3 months (trading days)
    for domain, tickers in DOMAIN_TICKERS.items():
        cols = [t for t in tickers if t in detect_prices.columns]
        if not cols:
            continue

        px = detect_prices[cols].astype(float).replace(0.0, np.nan).ffill()
        if px.dropna(how='all').empty:
            continue

        # Equal-weight domain index from normalized prices (normalization uses each column's first valid value).
        base = px.apply(lambda s: float(s.dropna().iloc[0]) if len(s.dropna()) else np.nan)
        idx = (px / base).mean(axis=1).rename('idx').dropna()
        if len(idx) < (FWD_H + 180):
            continue

        fwd_ret = (idx.shift(-FWD_H) / idx) - 1.0

        p = prob_df[domain] if ('prob_df' in globals()) and (domain in prob_df.columns) else pd.Series(dtype=float)
        df = pd.DataFrame({'p': p, 'fwd_ret': fwd_ret}).dropna()
        if len(df) < 180:
            continue

        # Higher p should imply more negative future return => correlate p with -fwd_ret
        corr = spearmanr(df['p'].to_numpy(), (-df['fwd_ret']).to_numpy()).correlation
        if corr is None or (not np.isfinite(corr)):
            continue

        forward_skill[domain] = float(np.clip(corr, 0.0, 1.0) * 100.0)
except Exception:
    forward_skill = {}

val_confidence: dict[str, float] = {}
if val_summary_rows:
    val_df = pd.DataFrame(val_summary_rows).set_index('domain').sort_index()
    for domain, row in val_df.iterrows():
        auc = float(row.get('auc', np.nan))
        f1 = float(row.get('best_f1', np.nan))

        # More interpretable mapping:
        # - AUC 0.50 ~= 50 (random), 1.00 -> 100
        # - F1 0..1 -> 0..100
        auc_score = float(np.clip(auc, 0.0, 1.0) * 100.0) if np.isfinite(auc) else np.nan
        f1_score = float(np.clip(f1, 0.0, 1.0) * 100.0) if np.isfinite(f1) else np.nan

        parts = []
        if np.isfinite(auc_score):
            parts.append(0.7 * auc_score if np.isfinite(f1_score) else auc_score)
        if np.isfinite(f1_score):
            parts.append(0.3 * f1_score if np.isfinite(auc_score) else f1_score)

        if parts:
            val_confidence[domain] = float(np.clip(sum(parts), 0.0, 100.0))


def _fallback_backtest_conf(series: pd.Series) -> float:
    s = series.dropna().astype(float)
    if s.empty:
        return np.nan

    p20 = float(np.percentile(s.to_numpy(), 20)) if len(s) else np.nan
    p80 = float(np.percentile(s.to_numpy(), 80)) if len(s) else np.nan
    spread = (p80 - p20) / 100.0 if (np.isfinite(p20) and np.isfinite(p80)) else np.nan

    n_factor = float(np.clip(len(s) / 180.0, 0.0, 1.0))
    spread_factor = float(np.clip(spread, 0.0, 1.0)) if np.isfinite(spread) else np.nan

    parts = []
    if np.isfinite(spread_factor):
        parts.append(0.7 * spread_factor)
    if np.isfinite(n_factor):
        parts.append(0.3 * n_factor)

    if not parts:
        return np.nan

    return float(np.clip(sum(parts), 0.0, 1.0) * 100.0)


# Ensure every domain gets a numeric backtest confidence (fallback if validation data is insufficient)
if 'prob_df' in globals():
    for domain in prob_df.columns:
        if domain not in val_confidence:
            val_confidence[domain] = _fallback_backtest_conf(prob_df[domain])

# Ensure a numeric default when validation/fallback are unavailable
if 'prob_df' in globals():
    for domain in prob_df.columns:
        v = val_confidence.get(domain, np.nan)
        if not np.isfinite(v):
            val_confidence[domain] = 50.0

# Blend in forward-skill metric (helps when labeled bubble windows are sparse)
if 'prob_df' in globals():
    for domain in prob_df.columns:
        skill = float(forward_skill.get(domain, np.nan))
        if np.isfinite(skill):
            base = float(val_confidence.get(domain, np.nan))
            if not np.isfinite(base):
                base = 50.0
            val_confidence[domain] = float(np.clip(0.8 * base + 0.2 * skill, 0.0, 100.0))

# Backfill into the already-built simple table
if ('simple_tbl' in globals()) and isinstance(simple_tbl, pd.DataFrame) and (not simple_tbl.empty):
    simple_tbl['Backtest Confidence (0-100)'] = pd.Series(val_confidence).reindex(simple_tbl.index).astype(float)


# -----------------------------
# 12.7b Ops: append audit log (JSONL)
# -----------------------------
if bool(globals().get('ENABLE_EWS_AUDIT_LOG', True)):
    try:
        audit_path = _env('EWS_AUDIT_LOG_PATH', '.bubble_ews_audit.jsonl')
        run_at = pd.Timestamp.now('UTC').isoformat()

        asof = None
        try:
            asof = str(alerts['asof'].max()) if ('alerts' in globals()) and isinstance(alerts, pd.DataFrame) and ('asof' in alerts.columns) else None
        except Exception:
            asof = None

        def _num_or_none(v):
            try:
                x = float(v)
                return float(x) if np.isfinite(x) else None
            except Exception:
                return None

        payload = {
            'run_at_utc': run_at,
            'asof': asof,
            'domains': {},
        }

        if ('alerts' in globals()) and isinstance(alerts, pd.DataFrame) and (not alerts.empty):
            for domain in sorted([str(x) for x in alerts.index]):
                row = alerts.loc[domain]
                dom = {
                    'prob_now': _num_or_none(row.get('bubble_likelihood_now', np.nan)),
                    'risk_regime': str(row.get('risk_regime', '')),
                    'threshold_elev': _num_or_none(row.get('hist_elev_threshold', np.nan)),
                    'threshold_high': _num_or_none(row.get('hist_high_threshold', np.nan)),
                    'early_warning_flag': bool(row.get('early_warning_flag', False)),
                    'drift_flag': bool(row.get('drift_flag', False)),
                    'drift': {
                        'hoeffding_flag': bool(row.get('hoeffding_drift_flag', False)),
                        'hoeffding_score': _num_or_none(row.get('hoeffding_drift_score', np.nan)),
                        'page_hinkley_flag': bool(row.get('page_hinkley_flag', False)),
                        'page_hinkley_score': _num_or_none(row.get('page_hinkley_score', np.nan)),
                        'cusum_flag': bool(row.get('cusum_flag', False)),
                        'cusum_score': _num_or_none(row.get('cusum_score', np.nan)),
                        'adwin_flag': bool(row.get('adwin_flag', False)),
                        'adwin_score': _num_or_none(row.get('adwin_score', np.nan)),
                    },
                    'prob_ci': {
                        'lo': _num_or_none(row.get('prob_ci_lo', np.nan)),
                        'hi': _num_or_none(row.get('prob_ci_hi', np.nan)),
                    },
                    'granger': {
                        'ok': bool(row.get('granger_ok', False)),
                        'p_min': _num_or_none(row.get('granger_p_min', np.nan)),
                        'best_lag': _num_or_none(row.get('granger_best_lag', np.nan)),
                        'n_obs': _num_or_none(row.get('granger_n_obs', np.nan)),
                        'horizon_days': _num_or_none(row.get('granger_horizon_days', np.nan)),
                        'max_lag_used': _num_or_none(row.get('granger_max_lag_used', np.nan)),
                        'alpha': _num_or_none(row.get('granger_alpha', np.nan)),
                        'significant': bool(row.get('granger_sig', False)),
                    },
                    'sec': {
                        'heat_0_100': _num_or_none(row.get('sec_heat_0_100', np.nan)),
                        'nudge_buckets': _num_or_none(row.get('sec_nudge_buckets', np.nan)),
                        'keyword_rate_per_10k': _num_or_none(row.get('sec_keyword_rate_per_10k', np.nan)),
                        'risk_rate_per_10k': _num_or_none(row.get('sec_risk_rate_per_10k', np.nan)),
                        'tickers_used': str(row.get('sec_tickers_used', '') or ''),
                    },
                    'transcripts': {
                        'heat_0_100': _num_or_none(row.get('transcript_heat_0_100', np.nan)),
                        'nudge_buckets': _num_or_none(row.get('transcript_nudge_buckets', np.nan)),
                        'keyword_rate_per_10k': _num_or_none(row.get('transcript_keyword_rate_per_10k', np.nan)),
                        'risk_rate_per_10k': _num_or_none(row.get('transcript_risk_rate_per_10k', np.nan)),
                        'tickers_used': str(row.get('transcript_tickers_used', '') or ''),
                        'docs_used': str(row.get('transcript_docs_used', '') or ''),
                    },
                    'backtest_confidence_0_100': _num_or_none(val_confidence.get(domain, np.nan)),
                }

                # BMA weights (if available)
                try:
                    if 'BMA_WEIGHTS_BY_DOMAIN' in globals() and isinstance(BMA_WEIGHTS_BY_DOMAIN, dict):
                        w = BMA_WEIGHTS_BY_DOMAIN.get(domain, {})
                        if isinstance(w, dict) and w:
                            dom['bma_weights'] = {k: _num_or_none(v) for k, v in w.items()}
                except Exception:
                    pass

                payload['domains'][domain] = dom

        with open(audit_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        print('[audit] Appended EWS audit entry: ' + audit_path)
    except Exception as e:
        print(f'[audit] Warning: could not write EWS audit log: {e}')


if RUN_DETAILED_VALIDATION and val_lead_rows:
    lead_tbl = pd.DataFrame(val_lead_rows)
    lead_tbl = lead_tbl.sort_values(['domain', 'window']).reset_index(drop=True)
    display(
        lead_tbl.style.format(
            {
                'thr_train_%': '{:.1f}%',
                'lead_time_days': '{:.0f}',
            }
        )
    )


# -----------------------------
# 12.8 Forward-looking 2026–2030 bubble-likelihood forecast (prototype)
# -----------------------------

yearly_mean_mom = pd.DataFrame()
year_end_mom = pd.DataFrame()
yearly_mean_mr = pd.DataFrame()
year_end_mr = pd.DataFrame()
meta = pd.DataFrame()

if RUN_FORECAST or INCLUDE_2026_2030_PROJECTION:
    forecast_end = pd.Timestamp('2030-12-31')
    future_probs_mom = {}
    future_probs_mr = {}
    future_meta_rows = []

    for domain in prob_df.columns:
        p_hist = prob_df[domain].dropna()
        if p_hist.empty:
            continue

        # Monthly series is more stable for long-horizon forecasting
        p_m = p_hist.resample('ME').mean().dropna()
        if len(p_m) < 36:
            continue

        # Momentum-like projection (transparent + less mean-reverting than ARIMA):
        # estimate recent slope (pp/month) and extrapolate forward, clipped to [0, 100].
        # Momentum-like projection in logit space (bounded):
        # - convert p(%) -> prob in (0,1)
        # - take logit, estimate recent momentum in logit units
        # - extrapolate with mild decay, then sigmoid back to %
        eps = 1e-3
        p_prob = (p_m / 100.0).clip(lower=eps, upper=1.0 - eps)
        x = np.log(p_prob / (1.0 - p_prob))
        dx = x.diff().dropna()
        if len(dx) < 3:
            continue

        slope_logit_per_month = float(dx.tail(3).mean())
        if len(dx) >= 12:
            slope_logit_per_month = 0.7 * slope_logit_per_month + 0.3 * float(dx.tail(12).median())
        if not np.isfinite(slope_logit_per_month):
            slope_logit_per_month = 0.0

        # Temper momentum for long horizons: shrink + tighter cap on monthly logit momentum
        slope_logit_per_month = float(slope_logit_per_month * 0.5)
        slope_logit_per_month = float(np.clip(slope_logit_per_month, -0.25, 0.25))
        x_last = float(x.iloc[-1])

        last_dt = p_m.index.max()
        start_dt = last_dt + pd.offsets.MonthEnd(1)
        future_idx = pd.date_range(start=start_dt, end=forecast_end, freq='ME')
        if len(future_idx) == 0:
            continue

        steps = np.arange(1, len(future_idx) + 1, dtype=float)
        half_life_months = 12.0
        decay = np.exp(-np.log(2.0) * (steps - 1.0) / half_life_months)
        delta = slope_logit_per_month * np.cumsum(decay)
        x_fc = x_last + delta
        prob_fc = 100.0 / (1.0 + np.exp(-x_fc))
        # Keep projections strictly away from 0/100 (and avoid rounding to 0.0% / 100.0% in the table)
        prob_fc = pd.Series(prob_fc, index=future_idx).clip(lower=0.1, upper=99.9)
        prob_fc.name = domain
        future_probs_mom[domain] = prob_fc

        # Mean-reversion projection (conservative anchor): revert logit(x) toward a historical baseline
        # with a fixed half-life. This is intentionally NOT trend-following.
        mu_logit = float(x.median()) if len(x) else x_last
        if not np.isfinite(mu_logit):
            mu_logit = x_last
        half_life_mr_months = 18.0
        decay_mr = np.exp(-np.log(2.0) * steps / half_life_mr_months)
        x_fc_mr = mu_logit + (x_last - mu_logit) * decay_mr
        prob_fc_mr = 100.0 / (1.0 + np.exp(-x_fc_mr))
        prob_fc_mr = pd.Series(prob_fc_mr, index=future_idx).clip(lower=0.1, upper=99.9)
        prob_fc_mr.name = domain
        future_probs_mr[domain] = prob_fc_mr

        thr_elev = float(domain_thresholds.get(domain, {}).get('elev', np.nan))
        thr_high = float(domain_thresholds.get(domain, {}).get('high', np.nan))
        elev_date = prob_fc[prob_fc >= thr_elev].index.min() if np.isfinite(thr_elev) else pd.NaT
        high_date = prob_fc[prob_fc >= thr_high].index.min() if np.isfinite(thr_high) else pd.NaT

        sub = prob_fc[(prob_fc.index.year >= 2026) & (prob_fc.index.year <= 2030)]
        future_meta_rows.append(
            {
                'domain': domain,
                'forecast_method': 'momentum_logit_decay',
                'slope_logit_per_month': slope_logit_per_month,
                'mean_prob_2026_2030': float(sub.mean()) if len(sub) else np.nan,
                'max_prob_2026_2030': float(sub.max()) if len(sub) else np.nan,
                'first_proj_elev_date': elev_date,
                'first_proj_high_date': high_date,
            }
        )

    future_prob_df_mom = pd.DataFrame(future_probs_mom)
    future_prob_df_mr = pd.DataFrame(future_probs_mr)

    if future_prob_df_mom.empty and future_prob_df_mr.empty:
        if RUN_FORECAST and (not SHOW_ONLY_SIMPLE_EWS_TABLE):
            print('[forecast] Not enough data to produce a 2026–2030 bubble-likelihood forecast for the configured domain tickers.')
    else:
        future_2026_2030_mom = pd.DataFrame()
        future_2026_2030_mr = pd.DataFrame()

        if not future_prob_df_mom.empty:
            future_2026_2030_mom = future_prob_df_mom[(future_prob_df_mom.index.year >= 2026) & (future_prob_df_mom.index.year <= 2030)]
            yearly_mean_mom = future_2026_2030_mom.groupby(future_2026_2030_mom.index.year).mean()
            year_end_mom = future_2026_2030_mom.groupby(future_2026_2030_mom.index.year).last()

        if not future_prob_df_mr.empty:
            future_2026_2030_mr = future_prob_df_mr[(future_prob_df_mr.index.year >= 2026) & (future_prob_df_mr.index.year <= 2030)]
            yearly_mean_mr = future_2026_2030_mr.groupby(future_2026_2030_mr.index.year).mean()
            year_end_mr = future_2026_2030_mr.groupby(future_2026_2030_mr.index.year).last()

        if RUN_FORECAST and (not SHOW_ONLY_SIMPLE_EWS_TABLE):
            if not yearly_mean_mom.empty:
                print("")
                print("[forecast] Yearly mean bubble likelihood (%) (2026–2030) — Momentum:")
                display(yearly_mean_mom.style.format('{:.1f}%'))
            if not yearly_mean_mr.empty:
                print("")
                print("[forecast] Yearly mean bubble likelihood (%) (2026–2030) — Mean-reversion:")
                display(yearly_mean_mr.style.format('{:.1f}%'))

        meta = pd.DataFrame(future_meta_rows).set_index('domain').sort_index()
        meta_display = meta.copy()
        for c in ['first_proj_elev_date', 'first_proj_high_date']:
            meta_display[c] = pd.to_datetime(meta_display[c]).dt.strftime('%Y-%m-%d')
        if RUN_FORECAST and (not SHOW_ONLY_SIMPLE_EWS_TABLE):
            display(
                meta_display.style.format(
                    {
                        'mean_prob_2026_2030': '{:.1f}%',
                        'max_prob_2026_2030': '{:.1f}%',
                    }
                )
            )

        if RUN_FORECAST and (not SHOW_ONLY_SIMPLE_EWS_TABLE):
            # Plot forecast paths (monthly)
            plt.figure(figsize=(12, 4))
            if not future_2026_2030_mom.empty:
                for d in future_2026_2030_mom.columns:
                    plt.plot(future_2026_2030_mom.index, future_2026_2030_mom[d], label=f'{d} (mom)')
            if not future_2026_2030_mr.empty:
                for d in future_2026_2030_mr.columns:
                    plt.plot(future_2026_2030_mr.index, future_2026_2030_mr[d], linestyle='--', alpha=0.8, label=f'{d} (mr)')
            plt.axhline(80, color='red', linestyle='--', alpha=0.5)
            plt.axhline(60, color='orange', linestyle='--', alpha=0.5)
            plt.title('Forecast bubble likelihood (2026–2030, prototype) — momentum vs mean-reversion')
            plt.ylabel('Bubble likelihood (%)')
            plt.legend()
            plt.tight_layout()
            plt.show()


# Append 2026–2030 year-end projections into the simple table (still one table output)
# We emit both momentum + mean-reversion columns for side-by-side comparison.
if INCLUDE_2026_2030_PROJECTION:
    if not year_end_mom.empty:
        for y in range(2026, 2031):
            col = f'{y}E Mom (%)'
            if y in year_end_mom.index:
                simple_tbl[col] = year_end_mom.loc[y].reindex(simple_tbl.index)
    if not year_end_mr.empty:
        for y in range(2026, 2031):
            col = f'{y}E MR (%)'
            if y in year_end_mr.index:
                simple_tbl[col] = year_end_mr.loc[y].reindex(simple_tbl.index)


# Add deterministic inputs coverage into the table (tickers with recent usable prices)
try:
    if 'domain_coverage' in globals() and isinstance(domain_coverage, dict) and domain_coverage:
        simple_tbl['Inputs Coverage (tickers ok/total)'] = pd.Series(domain_coverage).reindex(simple_tbl.index).astype(str)
except Exception:
    pass


if SHOW_ONLY_SIMPLE_EWS_TABLE:
    drop_cols = [
        'Nudge (+buckets)',
        'News articles (24h)',
        'News sentiment (avg)',
        'News heat (0-100)',
        'Alpha news status',
        'Yahoo headline sentiment (avg)',
        'Yahoo headlines (count)',
    ]
    simple_view = simple_tbl.drop(columns=['__bucket__'] + drop_cols, errors='ignore')

    fmt = {
        'Probability Now (%)': '{:.1f}%',
        'Confidence (0-100)': '{:.0f}',
        'Backtest Confidence (0-100)': '{:.0f}',
        'LPPL endgame (0-100)': '{:.0f}',
        'LPPL tc (days)': '{:.0f}',
        '30D Change (pp)': '{:+.1f}',
    }
    if 'Nudge (+buckets)' in simple_view.columns:
        fmt['Nudge (+buckets)'] = '{:+.0f}'
    if 'SEC heat (0-100)' in simple_view.columns:
        fmt['SEC heat (0-100)'] = '{:.0f}'
    if 'Transcript heat (0-100)' in simple_view.columns:
        fmt['Transcript heat (0-100)'] = '{:.0f}'
    if 'News articles (24h)' in simple_view.columns:
        fmt['News articles (24h)'] = '{:.0f}'
    if 'News sentiment (avg)' in simple_view.columns:
        fmt['News sentiment (avg)'] = '{:+.3f}'
    if 'News heat (0-100)' in simple_view.columns:
        fmt['News heat (0-100)'] = '{:.0f}'
    if 'Yahoo headline sentiment (avg)' in simple_view.columns:
        fmt['Yahoo headline sentiment (avg)'] = '{:+.3f}'
    if 'Yahoo headlines (count)' in simple_view.columns:
        fmt['Yahoo headlines (count)'] = '{:.0f}'
    for y in range(2026, 2031):
        for suffix in ['Mom', 'MR']:
            col = f'{y}E {suffix} (%)'
            if col in simple_view.columns:
                fmt[col] = '{:.1f}%'

    display(simple_view.style.format(fmt, na_rep='—').apply(_style_bucket, axis=1))

    # Dynamic interpretation (updates with live yfinance data)
    if 'explain' in globals():
        try:
            bullets = []
            asof = None
            try:
                asof = alerts['asof'].max() if ('alerts' in globals()) and ('asof' in alerts.columns) else None
            except Exception:
                asof = None
            if asof is not None:
                bullets.append(f"As-of date in this table: {asof}")

            # Column definitions (outsider-friendly)
            bullets.extend(
                [
                    '**How to read the early-warning dashboard (Section 12):**',
                    '**Model card framing:** Level (Probability Now) + Trend (30D Change/Trend) + Drift (Drift Flag) + Evidence (Confidence + Backtest Confidence).',
                    'Severity: a color bucket derived from Probability Now (%) and recent momentum (30D Change).',
                    'Probability Now (%): the model’s current bubble-likelihood estimate from price-based indicators (0–100).',
                    'Prob CI (90%): block-bootstrap confidence interval for Probability Now (%) (best-effort; blank when disabled).',
                    '30D Change (pp): change over ~30 days in the smoothed probability (positive = rising risk).',
                    'Trend: a plain label based on 30D Change (Flat / Rising / Rising fast / Falling...).',
                    'Drift Flag: OR across drift detectors (Hoeffding, Page–Hinkley, CUSUM, ADWIN) on the probability series (Yes = significant upward shift vs baseline).',
                    'Confidence (0–100): stability/decisiveness + data sufficiency + recency (not a guarantee).',
                    'Backtest Confidence (0–100): blend of walk-forward validation (AUC/F1) plus forward-drawdown skill; higher means the signal historically lined up better with future stress.',
                    'LPPL endgame (0-100): supplementary bubble “endgame” indicator (higher = LPPL fit suggests a nearer critical time).',
                    'LPPL tc (days): LPPL-implied days until the estimated critical time tc (best-effort; NaN when no stable fit).',
                    '2026E Mom (%) … 2030E Mom (%): scenario — momentum-style year-end projections (trend-following with decay, bounded in logit space).',
                    '2026E MR (%) … 2030E MR (%): scenario — mean-reversion year-end projections (revert toward a historical baseline; conservative anchor).',
                ]
            )

            try:
                if 'SEC heat (0-100)' in simple_view.columns:
                    bullets.append('SEC heat (0-100): filings-based disclosure/tone proxy from recent SEC 10-K/10-Q text (best-effort; requires EWS_SEC_USER_AGENT).')
            except Exception:
                pass

            try:
                if 'Transcript heat (0-100)' in simple_view.columns:
                    bullets.append(
                        'Transcript heat (0-100): local earnings-call transcript text proxy (best-effort; set RUN_TRANSCRIPTS_NLP=True and populate transcripts/ or EWS_TRANSCRIPTS_DIR).'
                    )
            except Exception:
                pass

            # Run-specific inputs coverage (how many domain tickers had recent price data)
            try:
                if 'domain_coverage' in globals() and isinstance(domain_coverage, dict) and domain_coverage:
                    parts = [f"{k}: {domain_coverage[k]}" for k in sorted(domain_coverage.keys())]
                    if parts:
                        bullets.append('Inputs coverage (tickers w/ recent prices): ' + '; '.join(parts))
            except Exception:
                pass

            # Quick “what’s happening now” summary
            try:
                pcol = pd.to_numeric(simple_view.get('Probability Now (%)'), errors='coerce')
                if pcol is not None and pcol.notna().any():
                    top_now = pcol.idxmax()
                    bullets.append(f"Highest current probability: {top_now} ({float(pcol.loc[top_now]):.1f}%).")
            except Exception:
                pass
            try:
                dcol = pd.to_numeric(simple_view.get('30D Change (pp)'), errors='coerce')
                if dcol is not None and dcol.notna().any():
                    top_rise = dcol.idxmax()
                    bullets.append(f"Fastest 30D rise: {top_rise} ({float(dcol.loc[top_rise]):+,.1f}pp).")
            except Exception:
                pass
            try:
                drift_col = simple_view.get('Drift Flag')
                if drift_col is not None:
                    drift_yes = [b for b in simple_view.index if str(simple_view.loc[b].get('Drift Flag', '')).strip().lower() == 'yes']
                    if drift_yes:
                        bullets.append('Drift flagged: ' + ', '.join(drift_yes))
            except Exception:
                pass

            bullets.append('')
            bullets.append('**Per-bubble snapshot (what matters):**')
            for bubble in simple_view.index:
                r = simple_view.loc[bubble]
                p = float(r.get('Probability Now (%)', np.nan))
                d30 = float(r.get('30D Change (pp)', np.nan))
                sev = str(r.get('Severity', ''))
                conf = float(r.get('Confidence (0-100)', np.nan))
                bt = float(r.get('Backtest Confidence (0-100)', np.nan))
                drift = str(r.get('Drift Flag', ''))
                trend = str(r.get('Trend', ''))

                drivers = ''
                try:
                    if 'domain_driver_summary' in globals() and isinstance(domain_driver_summary, dict):
                        drivers = str(domain_driver_summary.get(bubble, '') or '')
                except Exception:
                    drivers = ''

                bma = ''
                try:
                    if 'BMA_WEIGHTS_BY_DOMAIN' in globals() and isinstance(BMA_WEIGHTS_BY_DOMAIN, dict):
                        w = BMA_WEIGHTS_BY_DOMAIN.get(bubble, {})
                        if isinstance(w, dict) and w:
                            parts = sorted(w.items(), key=lambda kv: float(kv[1]), reverse=True)
                            bma = ','.join([f"{k}={float(v):.2f}" for k, v in parts])
                except Exception:
                    bma = ''

                extra_parts = []
                if drivers:
                    extra_parts.append(f"drivers={drivers}")
                if bma:
                    extra_parts.append(f"bma={bma}")
                extra = (" | " + " | ".join(extra_parts)) if extra_parts else ''

                bullets.append(
                    f"{bubble}: {sev} | now={p:.1f}% | 30D={d30:+.1f}pp ({trend}) | drift={drift} | model_conf={conf:.0f}/100 | backtest_conf={bt:.0f}/100{extra}"
                )

            explain(
                '12) Bubble early-warning table (how to read)',
                bullets
                + [
                    'All values are snapshot/live and will change when Yahoo Finance data changes.',
                    'If you want a conservative workflow: treat “Orange/Red + Rising” as a prompt to review exposure and hedges, not as a deterministic prediction.',
                ],
            )
        except Exception:
            pass


# [removed] Visualization charts omitted per request (AI bubble, Crypto bubble, Private Credit bubble).
print('[viz] Bubble likelihood charts omitted per request.')

# --------------- SHAP Waterfall Chart (top drivers per domain) ---------------
try:
    import matplotlib.pyplot as plt
    _shap_store = globals().get('domain_shap_values', {})
    if _shap_store:
        n_domains = len(_shap_store)
        fig_shap, axes_shap = plt.subplots(1, n_domains, figsize=(6 * n_domains, 5), squeeze=False)
        for idx_d, (dom, sv) in enumerate(sorted(_shap_store.items())):
            ax = axes_shap[0, idx_d]
            top = sorted(sv.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            top = list(reversed(top))  # smallest at top for waterfall layout
            names = [t[0] for t in top]
            vals = [t[1] for t in top]
            colors = ['#d62728' if v > 0 else '#1f77b4' for v in vals]
            ax.barh(names, vals, color=colors)
            ax.set_title(f'{dom} — SHAP drivers', fontsize=11)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel('SHAP value (φ = w·(x−E[x]))')
        plt.suptitle('Exact Linear Shapley Values — Top Bubble-Score Drivers', fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
        plt.show()
        print('[shap] Waterfall chart rendered and saved to shap_waterfall.png')
except Exception as _shap_ex:
    print(f'[shap] Waterfall chart skipped: {_shap_ex}')
"""

    rebalance_md = """## 13) Bubble-aware Automated Portfolio Rebalancing Engine (rule-based, refresh-safe)

This section turns the **bubble early warning system** outputs (Section 12) into a **systematic rebalancing recommendation**.

In plain English: when bubble risk rises, the engine **reduces** exposure to assets that historically move *with* the bubble proxies and **increases** exposure to assets that hedge them — while staying inside roster risk budgets and anti-churn guardrails.

**Inputs (live):**
- Section 12 alerts (bubble likelihood, regimes, early-warning flags)
- Recent Yahoo Finance return history (assets + composite bubble factors)
- Your roster baseline weights + risk limits

**Outputs:**
- Recommended weights (%) and LONG/SHORT change suggestions
- A refresh-safe CSV export: `rebalance_recommendations_latest.csv`

**Design goals:**
- Refresh-safe: every rerun (fresh Yahoo Finance data) recomputes signals and updates recommendations.
- Transparent + auditable: simple rules, explicit thresholds, and clear “why” columns.
- Supports **sign flips** (LONG ↔ SHORT) when bubble regimes change, while still respecting roster risk budgets.

**Important:** This is a **research / decision-support tool**. It produces suggested weights + a trade list export; it does not execute trades.
"""

    rebalance_code = """# -----------------------------
# 13) Bubble-aware Automated Portfolio Rebalancing Engine (rule-based)
# -----------------------------

# (Outputs intentionally minimal: user-facing table only)

# ---------- Config knobs ----------
# Lookback window used to estimate bubble-factor betas (daily OLS or time-varying RLS/Kalman-style)
REBAL_LOOKBACK_DAYS = 252 * 3  # ~3 years

# Beta estimation method: 'ols' or 'rls' (time-varying)
BETA_ESTIMATOR = str(globals().get('BETA_ESTIMATOR', 'rls') or 'rls').strip().lower()
BETA_RLS_LAMBDA_BASE = 0.995
BETA_RLS_LAMBDA_MIN = 0.97
BETA_RLS_DELTA = 1000.0
BETA_RLS_RIDGE = 1e-8

# How aggressive the sign-flip logic is
BETA_FLIP_THRESHOLD = 0.10     # minimum |risk_score| required to flip side
MAX_SIGN_FLIPS = 12            # cap number of assets that can flip in one run (anti-churn)

# How aggressively to size up/down around the roster baseline
TILT_STRENGTH = 0.35           # multiplier strength applied via tanh(risk_score)
MIN_MULT = 0.25
MAX_MULT = 1.75

# Cap how far abs(weights) can move away from the roster baseline abs weights
MAX_RISK_MULT = 1.50

# Optional export
SAVE_REBALANCE_OUTPUTS = True
OUTPUT_LATEST_CSV = 'rebalance_recommendations_latest.csv'




# ---------- Guardrails / inputs ----------
required = ['best_w', 'rets', 'alerts', 'bubble_factor_returns', 'asset_tickers', 'GROSS_EXPOSURE']
missing = [k for k in required if k not in globals()]
if missing:
    raise RuntimeError(f"[rebalance] Missing prerequisite variables (run prior sections): {missing}")

base_w = best_w.reindex(asset_tickers).fillna(0.0).astype(float)
base_abs = base_w.abs()
base_sign = np.sign(base_w)
base_sign = base_sign.replace({0.0: 1.0})  # treat flat as long for sign math

# Bubble regime severity (domain-level)
# Institutional-style: allow continuous severity from the model probability, not just discrete buckets.
USE_CONTINUOUS_SEVERITY = True

SEVERITY = {'Normal': 0.0, 'Elevated': 1.0, 'High': 2.0}


def _severity_from_alert_row(row: pd.Series) -> float:
    if bool(USE_CONTINUOUS_SEVERITY):
        p = row.get('bubble_likelihood_now', np.nan)
        try:
            p = float(p)
        except Exception:
            p = np.nan
        # Map 0..100% -> 0..2 severity (50% == 1.0)
        sev = float(np.clip(p / 50.0, 0.0, 2.0)) if np.isfinite(p) else 0.0
    else:
        sev = float(SEVERITY.get(str(row.get('risk_regime', 'Normal')), 0.0))

    # Small bump for rising risk (anti-lag)
    if bool(row.get('early_warning_flag', False)):
        sev += 0.25

    # News nudge: bump severity when Alpha (and/or Yahoo headlines) indicate extreme-negative news flow.
    try:
        n = float(row.get('news_nudge_buckets', 0.0))
        if np.isfinite(n) and n > 0:
            sev += 0.35 * float(n)
    except Exception:
        pass
    return float(np.clip(sev, 0.0, 2.5))


severity_by_domain = {}
for domain, row in alerts.iterrows():
    severity_by_domain[str(domain)] = _severity_from_alert_row(row)

max_sev = max(severity_by_domain.values()) if severity_by_domain else 0.0

# Bubble factor returns are domain-level composites computed in Section 3.
if bubble_factor_returns is None or bubble_factor_returns.empty:
    raise RuntimeError('[rebalance] bubble_factor_returns missing/empty; run Section 3 first.')

factor_cols = [c for c in bubble_factor_returns.columns if c in severity_by_domain]
if not factor_cols:
    factor_cols = list(bubble_factor_returns.columns)
if not factor_cols:
    raise RuntimeError('[rebalance] No bubble factor columns available.')


def estimate_factor_betas_daily(asset_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
    # Estimate daily OLS betas: asset ~ const + factors.
    # Returns: DataFrame index=asset, columns=factor tickers.
    out = pd.DataFrame(index=asset_returns.columns, columns=factor_returns.columns, dtype=float)
    for a in asset_returns.columns:
        df = pd.concat([asset_returns[a].rename('__asset__'), factor_returns], axis=1, join='inner').dropna()
        if df.shape[0] < 120:
            continue
        y = df['__asset__']
        X = sm.add_constant(df[factor_returns.columns], has_constant='add')
        try:
            fit = sm.OLS(y, X).fit()
        except Exception:
            continue
        for f in factor_returns.columns:
            out.loc[a, f] = float(fit.params.get(f, np.nan))
    return out


def estimate_factor_betas_timevarying_rls(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    lam: float,
) -> pd.DataFrame:
    # Time-varying betas via RLS (Kalman-equivalent random-walk state-space).
    # Returns last betas per asset.
    lam = float(lam)
    lam = float(np.clip(lam, 0.90, 0.999))
    delta = float(BETA_RLS_DELTA)
    ridge = float(BETA_RLS_RIDGE)

    factors = list(factor_returns.columns)
    out = pd.DataFrame(index=asset_returns.columns, columns=factors, dtype=float)

    m = int(len(factors)) + 1  # const + betas
    I = np.eye(m, dtype=float)

    for a in asset_returns.columns:
        df = pd.concat([asset_returns[a].rename('__asset__'), factor_returns], axis=1, join='inner')
        if df.shape[0] < 120:
            continue

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = pd.to_numeric(df['__asset__'], errors='coerce').astype(float).fillna(0.0).to_numpy(dtype=float)
        X = df[factors].apply(pd.to_numeric, errors='coerce').astype(float).fillna(0.0).to_numpy(dtype=float)
        if X.shape[0] != y.shape[0] or X.shape[0] < 120:
            continue

        theta = np.zeros(m, dtype=float)
        P = float(delta) * I

        x = np.empty(m, dtype=float)
        for t in range(X.shape[0]):
            x[0] = 1.0
            x[1:] = X[t, :]
            Px = P @ x
            denom = float(lam + float(x @ Px) + ridge)
            if (not np.isfinite(denom)) or denom <= 1e-12:
                continue
            K = Px / denom
            err = float(y[t] - float(x @ theta))
            theta = theta + K * err
            P = (P - np.outer(K, Px)) / float(lam)

        for j, f in enumerate(factors):
            v = float(theta[j + 1])
            out.loc[a, f] = v if np.isfinite(v) else np.nan

    return out


# Estimate betas on recent data
asset_r = rets[asset_tickers].tail(REBAL_LOOKBACK_DAYS).dropna(how='all')
factor_r = bubble_factor_returns.reindex(rets.index)[factor_cols].tail(REBAL_LOOKBACK_DAYS).dropna(how='all')

beta = pd.DataFrame()
beta_method_used = 'ols'
beta_rls_lambda_used = np.nan
try:
    # Regime-conditional adaptation: when bubble severity is high, use a lower lambda (faster beta drift).
    beta_rls_lambda_used = float(
        np.clip(
            float(BETA_RLS_LAMBDA_BASE) - 0.01 * float(max_sev),
            float(BETA_RLS_LAMBDA_MIN),
            float(BETA_RLS_LAMBDA_BASE),
        )
    )
except Exception:
    beta_rls_lambda_used = float(BETA_RLS_LAMBDA_BASE)

if str(BETA_ESTIMATOR) in ('rls', 'kalman', 'kalman_rls', 'timevarying'):
    beta = estimate_factor_betas_timevarying_rls(asset_r.fillna(0.0), factor_r.fillna(0.0), lam=beta_rls_lambda_used)
    beta_method_used = 'rls'

if beta is None or beta.empty or beta.dropna(how='all').empty:
    beta = estimate_factor_betas_daily(asset_r.fillna(0.0), factor_r.fillna(0.0))
    beta_method_used = 'ols'

# Build per-asset risk score as severity-weighted factor exposure
risk_score = pd.Series(0.0, index=asset_tickers)
contrib = pd.DataFrame(index=asset_tickers)

for domain, sev in severity_by_domain.items():
    fac = str(domain)
    if fac not in beta.columns:
        continue
    c = float(sev) * beta[fac].reindex(asset_tickers).astype(float)
    contrib[fac] = c
    risk_score = risk_score.add(c.fillna(0.0), fill_value=0.0)


def side_from_weight(w: float) -> str:
    if w > 0:
        return 'LONG'
    if w < 0:
        return 'SHORT'
    return 'FLAT'


# ---------- Decide sign flips (optional, anti-churn cap) ----------
reco_sign = base_sign.copy()
flip_assets = []

if max_sev > 0:
    for a, val in risk_score.abs().sort_values(ascending=False).items():
        if len(flip_assets) >= MAX_SIGN_FLIPS:
            break
        raw = float(risk_score.get(a, 0.0))
        if not np.isfinite(raw) or abs(raw) < BETA_FLIP_THRESHOLD:
            # Sorted desc; once under threshold, remaining will be too.
            break
        desired = (-1.0 if raw > 0 else 1.0)
        if float(reco_sign.get(a, 1.0)) != desired:
            reco_sign[a] = desired
            flip_assets.append(a)


# ---------- Size tilts around baseline (rule-based fallback) ----------
# tanh keeps it stable; positive risk -> downsize, negative risk -> upsize
r = np.tanh(risk_score.fillna(0.0))
mult = (1.0 - TILT_STRENGTH * r).clip(lower=MIN_MULT, upper=MAX_MULT)

abs_tilted = base_abs * mult

# Per-asset cap: don’t exceed MAX_RISK_MULT x the roster baseline abs weight
cap_abs = (base_abs * MAX_RISK_MULT)

# Global cap as a final safety net (uses existing MAX_ABS_WEIGHT from config section)
cap_abs = cap_abs.clip(upper=float(MAX_ABS_WEIGHT))

abs_capped = pd.concat([abs_tilted, cap_abs], axis=1).min(axis=1)

w_raw = reco_sign * abs_capped
gross = float(w_raw.abs().sum())
if gross <= 0:
    raise RuntimeError('[rebalance] Degenerate rebalance result (gross==0).')
w_rule = w_raw * (GROSS_EXPOSURE / gross)


# ---------- Mean–CVaR optimizer (minimal dependencies; SciPy linprog) ----------
# Goal: maximize a regime-conditional expected return while penalizing tail risk (CVaR)
# and transaction costs (turnover) — all via a linear program.
USE_MEAN_CVAR_OPTIMIZER = True
CVAR_ALPHA = 0.95
BASE_RISK_AVERSION = 5.0
TCOST_BPS = 10.0  # per unit notional traded (simple linear model)
ENFORCE_NET_EXPOSURE = False
import os

# Turnover = 0.5 * sum_i |delta_i|. Default to a conservative cap (can be overridden via env).
# - Set EWS_MAX_TURNOVER="0.35" for explicit cap (fraction of notional traded)
# - Set EWS_MAX_TURNOVER="none" to disable the hard cap
MAX_TURNOVER = 0.35
try:
    _mt = str(os.getenv('EWS_MAX_TURNOVER', '') or '').strip().lower()
    if _mt in ('', 'default'):
        pass
    elif _mt in ('none', 'null', 'off', 'disable', 'disabled'):
        MAX_TURNOVER = None
    else:
        MAX_TURNOVER = float(_mt)
except Exception:
    pass

# Regime-conditional expected return adjustment:
# When bubble severity is high, penalize positive bubble-factor exposure (crash risk)
# and slightly reward negative exposure (hedges).
MU_EWMA_HALFLIFE_DAYS = 252
MU_RISK_ADJ_DAILY = 0.0005  # daily return units (tune as needed)


def _estimate_mu_daily(asset_returns: pd.DataFrame) -> pd.Series:
    if asset_returns is None or asset_returns.empty:
        return pd.Series(dtype=float)
    try:
        return asset_returns.ewm(halflife=int(MU_EWMA_HALFLIFE_DAYS), min_periods=60, adjust=False).mean().iloc[-1].astype(float)
    except Exception:
        return asset_returns.mean().astype(float)


def _mean_cvar_optimize(
    asset_returns: pd.DataFrame,
    base_w: pd.Series,
    reco_sign: pd.Series,
    cap_abs: pd.Series,
    gross_exposure: float,
    risk_score: pd.Series,
    max_sev: float,
) -> tuple[pd.Series, dict]:
    from scipy.optimize import linprog

    cols = list(base_w.index)
    R = asset_returns.reindex(columns=cols).dropna(how='all').fillna(0.0)
    if R.shape[0] < 120:
        raise RuntimeError(f'[rebalance] Not enough return history for CVaR optimization (rows={R.shape[0]}).')

    s = reco_sign.reindex(cols).fillna(1.0).astype(float).replace({0.0: 1.0})

    cap = cap_abs.reindex(cols).fillna(float(MAX_ABS_WEIGHT)).astype(float).clip(lower=0.0)

    mu_base = _estimate_mu_daily(R)
    rs = risk_score.reindex(cols).fillna(0.0).astype(float)
    rs_sd = float(rs.std(ddof=0))
    rs_z = (rs - float(rs.mean())) / (rs_sd + 1e-12)
    mu = mu_base - float(MU_RISK_ADJ_DAILY) * float(max_sev) * rs_z

    alpha = float(CVAR_ALPHA)
    if not (0.50 < alpha < 0.995):
        alpha = 0.95

    gamma = float(BASE_RISK_AVERSION) * (1.0 + float(max_sev))

    # Transaction cost coefficient applies to sum |delta|; turnover is 0.5 * sum |delta|.
    tc_coeff = float(TCOST_BPS) / 20000.0

    X = R.to_numpy(dtype=float)  # shape T x N
    Tn, N = X.shape

    # variable layout: [a0..aN-1, z, u0..uT-1, d0..dN-1]
    idx_a0 = 0
    idx_z = N
    idx_u0 = N + 1
    idx_d0 = N + 1 + Tn
    nvar = N + 1 + Tn + N

    c = np.zeros(nvar, dtype=float)
    c[idx_a0:idx_a0 + N] = (-mu.reindex(cols).fillna(0.0).to_numpy(dtype=float) * s.to_numpy(dtype=float))
    c[idx_z] = float(gamma)
    c[idx_u0:idx_u0 + Tn] = float(gamma) / (float((1.0 - alpha)) * float(Tn))
    c[idx_d0:idx_d0 + N] = float(tc_coeff)

    A_eq = []
    b_eq = []

    # gross exposure: sum a = gross
    row = np.zeros(nvar, dtype=float)
    row[idx_a0:idx_a0 + N] = 1.0
    A_eq.append(row)
    b_eq.append(float(gross_exposure))

    if bool(ENFORCE_NET_EXPOSURE):
        target_net = float(base_w.sum())
        row2 = np.zeros(nvar, dtype=float)
        row2[idx_a0:idx_a0 + N] = s.to_numpy(dtype=float)
        A_eq.append(row2)
        b_eq.append(float(target_net))

    A_ub = []
    b_ub = []

    # CVaR constraints: loss_t - z - u_t <= 0, where loss_t = -sum_i (s_i * a_i * r_ti)
    svec = s.to_numpy(dtype=float)
    for t in range(Tn):
        row = np.zeros(nvar, dtype=float)
        row[idx_a0:idx_a0 + N] = -(svec * X[t, :])
        row[idx_z] = -1.0
        row[idx_u0 + t] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    w0 = base_w.reindex(cols).fillna(0.0).astype(float).to_numpy(dtype=float)

    # d_i >= (s_i*a_i - w0_i)  => s_i*a_i - d_i <= w0_i
    for i in range(N):
        row = np.zeros(nvar, dtype=float)
        row[idx_a0 + i] = float(svec[i])
        row[idx_d0 + i] = -1.0
        A_ub.append(row)
        b_ub.append(float(w0[i]))

    # d_i >= -(s_i*a_i - w0_i) => -s_i*a_i - d_i <= -w0_i
    for i in range(N):
        row = np.zeros(nvar, dtype=float)
        row[idx_a0 + i] = -float(svec[i])
        row[idx_d0 + i] = -1.0
        A_ub.append(row)
        b_ub.append(-float(w0[i]))

    if MAX_TURNOVER is not None:
        # 0.5 * sum d_i <= MAX_TURNOVER
        row = np.zeros(nvar, dtype=float)
        row[idx_d0:idx_d0 + N] = 0.5
        A_ub.append(row)
        b_ub.append(float(MAX_TURNOVER))

    bounds = []
    for i in range(N):
        bounds.append((0.0, float(cap.iloc[i])))
    bounds.append((None, None))  # z
    bounds.extend([(0.0, None)] * Tn)  # u
    bounds.extend([(0.0, None)] * N)  # d

    res = linprog(
        c,
        A_ub=np.asarray(A_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        A_eq=np.asarray(A_eq, dtype=float),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=bounds,
        method='highs',
    )
    if not getattr(res, 'success', False):
        msg = getattr(res, 'message', 'unknown failure')
        raise RuntimeError(f'[rebalance] linprog failed: {msg}')

    a = np.asarray(res.x[idx_a0:idx_a0 + N], dtype=float)
    w = (svec * a)
    w = pd.Series(w, index=cols).astype(float)

    port = (X @ w.to_numpy(dtype=float))
    loss = -port
    var = float(np.quantile(loss, alpha))
    tail = loss[loss >= var - 1e-12]
    cvar = float(np.mean(tail)) if len(tail) else float('nan')
    exp = float(np.dot(mu.reindex(cols).fillna(0.0).to_numpy(dtype=float), w.to_numpy(dtype=float)))

    d = np.asarray(res.x[idx_d0:idx_d0 + N], dtype=float)
    turnover = 0.5 * float(np.sum(np.abs(d)))

    meta = {
        'exp_ret_daily': exp,
        'cvar_loss': cvar,
        'turnover': turnover,
        'risk_aversion': gamma,
    }
    return w, meta


# Choose final weights: optimizer (if available) else rule-based
w_reco = w_rule.copy()
opt_meta = {}
if bool(USE_MEAN_CVAR_OPTIMIZER):
    try:
        w_opt, opt_meta = _mean_cvar_optimize(asset_r, base_w, reco_sign, cap_abs, float(GROSS_EXPOSURE), risk_score, float(max_sev))
        if w_opt is not None and float(w_opt.abs().sum()) > 0:
            w_opt = w_opt * (float(GROSS_EXPOSURE) / float(w_opt.abs().sum()))
            w_reco = w_opt.reindex(asset_tickers).fillna(0.0)
    except Exception:
        opt_meta = {}


# ---------- Reporting ----------
delta = (w_reco - base_w)
turnover = 0.5 * float(delta.abs().sum())



# Minimal user-facing table
curr_side = [side_from_weight(x) for x in base_w.values]
reco_side = [side_from_weight(x) for x in w_reco.values]
long_short_change = [f"{c} -> {r}" if c != r else f"Stay {c}" for c, r in zip(curr_side, reco_side)]
delta_abs = (w_reco.values * 100.0) - (base_w.values * 100.0)

# Live asset name (best-effort): prefer Yahoo Finance name lookup when available.
def _asset_name(ticker: str) -> str:
    nm = ''
    if 'yahoo_name' in globals():
        try:
            nm = yahoo_name(ticker) or ''
        except Exception:
            nm = ''
    if (not nm) and ('name_by_ticker' in globals()):
        try:
            nm = str(name_by_ticker.get(ticker, '') or '')
        except Exception:
            nm = ''
    return nm

asset_names = [_asset_name(t) for t in asset_tickers]

simple_tbl = pd.DataFrame(
    {
        'Ticker': asset_tickers,
        'Ticker Asset Name': asset_names,
        'Current Weight (%)': base_w.values * 100.0,
        'Recommended Weight (%)': w_reco.values * 100.0,
        'Long/Short Change': long_short_change,
        '__delta_abs__': np.abs(delta_abs),
    }
)

simple_tbl = simple_tbl.sort_values('__delta_abs__', ascending=False)

display(
    simple_tbl[['Ticker', 'Ticker Asset Name', 'Current Weight (%)', 'Recommended Weight (%)', 'Long/Short Change']]
    .reset_index(drop=True)
    .style.format({'Current Weight (%)': '{:+.2f}', 'Recommended Weight (%)': '{:+.2f}'})
)

# Dynamic interpretation (updates with live yfinance data)
if 'explain' in globals():
    try:
        bullets = []
        # Outsider-friendly summary of what this engine is doing
        try:
            bullets.append('This engine is rule-based: it converts bubble risk (Section 12) + factor exposures into portfolio tilts and (optionally) limited sign flips.')
            bullets.append(f"Lookback used for factor betas: ~{int(REBAL_LOOKBACK_DAYS)} trading days")
        except Exception:
            pass
        try:
            if str(globals().get('beta_method_used', '') or '').strip().lower() in ('rls', 'kalman', 'kalman_rls', 'timevarying'):
                lam = float(globals().get('beta_rls_lambda_used', np.nan))
                if np.isfinite(lam):
                    bullets.append(f"Beta estimator: time-varying (RLS/Kalman-style), λ≈{lam:.3f}")
                else:
                    bullets.append("Beta estimator: time-varying (RLS/Kalman-style)")
            else:
                bullets.append("Beta estimator: OLS (static)")
        except Exception:
            pass
        try:
            if 'severity_by_domain' in globals() and isinstance(severity_by_domain, dict) and severity_by_domain:
                top_domain = max(severity_by_domain, key=severity_by_domain.get)
                bullets.append(f"Primary bubble driver right now: {top_domain} (severity={float(severity_by_domain.get(top_domain, 0.0)):.2f})")
        except Exception:
            pass
        try:
            bullets.append(f"Estimated turnover: {turnover:.1%} (higher = more trading/churn)")
        except Exception:
            pass
        try:
            bullets.append(f"Suggested sign flips: {len(flip_assets)} (capped at {MAX_SIGN_FLIPS})")
        except Exception:
            pass
        try:
            if bool(USE_MEAN_CVAR_OPTIMIZER) and isinstance(opt_meta, dict) and opt_meta:
                er = float(opt_meta.get('exp_ret_daily', 0.0))
                cv = float(opt_meta.get('cvar_loss', np.nan))
                ra = float(opt_meta.get('risk_aversion', np.nan))
                bullets.append(f"Sizing method: Mean–CVaR optimizer (SciPy linprog). exp_ret≈{er * 252.0:.1%}/yr; CVaR_loss≈{cv:.4f} (daily); risk_aversion={ra:.2f}.")
            else:
                bullets.append("Sizing method: rule-based tilts (optimizer disabled/unavailable).")
        except Exception:
            pass
        try:
            top = simple_tbl.head(5)
            moves = []
            for _i, row in top.iterrows():
                moves.append(f"{row['Ticker']}: {row['Current Weight (%)']:+.2f}% → {row['Recommended Weight (%)']:+.2f}%")
            if moves:
                bullets.append('Largest recommended changes: ' + '; '.join(moves))
        except Exception:
            pass

        explain('13) Rebalancing recommendations (how to read)', bullets)
    except Exception:
        pass


# ---------- Hedging overlay scaffold (protective puts / collars) ----------
# Disabled by default. Best-effort: uses yfinance options chains when available.
ENABLE_HEDGING_OVERLAY = bool(globals().get('ENABLE_HEDGING_OVERLAY', False))
HEDGE_DTE_TARGET = int(globals().get('HEDGE_DTE_TARGET', 90) or 90)
HEDGE_PUT_OTM = float(globals().get('HEDGE_PUT_OTM', 0.90) or 0.90)   # strike = spot * pct (e.g. 0.90 => 10% OTM put)
HEDGE_CALL_OTM = float(globals().get('HEDGE_CALL_OTM', 1.05) or 1.05) # strike = spot * pct (e.g. 1.05 => 5% OTM call)

# Default mapping from bubble domain -> hedge-underlying proxy
HEDGE_UNDERLYING_BY_DOMAIN = globals().get(
    'HEDGE_UNDERLYING_BY_DOMAIN',
    {
        'AI bubble': 'QQQ',
        'Private Credit bubble': 'HYG',
        'Crypto bubble': 'BITO',
        '__default__': 'SPY',
    },
)


def _pick_expiry(options: list[str], target_days: int) -> str | None:
    if not options:
        return None
    today = pd.Timestamp.today().normalize()
    want = today + pd.Timedelta(days=int(target_days))
    best = None
    best_d = None
    for o in options:
        try:
            dt = pd.to_datetime(o).normalize()
        except Exception:
            continue
        if dt < today:
            continue
        d = abs((dt - want).days)
        if best_d is None or d < best_d:
            best = o
            best_d = d
    return best


def _mid_price(row: pd.Series) -> float:
    try:
        bid = float(row.get('bid', np.nan))
    except Exception:
        bid = np.nan
    try:
        ask = float(row.get('ask', np.nan))
    except Exception:
        ask = np.nan
    try:
        last = float(row.get('lastPrice', np.nan))
    except Exception:
        last = np.nan
    if np.isfinite(bid) and np.isfinite(ask) and ask >= bid:
        return 0.5 * (bid + ask)
    if np.isfinite(last):
        return last
    return np.nan


def _spot_price(ticker: str) -> float:
    # Prefer already-downloaded prices when possible.
    try:
        if 'detect_prices' in globals() and isinstance(detect_prices, pd.DataFrame) and ticker in detect_prices.columns:
            s = pd.to_numeric(detect_prices[ticker], errors='coerce').dropna()
            if not s.empty:
                return float(s.iloc[-1])
    except Exception:
        pass
    try:
        import yfinance as yf
        h = yf.download(ticker, period='10d', auto_adjust=True, progress=False)
        if isinstance(h, pd.DataFrame) and (not h.empty) and ('Close' in h.columns):
            s = pd.to_numeric(h['Close'], errors='coerce').dropna()
            if not s.empty:
                return float(s.iloc[-1])
    except Exception:
        pass
    return np.nan


if ENABLE_HEDGING_OVERLAY:
    try:
        import yfinance as yf

        top_domain = None
        try:
            if 'severity_by_domain' in globals() and isinstance(severity_by_domain, dict) and severity_by_domain:
                top_domain = max(severity_by_domain, key=severity_by_domain.get)
        except Exception:
            top_domain = None

        underlying = None
        try:
            if top_domain is not None:
                underlying = str(HEDGE_UNDERLYING_BY_DOMAIN.get(str(top_domain), '') or '').strip()
        except Exception:
            underlying = None
        if not underlying:
            underlying = str(HEDGE_UNDERLYING_BY_DOMAIN.get('__default__', 'SPY') or 'SPY').strip()

        spot = _spot_price(underlying)
        if not np.isfinite(spot) or spot <= 0:
            raise RuntimeError(f'[hedge] Could not resolve spot price for {underlying}.')

        tkr = yf.Ticker(underlying)
        opts = list(getattr(tkr, 'options', []) or [])
        exp = _pick_expiry(opts, target_days=int(HEDGE_DTE_TARGET))
        if not exp:
            raise RuntimeError(f'[hedge] No options expiries available for {underlying}.')

        chain = tkr.option_chain(exp)
        puts = getattr(chain, 'puts', pd.DataFrame()).copy()
        calls = getattr(chain, 'calls', pd.DataFrame()).copy()
        if puts.empty or calls.empty:
            raise RuntimeError(f'[hedge] Empty options chain for {underlying} @ {exp}.')

        # Choose strikes by moneyness targets
        put_target = float(spot) * float(HEDGE_PUT_OTM)
        call_target = float(spot) * float(HEDGE_CALL_OTM)

        put_row = None
        try:
            sub = puts[pd.to_numeric(puts.get('strike'), errors='coerce') <= float(put_target)].copy()
            sub['strike'] = pd.to_numeric(sub.get('strike'), errors='coerce')
            sub = sub.dropna(subset=['strike']).sort_values('strike')
            if not sub.empty:
                put_row = sub.iloc[-1]
        except Exception:
            put_row = None

        call_row = None
        try:
            sub = calls[pd.to_numeric(calls.get('strike'), errors='coerce') >= float(call_target)].copy()
            sub['strike'] = pd.to_numeric(sub.get('strike'), errors='coerce')
            sub = sub.dropna(subset=['strike']).sort_values('strike')
            if not sub.empty:
                call_row = sub.iloc[0]
        except Exception:
            call_row = None

        put_mid = _mid_price(put_row) if put_row is not None else np.nan
        call_mid = _mid_price(call_row) if call_row is not None else np.nan

        put_strike = float(put_row.get('strike')) if put_row is not None and np.isfinite(put_row.get('strike', np.nan)) else np.nan
        call_strike = float(call_row.get('strike')) if call_row is not None and np.isfinite(call_row.get('strike', np.nan)) else np.nan

        put_cost_pct = (float(put_mid) / float(spot)) if np.isfinite(put_mid) else np.nan
        collar_net_pct = ((float(put_mid) - float(call_mid)) / float(spot)) if (np.isfinite(put_mid) and np.isfinite(call_mid)) else np.nan

        hedge_tbl = pd.DataFrame(
            [
                {
                    'Primary driver': str(top_domain or ''),
                    'Underlying': underlying,
                    'Spot': float(spot),
                    'Expiry': str(exp),
                    'Put strike': put_strike,
                    'Put mid': put_mid,
                    'Put cost (% notional)': 100.0 * float(put_cost_pct) if np.isfinite(put_cost_pct) else np.nan,
                    'Call strike (collar)': call_strike,
                    'Call mid (collar)': call_mid,
                    'Collar net cost (% notional)': 100.0 * float(collar_net_pct) if np.isfinite(collar_net_pct) else np.nan,
                }
            ]
        )

        print('[hedge] Overlay scaffold (best-effort; informational only):')
        display(
            hedge_tbl.style.format(
                {
                    'Spot': '{:,.2f}',
                    'Put strike': '{:,.2f}',
                    'Put mid': '{:,.2f}',
                    'Put cost (% notional)': '{:.2f}%',
                    'Call strike (collar)': '{:,.2f}',
                    'Call mid (collar)': '{:,.2f}',
                    'Collar net cost (% notional)': '{:.2f}%',
                },
                na_rep='—',
            )
        )
        print(
            '[hedge] Notes: cost is per-underlying notional (option_mid / spot). This does not size contracts for your portfolio. '
            'Treat as a parameter suggestion (expiry + moneyness) and confirm pricing/greeks with your execution stack.'
        )
    except Exception as e:
        print(f'[hedge] Warning: could not build hedging overlay: {e}')

if SAVE_REBALANCE_OUTPUTS:
    tbl_out = simple_tbl[['Ticker', 'Ticker Asset Name', 'Current Weight (%)', 'Recommended Weight (%)', 'Long/Short Change']].copy()
    tbl_out.to_csv(OUTPUT_LATEST_CSV, index=False)

    # Audit trail (JSONL): one entry per run (best-effort)
    try:
        if bool(globals().get('ENABLE_REBALANCE_AUDIT_LOG', True)):
            import json
            audit_path = (os.getenv('REBALANCE_AUDIT_LOG_PATH', '') or '').strip() or '.bubble_rebalance_audit.jsonl'
            run_at = pd.Timestamp.now('UTC').isoformat()

            asof = None
            try:
                asof = str(alerts['asof'].max()) if ('alerts' in globals()) and isinstance(alerts, pd.DataFrame) and ('asof' in alerts.columns) else None
            except Exception:
                asof = None

            def _num_or_none(v):
                try:
                    x = float(v)
                    return float(x) if np.isfinite(x) else None
                except Exception:
                    return None

            payload = {
                'run_at_utc': run_at,
                'asof': asof,
                'turnover': _num_or_none(turnover),
                'max_turnover': (_num_or_none(MAX_TURNOVER) if MAX_TURNOVER is not None else None),
                'use_optimizer': bool(globals().get('USE_MEAN_CVAR_OPTIMIZER', False)),
                'optimizer_meta': (opt_meta if ('opt_meta' in globals()) and isinstance(opt_meta, dict) else {}),
                'beta_method': str(globals().get('beta_method_used', '') or ''),
                'beta_rls_lambda': (_num_or_none(globals().get('beta_rls_lambda_used', np.nan)) if ('beta_rls_lambda_used' in globals()) else None),
                'weights_current': {str(k): _num_or_none(v) for k, v in base_w.items()},
                'weights_recommended': {str(k): _num_or_none(v) for k, v in w_reco.items()},
            }
            with open(audit_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
            print('[audit] Appended rebalance audit entry: ' + audit_path)
    except Exception as e:
        print(f'[audit] Warning: could not write rebalance audit log: {e}')
"""

    # NOTE: Build cells from this file's *raw* triple-quoted blocks, not the evaluated
    # Python string values. This avoids Python interpreting escape sequences (e.g. "\n")
    # inside code strings, which can otherwise corrupt notebook cell source.
    import re

    self_text = Path(__file__).read_text(encoding='utf-8')
    raw_blocks: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\s*=\s*"""(.*?)"""', self_text, re.S):
        raw_blocks[m.group(1)] = m.group(2)

    def _raw(name: str) -> str:
        if name not in raw_blocks:
            raise KeyError(f"Missing raw notebook block: {name}")
        return raw_blocks[name]

    nb['cells'] = [
        nbf.v4.new_markdown_cell(_raw('title_md')),
        nbf.v4.new_code_cell(_raw('setup_code')),
        nbf.v4.new_markdown_cell(_raw('market_md')),
        nbf.v4.new_code_cell(_raw('market_code')),
        nbf.v4.new_markdown_cell(_raw('config_md')),
        nbf.v4.new_code_cell(_raw('config_code')),
        nbf.v4.new_markdown_cell(_raw('data_md')),
        nbf.v4.new_code_cell(_raw('data_code')),
        nbf.v4.new_markdown_cell(_raw('returns_md')),
        nbf.v4.new_code_cell(_raw('returns_code')),
        nbf.v4.new_markdown_cell(_raw('corr_md')),
        nbf.v4.new_code_cell(_raw('corr_code')),
        nbf.v4.new_markdown_cell(_raw('portfolio_md')),
        nbf.v4.new_code_cell(_raw('portfolio_code')),
        nbf.v4.new_markdown_cell(_raw('frontier_md')),
        nbf.v4.new_code_cell(_raw('frontier_code')),
        nbf.v4.new_markdown_cell(_raw('backtest_md')),
        nbf.v4.new_code_cell(_raw('backtest_code')),
        nbf.v4.new_markdown_cell(_raw('mc_md')),
        nbf.v4.new_code_cell(_raw('mc_code')),
        nbf.v4.new_markdown_cell(_raw('forecast_md')),
        nbf.v4.new_code_cell(_raw('forecast_code')),
        nbf.v4.new_markdown_cell(_raw('stress_md')),
        nbf.v4.new_code_cell(_raw('stress_code')),
        nbf.v4.new_markdown_cell(_raw('scenario_md')),
        nbf.v4.new_code_cell(_raw('scenario_code')),
        nbf.v4.new_markdown_cell(_raw('summary_md')),
        nbf.v4.new_markdown_cell(_raw('ews_md')),
        nbf.v4.new_code_cell(_raw('ews_code')),
        nbf.v4.new_markdown_cell(_raw('rebalance_md')),
        nbf.v4.new_code_cell(_raw('rebalance_code')),
    ]

    nb['metadata'] = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'name': 'python',
            'version': '3',
        },
    }

    out_path = Path('Bubble_Resilient_Portfolio.ipynb')
    out_path.write_text(nbf.writes(nb), encoding='utf-8')
    print(f'Wrote {out_path.resolve()}')


if __name__ == '__main__':
    main()
