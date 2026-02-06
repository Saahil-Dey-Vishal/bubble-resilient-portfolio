from __future__ import annotations

import importlib
import math
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Reduce yfinance/pandas deprecation noise (doesn't affect correctness).
_pandas4warning = getattr(pd.errors, "Pandas4Warning", None)
if _pandas4warning is not None:
    warnings.filterwarnings("ignore", category=_pandas4warning)


# -----------------------------
# Config (defaults; override via env vars)
# -----------------------------
DEFAULT_DOMAIN_TICKERS: dict[str, list[str]] = {
    "AI bubble": ["QQQ", "ROBT", "U8S1.SG", "ESIFF", "WTAI", "NMX101010.FGI"],
    "Private Credit bubble": ["HYG", "HYGU.L", "HYIN", "TAKMX", "VPC"],
    "Crypto bubble": ["BTC-USD", "^SPCMCFUE"],
}

DEFAULT_MACRO_TICKERS: list[str] = [
    "^VIX",
    "^VIX9D",
    "^VIX3M",
    "^VIX6M",
    "^VVIX",
    "^CPC",
    "^IRX",
    "^FVX",
    "^TNX",
    "^TYX",
    "SPY",
    "IWM",
    "QQQ",
    "ARKK",
    "SOXX",
    "XLK",
    "XLY",
    "XLP",
    "XLU",
    "HYG",
    "LQD",
    "TIP",
    "IEF",
]

HIST_START = os.getenv("MC_HIST_START", "1999-01-01")
EVENT_DB_PATH = os.getenv("MC_BUBBLE_EVENTS_DB", "bubble_events_database.csv")

# Synthetic MC controls
N_SIM = int(os.getenv("MC_N_SIM", "250") or 250)
HORIZON_DAYS = int(os.getenv("MC_HORIZON_DAYS", str(252 * 3)) or (252 * 3))
BUBBLE_PROB = float(os.getenv("MC_BUBBLE_PROB", "0.50") or 0.50)
BUBBLE_BUILD_DAYS = int(os.getenv("MC_BUBBLE_BUILD_DAYS", str(252)) or 252)
CRASH_DAYS = int(os.getenv("MC_CRASH_DAYS", "21") or 21)
CRASH_MAG = float(os.getenv("MC_CRASH_MAG", "-0.40") or -0.40)  # total crash return over CRASH_DAYS
LPPL_TC_MAX_DAYS = int(os.getenv("MC_LPPL_TC_MAX_DAYS", "180") or 180)

RANDOM_SEED = int(os.getenv("MC_RANDOM_SEED", "42") or 42)


# -----------------------------
# Utilities: data download
# -----------------------------

def _to_close_df(raw: pd.DataFrame | None, tickers: list[str]) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        field = "Close" if ("Close" in raw.columns.get_level_values(0)) else ("Adj Close" if ("Adj Close" in raw.columns.get_level_values(0)) else None)
        if field is None:
            return pd.DataFrame()
        close = raw[field].copy()
    else:
        field = "Close" if ("Close" in raw.columns) else ("Adj Close" if ("Adj Close" in raw.columns) else None)
        if field is None:
            return pd.DataFrame()
        if len(tickers) == 1:
            close = raw[[field]].rename(columns={field: tickers[0]}).copy()
        else:
            return pd.DataFrame()

    if isinstance(close, pd.Series):
        name = tickers[0] if len(tickers) == 1 else "close"
        close = close.to_frame(name=name)

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    for t in tickers:
        if t not in close.columns:
            close[t] = np.nan

    return close[tickers]


def download_close_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    tickers = list(dict.fromkeys([t for t in tickers if isinstance(t, str) and t.strip()]))
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    close = _to_close_df(raw, tickers)
    if close.empty:
        return pd.DataFrame()

    close = close.resample("B").last().ffill()
    return close


def download_volume(tickers: list[str], start: str) -> pd.DataFrame:
    tickers = list(dict.fromkeys([t for t in tickers if isinstance(t, str) and t.strip()]))
    if not tickers:
        return pd.DataFrame()

    raw: pd.DataFrame | None = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=tickers)

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        if "Volume" in lvl0:
            vol = raw["Volume"]
            if isinstance(vol, pd.Series):
                vol = vol.to_frame(name=tickers[0] if len(tickers) == 1 else "Volume")
            else:
                vol = vol.copy()
        else:
            vol = pd.DataFrame(index=pd.to_datetime(raw.index))
    else:
        if len(tickers) == 1 and ("Volume" in raw.columns):
            vol = raw[["Volume"]].rename(columns={"Volume": tickers[0]}).copy()
        else:
            vol = pd.DataFrame(index=pd.to_datetime(raw.index))

    vol.index = pd.to_datetime(vol.index)
    vol = vol.sort_index().resample("B").last()

    for t in tickers:
        if t not in vol.columns:
            vol[t] = np.nan

    return vol[tickers].fillna(0.0)


# -----------------------------
# Indicators (subset aligned to Section 12)
# -----------------------------


def zscore(s: pd.Series, window: int = 252 * 5) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    if window and window > 3:
        minp = max(30, int(window * 0.25))
        mu = s.rolling(window, min_periods=minp).mean()
        sd = s.rolling(window, min_periods=minp).std()
        return ((s - mu) / sd).replace([np.inf, -np.inf], np.nan)

    mu = float(s.mean())
    sd = float(s.std())
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd


def hurst_dfa_from_price(price: pd.Series) -> float:
    x = pd.to_numeric(price, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 252:
        return float("nan")

    lx = np.log(np.clip(x.to_numpy(dtype=float), 1e-12, None))
    r = np.diff(lx)
    r = r[np.isfinite(r)]
    n = int(len(r))
    if n < 240:
        return float("nan")

    r = r - float(np.mean(r))
    y = np.cumsum(r)

    s_min = 10
    s_max = max(20, int(n / 4))
    if s_max <= s_min + 1:
        return float("nan")

    scales = np.unique(np.logspace(np.log10(s_min), np.log10(s_max), num=12).astype(int))
    scales = scales[(scales >= s_min) & (scales <= s_max)]

    Fs: list[float] = []
    Ss: list[int] = []
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
        return float("nan")

    F = np.asarray(Fs, dtype=float)
    S = np.asarray(Ss, dtype=float)
    mask = np.isfinite(F) & (F > 0) & np.isfinite(S) & (S > 0)
    if mask.sum() < 4:
        return float("nan")

    alpha = float(np.polyfit(np.log(S[mask]), np.log(F[mask]), deg=1)[0])
    return float(np.clip(alpha, 0.0, 1.0))


def garch_sigma_fast(r: pd.Series, alpha: float = 0.05, beta: float = 0.94) -> pd.Series:
    r = pd.to_numeric(r, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
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


def compute_indicators(price: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    price = pd.to_numeric(price, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if price.empty:
        return pd.DataFrame()

    r = price.pct_change()

    ma_long = price.rolling(252).mean()
    sd_long = price.rolling(252).std().replace(0.0, np.nan)
    z_price = (price - ma_long) / sd_long

    mom_1m = price.pct_change(21)
    mom_3m = price.pct_change(63)
    accel_1m = mom_1m - mom_3m

    vol_1m = r.rolling(21).std()
    vol_6m = r.rolling(126).std()
    vol_ratio = vol_1m / vol_6m

    sig = garch_sigma_fast(r)
    sig_1m = sig.rolling(21).mean()
    sig_6m = sig.rolling(126).mean()
    garch_vol_ratio = sig_1m / sig_6m

    z = (r.abs() / (sig + 1e-12)).replace([np.inf, -np.inf], np.nan)
    jump_flag = (z > 3.0).astype(float)
    jump_rate_3m = jump_flag.rolling(63).mean()
    jump_mag_3m = z.where(jump_flag > 0).rolling(63).mean()

    dd_1y = price / price.rolling(252).max() - 1.0

    # Scalar DFA Hurst on the latest window (fast approximation)
    hurst = pd.Series(index=price.index, dtype=float)
    try:
        h = hurst_dfa_from_price(price)
        hurst.loc[price.index[-1]] = h
        hurst = hurst.ffill()
    except Exception:
        hurst = pd.Series(index=price.index, dtype=float)

    out = pd.DataFrame(
        {
            "z_price": z_price,
            "mom_1m": mom_1m,
            "mom_3m": mom_3m,
            "accel_1m": accel_1m,
            "vol_ratio": vol_ratio,
            "garch_vol_ratio": garch_vol_ratio,
            "jump_rate_3m": jump_rate_3m,
            "jump_mag_3m": jump_mag_3m,
            "drawdown_1y": dd_1y,
            "hurst": hurst,
        }
    )

    if volume is not None:
        v = pd.to_numeric(volume, errors="coerce").reindex(out.index).fillna(0.0).astype(float)
        out["z_volume"] = zscore(v)

    return out


def composite_score(ind: pd.DataFrame) -> pd.Series:
    if ind is None or ind.empty:
        return pd.Series(dtype=float)

    z = pd.DataFrame(index=ind.index)
    for col in [
        "z_price",
        "mom_1m",
        "mom_3m",
        "accel_1m",
        "vol_ratio",
        "garch_vol_ratio",
        "jump_rate_3m",
        "jump_mag_3m",
        "hurst",
        "z_volume",
    ]:
        if col in ind.columns:
            z[col] = zscore(ind[col]).reindex(ind.index)

    w = {
        "z_price": 0.22,
        "mom_1m": 0.10,
        "mom_3m": 0.18,
        "accel_1m": 0.10,
        "vol_ratio": 0.10,
        "garch_vol_ratio": 0.05,
        "jump_rate_3m": 0.03,
        "jump_mag_3m": 0.02,
        "hurst": 0.06,
        "z_volume": 0.05,
    }

    score = pd.Series(0.0, index=z.index)
    for k, wk in w.items():
        if k in z.columns:
            score = score.add(float(wk) * z[k].fillna(0.0), fill_value=0.0)

    return score.rename("score")


def logistic_probability(score: pd.Series) -> pd.Series:
    y = pd.to_numeric(score, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    if len(y.dropna()) < 60:
        return pd.Series(index=score.index, dtype=float)

    w = 252 * 5
    minp = int(min(w, max(252, int(w * 0.6))))

    mu = y.rolling(w, min_periods=minp).mean()
    sd = y.rolling(w, min_periods=minp).std().replace(0.0, np.nan)
    z = (y - mu) / sd

    mu2 = y.expanding(min_periods=60).mean()
    sd2 = y.expanding(min_periods=60).std().replace(0.0, np.nan)
    z2 = (y - mu2) / sd2
    z = z.fillna(z2)

    z0 = 1.0
    k = 1.2
    p = 100.0 / (1.0 + np.exp(-float(k) * (z - float(z0))))
    return pd.Series(p, index=score.index).rename("p")


# -----------------------------
# Domain index construction
# -----------------------------


def domain_index(prices: pd.DataFrame, tickers: list[str]) -> pd.Series:
    cols = [t for t in tickers if t in prices.columns]
    if not cols:
        return pd.Series(dtype=float)

    px = prices[cols].astype(float).replace(0.0, np.nan).ffill()
    if px.dropna(how="all").empty:
        return pd.Series(dtype=float)

    base = px.apply(lambda s: float(s.dropna().iloc[0]) if len(s.dropna()) else np.nan)
    idx = (px / base).mean(axis=1, skipna=True).rename("idx")
    return idx.dropna()


# -----------------------------
# Real-event validation (bubble_events_database.csv)
# -----------------------------


def load_event_db(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, skipinitialspace=True)
    except Exception:
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]

    for c in ["event_id", "event_name", "asset_class", "region", "domains", "tags", "notes"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    for c in ["start_date", "peak_date", "end_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df


def label_events(index: pd.Index, events: pd.DataFrame, domain: str, lead_days: int = 90) -> pd.Series:
    idx = pd.to_datetime(index, errors="coerce")
    idx = pd.DatetimeIndex(idx)
    y = pd.Series(0, index=idx, dtype=int)
    if events is None or events.empty:
        return y

    dom = str(domain).strip().lower()
    if "domains" not in events.columns:
        return y

    sub = events[events["domains"].astype(str).str.lower().str.contains(dom, regex=False)]
    for _i, r in sub.iterrows():
        start_dt = r.get("start_date", pd.NaT)
        end_dt = r.get("end_date", pd.NaT)
        if pd.isna(start_dt):
            start_dt = r.get("peak_date", pd.NaT)
        if pd.isna(end_dt):
            end_dt = r.get("peak_date", pd.NaT)
        if pd.isna(start_dt) or pd.isna(end_dt):
            continue

        lead_start = pd.to_datetime(start_dt) - pd.Timedelta(days=int(lead_days))
        y.loc[(y.index >= lead_start) & (y.index <= pd.to_datetime(end_dt))] = 1

    return y


def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    mask = np.isfinite(fpr) & np.isfinite(tpr)
    if mask.sum() < 2:
        return float("nan")
    fpr = fpr[mask]
    tpr = tpr[mask]
    order = np.argsort(fpr)
    x = fpr[order]
    y = tpr[order]
    if len(x) < 2:
        return float("nan")
    dx = np.diff(x)
    return float(np.sum(dx * (y[1:] + y[:-1]) / 2.0))


def confusion_metrics(alert: pd.Series, y: pd.Series) -> dict[str, float]:
    common = alert.index.intersection(y.index)
    ap = alert.reindex(common).fillna(0).astype(int)
    yp = y.reindex(common).fillna(0).astype(int)

    tp = int(((ap == 1) & (yp == 1)).sum())
    fp = int(((ap == 1) & (yp == 0)).sum())
    fn = int(((ap == 0) & (yp == 1)).sum())
    tn = int(((ap == 0) & (yp == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "fpr": float(fpr), "tpr": float(recall)}


def validate_on_real_events(prob: pd.Series, events: pd.DataFrame, domain: str) -> dict[str, float]:
    p = pd.to_numeric(prob, errors="coerce").astype(float).dropna()
    if p.empty:
        return {"auc": float("nan"), "best_f1": float("nan"), "best_thr": float("nan")}

    y = label_events(p.index, events, domain, lead_days=90)
    if int(y.sum()) == 0:
        return {"auc": float("nan"), "best_f1": float("nan"), "best_thr": float("nan")}

    thr_grid = np.linspace(0.0, 100.0, 51)
    best = {"thr": float("nan"), "f1": -float("inf")}
    fprs = []
    tprs = []

    for thr in thr_grid:
        alert = (p >= float(thr)).astype(int)
        m = confusion_metrics(alert, y)
        fprs.append(m["fpr"])
        tprs.append(m["tpr"])
        if np.isfinite(m["f1"]) and float(m["f1"]) > float(best["f1"]):
            best = {"thr": float(thr), "f1": float(m["f1"])}

    auc = auc_from_roc(np.array(fprs, dtype=float), np.array(tprs, dtype=float))
    return {"auc": float(auc), "best_f1": float(best["f1"]), "best_thr": float(best["thr"]), "events_labeled_days": float(y.sum())}


def df_to_markdown(df: pd.DataFrame, float_precision: int = 3) -> str:
    """Minimal DataFrame->Markdown renderer without optional `tabulate` dependency."""

    if df is None or df.empty:
        return ""

    show = df.reset_index()
    cols = [str(c) for c in show.columns]

    prec = int(float_precision)

    def fmt(v: object) -> str:
        if v is None:
            return ""
        try:
            if pd.isna(v):  # type: ignore[arg-type]
                return ""
        except Exception:
            pass

        if isinstance(v, (float, np.floating)):
            fv = float(v)
            if not np.isfinite(fv):
                return ""
            return f"{fv:.{prec}f}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = [
        "| " + " | ".join(fmt(row[c]) for c in show.columns) + " |"
        for _i, row in show.iterrows()
    ]
    return "\n".join([header, sep, *body])


# -----------------------------
# Synthetic bubble injection + Monte Carlo
# -----------------------------


def lppl_path(n: int, tc_ahead: int, m: float = 0.5, omega: float = 8.0, c: float = 0.05) -> np.ndarray:
    # Deterministic LPPL-like multiplier path (normalized to 1 at start)
    # This is a simplified generator (not a fitted model).
    t = np.arange(n, dtype=float)
    tc = float(n - 1 + max(5, int(tc_ahead)))
    dt = np.clip(tc - t, 1e-6, None)

    # Core super-exponential acceleration term
    core = (dt ** m)
    core = (core - core.min()) / (core.max() - core.min() + 1e-12)
    core = 1.0 + 0.35 * (1.0 - core)  # accelerate toward the end

    # Log-periodic wiggles
    wig = 1.0 + float(c) * np.cos(float(omega) * np.log(dt))

    mult = core * wig
    mult = mult / float(mult[0])
    return mult


def inject_bubble(price0: float, horizon: int, rng: np.random.Generator) -> tuple[pd.Series, dict[str, int]]:
    # Build a synthetic path with an LPPL-like run-up and a crash.
    idx = pd.date_range("2020-01-01", periods=int(horizon), freq="B")

    bubble = rng.random() < float(BUBBLE_PROB)

    # Baseline noise: IID normal (kept simple; EWS features include GARCH/jump proxies downstream)
    base_sigma = 0.012
    base_mu = 0.0002
    r = rng.normal(loc=base_mu, scale=base_sigma, size=int(horizon))

    crash_start = int(horizon) - int(CRASH_DAYS)

    if bubble:
        build_start = max(0, crash_start - int(BUBBLE_BUILD_DAYS))
        mult = lppl_path(int(crash_start - build_start), tc_ahead=min(int(LPPL_TC_MAX_DAYS), 120), m=0.6, omega=8.0, c=0.04)

        # Convert multiplier into incremental returns that overlay the baseline returns
        overlay = np.zeros(int(horizon), dtype=float)
        seg = mult
        seg_r = np.diff(np.log(seg + 1e-12), prepend=np.log(seg[0] + 1e-12))
        overlay[build_start:crash_start] = seg_r
        r = r + overlay

        # Crash: distribute CRASH_MAG over CRASH_DAYS using a smooth ramp
        crash_total = float(CRASH_MAG)
        ramp = np.linspace(0.3, 1.0, int(CRASH_DAYS), dtype=float)
        ramp = ramp / float(ramp.sum())
        r[crash_start:] = r[crash_start:] + crash_total * ramp

    px = float(price0) * np.exp(np.cumsum(r))
    meta = {
        "bubble": int(bubble),
        "crash_start": int(crash_start),
        "crash_days": int(CRASH_DAYS),
    }

    return pd.Series(px, index=idx), meta


def evaluate_detection(prob: pd.Series, meta: dict[str, int], thr: float = 60.0) -> dict[str, float]:
    # Did the model cross threshold BEFORE crash_start?
    p = pd.to_numeric(prob, errors="coerce").astype(float)
    if p.dropna().empty:
        return {"detected": 0.0, "lead_days": float("nan")}

    crash_start = int(meta.get("crash_start", len(p) - 1))
    crash_dt = p.index[min(crash_start, len(p) - 1)]

    pre = p[p.index < crash_dt]
    hit = pre[pre >= float(thr)].index.min()
    if pd.isna(hit):
        return {"detected": 0.0, "lead_days": float("nan")}

    return {"detected": 1.0, "lead_days": float((crash_dt - hit).days)}


def run_mc(domain: str, price0: float, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _ in range(int(N_SIM)):
        px, meta = inject_bubble(price0, int(HORIZON_DAYS), rng)
        ind = compute_indicators(px)
        score = composite_score(ind)
        prob = logistic_probability(score)
        ev = evaluate_detection(prob, meta, thr=60.0)
        rows.append({"domain": domain, **meta, **ev})
    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    report = root / "mc_validation_report.md"

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rng = np.random.default_rng(RANDOM_SEED)

    # Dependency probe (for transparency)
    deps = {}
    for name in ["yfinance", "pandas", "numpy", "scipy", "statsmodels"]:
        try:
            mod = importlib.import_module(name)
            deps[name] = getattr(mod, "__version__", "ok")
        except Exception as e:
            deps[name] = f"MISSING: {type(e).__name__}: {str(e)[:80]}"

    # Build ticker universe
    domain_tickers = DEFAULT_DOMAIN_TICKERS
    tickers = sorted({t for ts in domain_tickers.values() for t in ts} | set(DEFAULT_MACRO_TICKERS))

    prices = download_close_prices(tickers, start=HIST_START)

    # Domain probabilities on real history (logistic model on composite score of the domain index)
    event_db_path = Path(EVENT_DB_PATH)
    if not event_db_path.is_absolute():
        event_db_path = root / event_db_path
    events = load_event_db(str(event_db_path))

    real_rows = []
    domain_idx = {}
    for dom, tks in domain_tickers.items():
        idx = domain_index(prices, tks)
        domain_idx[dom] = idx

        ind = compute_indicators(idx)
        score = composite_score(ind)
        p = logistic_probability(score)

        m = validate_on_real_events(p, events, dom)
        real_rows.append({"domain": dom, **m, "prob_last": float(p.dropna().iloc[-1]) if not p.dropna().empty else float("nan")})

    real_tbl = pd.DataFrame(real_rows).set_index("domain").sort_index()

    # Monte Carlo synthetic injection
    mc_frames = []
    for dom, idx in domain_idx.items():
        if idx is None or idx.dropna().empty:
            continue
        px0 = float(idx.dropna().iloc[-1])
        mc_frames.append(run_mc(dom, px0, rng))

    mc_tbl = pd.concat(mc_frames, ignore_index=True) if mc_frames else pd.DataFrame()

    # Summaries
    mc_summary = pd.DataFrame()
    if not mc_tbl.empty:
        def _agg(g: pd.DataFrame) -> dict[str, float]:
            det = float(g["detected"].mean()) if ("detected" in g.columns) else float("nan")
            lt = pd.to_numeric(g["lead_days"], errors="coerce") if ("lead_days" in g.columns) else pd.Series(dtype=float)
            lt = lt.dropna()
            return {
                "detected_rate": det,
                "lead_days_mean": float(lt.mean()) if len(lt) else float("nan"),
                "lead_days_p50": float(lt.median()) if len(lt) else float("nan"),
                "lead_days_p10": float(lt.quantile(0.10)) if len(lt) else float("nan"),
                "lead_days_p90": float(lt.quantile(0.90)) if len(lt) else float("nan"),
                "n_sims": float(len(g)),
                "bubble_share": float(pd.to_numeric(g["bubble"], errors="coerce").mean()) if ("bubble" in g.columns) else float("nan"),
            }

        mc_summary = mc_tbl.groupby("domain").apply(lambda g: pd.Series(_agg(g))).sort_index()

    # Write report
    lines = []
    lines.append("# Monte Carlo + Event DB Validation Report\n")
    lines.append(f"- Run at: **{now_utc}**")
    lines.append(f"- Python: `{sys.version.split()[0]}`")
    lines.append(f"- Platform: `{platform.platform()}`")
    lines.append("")

    lines.append("## Dependency probe")
    for k in sorted(deps.keys()):
        lines.append(f"- {k}: {deps[k]}")
    lines.append("")

    lines.append("## Inputs")
    lines.append(f"- Event DB: `{event_db_path}`")
    lines.append(f"- Domains: {list(domain_tickers.keys())}")
    lines.append(f"- Total tickers downloaded (domains + macro): {len(tickers)}")
    lines.append("")

    lines.append("## Real-history validation (event DB)")
    if real_tbl.empty:
        lines.append("- No real-history validation results (insufficient data).")
    else:
        lines.append(df_to_markdown(real_tbl))
    lines.append("")

    lines.append("## Synthetic bubble injection + Monte Carlo")
    lines.append("Simulation settings:")
    lines.append(f"- N_SIM={int(N_SIM)} per domain")
    lines.append(f"- HORIZON_DAYS={int(HORIZON_DAYS)}")
    lines.append(f"- BUBBLE_PROB={float(BUBBLE_PROB):.2f}")
    lines.append(f"- BUBBLE_BUILD_DAYS={int(BUBBLE_BUILD_DAYS)}")
    lines.append(f"- CRASH_DAYS={int(CRASH_DAYS)}")
    lines.append(f"- CRASH_MAG={float(CRASH_MAG):.2f}")
    lines.append("")

    if mc_summary.empty:
        lines.append("- No Monte Carlo results.")
    else:
        lines.append(df_to_markdown(mc_summary))

    report.write_text("\n".join(lines), encoding="utf-8")

    print("STATUS=OK")
    print("REPORT=" + str(report.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
