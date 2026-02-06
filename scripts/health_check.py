from __future__ import annotations

import importlib
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf
import pandas as pd
import yfinance as yf


def main() -> int:
    root = Path('.')
    report = root / 'health_check_report.md'

    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    required_files = [
        'Bubble_Resilient_Portfolio.ipynb',
        'Bubble_resilient_portfolio_notebook.py',
        'FICC_and_Alternatives_50_Ticker_Roster_Updated_Long_Short_Positions_2026_to_2030.csv',
        'bubble_events_database.csv',
        'requirements.txt',
    ]
    missing_files = [p for p in required_files if not (root / p).exists()]

    imports = [
        'yfinance',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
        'statsmodels',
        'requests',
        'nbformat',
    ]
    import_status: dict[str, str] = {}
    for pkg in imports:
        try:
            importlib.import_module(pkg)
            import_status[pkg] = 'ok'
        except Exception as e:
            import_status[pkg] = f'FAIL: {type(e).__name__}: {str(e)[:120]}'

    # Roster CSV sanity
    roster_info: dict[str, object] = {}
    roster_path = root / 'FICC_and_Alternatives_50_Ticker_Roster_Updated_Long_Short_Positions_2026_to_2030.csv'
    try:
        roster_df = pd.read_csv(roster_path)
        roster_info['rows'] = int(len(roster_df))
        roster_info['cols'] = [str(c) for c in roster_df.columns.tolist()]
    except Exception as e:
        roster_info['error'] = f'{type(e).__name__}: {str(e)[:200]}'

    # Bubble events DB sanity (expanded labeling for Section 12 validation)
    bubble_events_info: dict[str, object] = {}
    bubble_events_path = root / 'bubble_events_database.csv'
    try:
        ev = pd.read_csv(bubble_events_path)
        bubble_events_info['rows'] = int(len(ev))
        bubble_events_info['cols'] = [str(c) for c in ev.columns.tolist()]
    except Exception as e:
        bubble_events_info['error'] = f'{type(e).__name__}: {str(e)[:200]}'

    # Live Yahoo Finance sanity (bubble tickers)
    bubbles = {
        'AI bubble': ['QQQ', 'ROBT', 'U8S1.SG', 'ESIFF', 'WTAI', 'NMX101010.FGI'],
        'Private Credit bubble': ['HYG', 'HYGU.L', 'HYIN', 'TAKMX', 'VPC'],
        'Crypto bubble': ['BTC-USD', '^SPCMCFUE'],
    }
    all_tickers = sorted({t for ts in bubbles.values() for t in ts})

    try:
        raw = yf.download(
            all_tickers,
            period='14d',
            interval='1d',
            auto_adjust=True,
            progress=False,
            group_by='column',
            threads=True,
        )
    except Exception as e:
        raw = None
        yf_error = f'{type(e).__name__}: {str(e)[:200]}'
    else:
        yf_error = ''

    def extract_last_close(ticker: str) -> dict[str, object]:
        out: dict[str, object] = {
            'ticker': ticker,
            'ok': False,
            'last_date': None,
            'last_close': None,
            'error': '',
        }
        try:
            if raw is None or getattr(raw, 'empty', True):
                out['error'] = yf_error or 'empty_download'
                return out

            if isinstance(raw.columns, pd.MultiIndex):
                field = (
                    'Close'
                    if 'Close' in raw.columns.get_level_values(0)
                    else ('Adj Close' if 'Adj Close' in raw.columns.get_level_values(0) else None)
                )
                if field is None:
                    out['error'] = 'no_close_field'
                    return out
                if (field, ticker) not in raw.columns:
                    out['error'] = 'ticker_missing'
                    return out
                s = raw[(field, ticker)].dropna()
            else:
                field = 'Close' if 'Close' in raw.columns else ('Adj Close' if 'Adj Close' in raw.columns else None)
                if field is None:
                    out['error'] = 'no_close_field'
                    return out
                s = raw[field].dropna()

            if s.empty:
                out['error'] = 'no_data'
                return out

            out['ok'] = True
            out['last_date'] = str(pd.to_datetime(s.index[-1]).date())
            out['last_close'] = float(s.iloc[-1])
            return out
        except Exception as e:
            out['error'] = f'{type(e).__name__}: {str(e)[:160]}'
            return out

    price_checks = [extract_last_close(t) for t in all_tickers]

    # Notebook compile check
    nb_compile: dict[str, object] = {}
    try:
        nb_path = root / 'Bubble_Resilient_Portfolio.ipynb'
        nb = nbf.read(nb_path, as_version=4)
        errs = []
        code_cells = 0
        for i, cell in enumerate(nb.cells):
            if cell.get('cell_type') != 'code':
                continue
            code_cells += 1
            src = cell.get('source') or ''
            try:
                compile(src, f'cell_{i}', 'exec')
            except Exception as e:
                errs.append((i, type(e).__name__, str(e)))
        nb_compile['code_cells'] = int(code_cells)
        nb_compile['errors'] = errs
    except Exception as e:
        nb_compile['error'] = f'{type(e).__name__}: {str(e)[:200]}'

    jupyter_path = shutil.which('jupyter')

    fails: list[str] = []
    warns: list[str] = []

    if missing_files:
        fails.append(f"Missing required files: {missing_files}")
    if any(v.startswith('FAIL') for v in import_status.values()):
        fails.append('One or more required imports failed')
    if roster_info.get('error'):
        fails.append('Roster CSV failed to load')
    if bubble_events_info.get('error'):
        fails.append('Bubble events database failed to load')
    else:
        rows_val = bubble_events_info.get('rows', 0)
        try:
            rows_i = int(rows_val) if isinstance(rows_val, (int, float, str)) else 0
        except Exception:
            rows_i = 0
        if rows_i < 15:
            fails.append('Bubble events database has < 15 rows (insufficient for expanded validation)')

    for t in ['QQQ', 'HYG', 'BTC-USD']:
        r = next((x for x in price_checks if x['ticker'] == t), None)
        if not r or not r.get('ok'):
            fails.append(f'yfinance core ticker failed: {t} ({(r or {}).get("error")})')

    no_data = [r['ticker'] for r in price_checks if not r.get('ok')]
    if no_data:
        warns.append(
            f'yfinance returned no data for {len(no_data)}/{len(price_checks)} tickers: {no_data[:8]}'
            + (' ...' if len(no_data) > 8 else '')
        )

    if nb_compile.get('error'):
        fails.append('Notebook read/compile check failed')
    else:
        errs = nb_compile.get('errors')
        if isinstance(errs, list) and errs:
            fails.append(f"Notebook compile errors in {len(errs)} cells")

    status = 'PASS' if not fails else 'FAIL'
    if status == 'PASS' and warns:
        status = 'WARN'

    lines: list[str] = []
    lines.append('# Project Health Check Report\n')
    lines.append(f'- Run at: **{now_utc}**')
    lines.append(f'- Status: **{status}**\n')

    lines.append('## Environment')
    lines.append(f'- Python: `{sys.version.split()[0]}`')
    lines.append(f'- Platform: `{platform.platform()}`')
    lines.append(f'- Jupyter available: `{bool(jupyter_path)}`' + (f' (`{jupyter_path}`)' if jupyter_path else ''))
    lines.append('')

    lines.append('## Bubble events database')
    if bubble_events_info.get('error'):
        lines.append(f"- ERROR: {bubble_events_info['error']}")
    else:
        cols_val = bubble_events_info.get('cols')
        cols = cols_val if isinstance(cols_val, list) else []
        lines.append(f"- Rows: {bubble_events_info.get('rows')}")
        lines.append(f"- Columns: {cols[:25]}" + (' ...' if len(cols) > 25 else ''))
    lines.append('')

    lines.append('## Required files')
    for p in required_files:
        lines.append(f"- {p}: {'ok' if (root / p).exists() else 'MISSING'}")
    lines.append('')

    lines.append('## Imports')
    for k in imports:
        lines.append(f'- {k}: {import_status.get(k)}')
    lines.append('')

    lines.append('## Roster CSV')
    if roster_info.get('error'):
        lines.append(f"- ERROR: {roster_info['error']}")
    else:
        cols_val = roster_info.get('cols')
        cols = cols_val if isinstance(cols_val, list) else []
        lines.append(f"- Rows: {roster_info.get('rows')}")
        lines.append(f"- Columns: {cols[:20]}" + (' ...' if len(cols) > 20 else ''))
    lines.append('')

    lines.append('## Live Yahoo Finance (yfinance) checks')
    if yf_error:
        lines.append(f'- yfinance download error: {yf_error}')
    for r in price_checks:
        flag = 'ok' if r.get('ok') else f"FAIL({r.get('error')})"
        lines.append(f"- {r['ticker']}: {flag}, last_date={r.get('last_date')}, last_close={r.get('last_close')}")
    lines.append('')

    lines.append('## Notebook code-cell compile check')
    if nb_compile.get('error'):
        lines.append(f"- ERROR: {nb_compile['error']}")
    else:
        lines.append(f"- Code cells: {nb_compile.get('code_cells')}")
        errs_val = nb_compile.get('errors')
        errs = errs_val if isinstance(errs_val, list) else []
        lines.append(f"- Compile errors: {len(errs)}")
        for i, ename, msg in errs[:10]:
            lines.append(f"  - cell[{i}]: {ename}: {msg}")

    lines.append('')
    lines.append('## Notes')
    lines.append('- No database layer detected in this repo (CSV + live Yahoo Finance pulls).')
    lines.append('- For maximum confidence before publishing, run the notebook end-to-end (Run All) in a fresh kernel.')

    report.write_text('\n'.join(lines), encoding='utf-8')

    print(f'STATUS={status}')
    print(f'REPORT={report.resolve()}')
    if fails:
        print('FAIL_REASONS:')
        for f in fails:
            print(' -', f)
    if warns:
        print('WARNINGS:')
        for w in warns:
            print(' -', w)

    return 0 if status != 'FAIL' else 1


if __name__ == '__main__':
    raise SystemExit(main())
