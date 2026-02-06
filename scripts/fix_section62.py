#!/usr/bin/env python3
"""
Rewrite Section 6.2 with corrected statistics:
1. Fix axis swap (X=Risk, Y=Return - standard convention)
2. Remove pie charts
3. Reframe around downside protection (max drawdown), not raw returns
4. Compare BOTH portfolios in Monte Carlo crash scenarios
"""

import json

NB_PATH = '/Users/saahildey/Bubble Resilient Portfolio/Bubble_Resilient_Portfolio.ipynb'

NEW_SECTION_62 = '''# ============================================================================
# SECTION 6.2: POST-REBALANCING ANALYSIS DASHBOARD
# ============================================================================
# This section produces 3 live charts showing the DEFENSIVE value of rebalancing:
#   1. Risk vs Return comparison (standard axes: X=Risk, Y=Return)
#   2. Drawdown Protection: How both portfolios handle historical crashes
#   3. Crash Survival Monte Carlo: Both portfolios under bubble crash scenarios
#
# KEY INSIGHT: A bubble-resilient portfolio INTENTIONALLY accepts lower expected
# return in exchange for crash protection. Lower return + lower risk = WORKING.
# ============================================================================

print('=' * 80)
print('SECTION 6.2: POST-REBALANCING ANALYSIS DASHBOARD')
print('=' * 80)
print()

# --------------------------------------------------------------------------
# SHARED SETUP: Use rebalanced weights (w_reco) from Section 6
# --------------------------------------------------------------------------

train_start_62 = (pd.Timestamp.today().normalize() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
train_62 = rets.loc[train_start_62:, asset_tickers].dropna(how='all').fillna(0.0)

mu_62 = train_62.mean() * 252
cov_62 = train_62.cov() * 252

def port_mu_sigma_62(weights):
    """Compute annualized expected return and volatility for a portfolio."""
    m = float(np.dot(weights, mu_62.reindex(asset_tickers).fillna(0.0).values))
    v = float(np.dot(weights, np.dot(cov_62.reindex(index=asset_tickers, columns=asset_tickers).fillna(0.0).values, weights)))
    return m, float(np.sqrt(max(v, 0.0)))

# Original portfolio (Section 4 weights from CSV)
original_w = base_w.reindex(asset_tickers).fillna(0.0).values
original_m, original_s = port_mu_sigma_62(original_w)

# Post-rebalancing portfolio (Section 6 recommended weights)
rebalanced_w = w_reco.reindex(asset_tickers).fillna(0.0).values
rebalanced_m, rebalanced_s = port_mu_sigma_62(rebalanced_w)

# ============================================================================
# CHART 1: RISK vs RETURN COMPARISON (Standard Axes: X=Risk, Y=Return)
# ============================================================================

print('--- 6.2.1) Risk vs Return Comparison (Standard Efficient Frontier) ---')
print()

rng_62 = np.random.default_rng(RANDOM_SEED + 62)
N_SAMPLES_62 = 4000
pts_62 = []
signs_62 = sign.reindex(asset_tickers).fillna(1.0).values

for _ in range(N_SAMPLES_62):
    raw = rng_62.dirichlet(np.ones(len(asset_tickers)))
    w = raw * signs_62
    w = w * (GROSS_EXPOSURE / np.sum(np.abs(w)))
    if np.max(np.abs(w)) > MAX_ABS_WEIGHT:
        continue
    m, s = port_mu_sigma_62(w)
    pts_62.append((s, m))  # (risk, return) for plotting

pts_62 = np.array(pts_62)

# Efficient frontier envelope (upper boundary)
order_62 = np.argsort(pts_62[:, 0])  # sort by risk (x-axis)
sorted_pts_62 = pts_62[order_62]
front_62 = []
best_m_62 = -np.inf
for s, m in sorted_pts_62:
    if m > best_m_62:
        front_62.append((s, m))
        best_m_62 = m
front_62 = np.array(front_62) if front_62 else np.array([]).reshape(0, 2)

# Plot: X = Risk (volatility), Y = Return (standard convention)
fig1, ax1 = plt.subplots(figsize=(12, 7))
ax1.scatter(pts_62[:, 0], pts_62[:, 1], s=10, alpha=0.20, color='gray', label='Sampled portfolios')

if len(front_62) > 1:
    ax1.plot(front_62[:, 0], front_62[:, 1], color='orange', linewidth=2.5, label='Efficient frontier')

# Original = Red star, Rebalanced = Green star
ax1.scatter([original_s], [original_m], s=250, marker='*', color='red', 
            label=f'Original: {original_m:.2%} return, {original_s:.2%} risk', zorder=10)
ax1.scatter([rebalanced_s], [rebalanced_m], s=250, marker='*', color='green', 
            label=f'Rebalanced: {rebalanced_m:.2%} return, {rebalanced_s:.2%} risk', zorder=10)

# Arrow showing the defensive repositioning
if abs(rebalanced_m - original_m) > 0.001 or abs(rebalanced_s - original_s) > 0.001:
    ax1.annotate('', xy=(rebalanced_s, rebalanced_m), xytext=(original_s, original_m),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2.5, ls='--'))
    # Add text explaining the move
    mid_s = (original_s + rebalanced_s) / 2
    mid_m = (original_m + rebalanced_m) / 2
    ax1.annotate('Defensive\\nRepositioning', xy=(mid_s, mid_m), fontsize=9, color='blue',
                 ha='left', va='bottom')

ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax1.set_xlabel('Risk (Annualized Volatility)', fontsize=12)
ax1.set_ylabel('Expected Return (Annualized)', fontsize=12)
ax1.set_title('Chart 1: Risk vs Return - Bubble-Aware Defensive Repositioning', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.show()

delta_return = rebalanced_m - original_m
delta_risk = rebalanced_s - original_s

print(f'Original Portfolio: Return={original_m:.2%}, Risk={original_s:.2%}')
print(f'Rebalanced Portfolio: Return={rebalanced_m:.2%}, Risk={rebalanced_s:.2%}')
print(f'Change: Return {delta_return:+.2%}, Risk {delta_risk:+.2%}')
if delta_risk < 0:
    print(f'✓ Risk REDUCED by {abs(delta_risk):.2%} - defensive positioning working')
if delta_return < 0:
    print(f'  (Lower return = insurance premium for crash protection)')
print()

# ============================================================================
# CHART 2: DRAWDOWN PROTECTION BACKTEST
# ============================================================================

print('=' * 80)
print('--- 6.2.2) Drawdown Protection: Max Drawdown During Historical Crashes ---')
print('=' * 80)
print()

CRASH_PERIODS = {
    'Dot-com (2000-02)': ('2000-03-01', '2002-10-31'),
    'GFC (2007-09)': ('2007-10-01', '2009-03-31'),
    'COVID (2020)': ('2020-02-01', '2020-04-30'),
}

full_rets = rets[asset_tickers].dropna(how='all').fillna(0.0)

def compute_cumulative(ret_df, weights, start, end):
    period_rets = ret_df.loc[start:end]
    if period_rets.empty:
        return pd.Series(dtype=float)
    port_daily = period_rets.values @ weights
    return (1 + pd.Series(port_daily, index=period_rets.index)).cumprod()

def max_drawdown(cum_series):
    """Compute maximum drawdown from a cumulative return series."""
    if cum_series.empty or len(cum_series) < 2:
        return np.nan
    peak = cum_series.expanding().max()
    dd = (cum_series - peak) / peak
    return float(dd.min())

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
backtest_stats = []

for idx, (crash_name, (start, end)) in enumerate(CRASH_PERIODS.items()):
    ax = axes2[idx]
    rebal_cum = compute_cumulative(full_rets, rebalanced_w, start, end)
    original_cum = compute_cumulative(full_rets, original_w, start, end)
    
    if rebal_cum.empty or original_cum.empty:
        ax.text(0.5, 0.5, f'No data for {crash_name}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(crash_name, fontweight='bold')
        continue
    
    # Normalize to start at 1
    rebal_cum = rebal_cum / rebal_cum.iloc[0]
    original_cum = original_cum / original_cum.iloc[0]
    
    # Plot both portfolios
    ax.plot(original_cum.index, original_cum.values, color='red', lw=2, ls='--', label='Original')
    ax.plot(rebal_cum.index, rebal_cum.values, color='green', lw=2, label='Rebalanced')
    
    ax.axhline(y=1.0, color='black', ls='-', lw=0.5, alpha=0.3)
    ax.set_title(crash_name, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of $1')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Compute max drawdowns
    dd_original = max_drawdown(original_cum)
    dd_rebal = max_drawdown(rebal_cum)
    
    backtest_stats.append({
        'Period': crash_name,
        'Original Max DD': f'{dd_original:.1%}' if np.isfinite(dd_original) else '—',
        'Rebalanced Max DD': f'{dd_rebal:.1%}' if np.isfinite(dd_rebal) else '—',
        'DD Reduction': f'{(dd_original - dd_rebal):.1%}' if (np.isfinite(dd_original) and np.isfinite(dd_rebal)) else '—',
    })

plt.suptitle('Chart 2: Drawdown Protection - Max Drawdown During Crashes', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

if backtest_stats:
    print('Max Drawdown Comparison (more negative = worse):')
    display(pd.DataFrame(backtest_stats))
print()

# ============================================================================
# CHART 3: CRASH SURVIVAL MONTE CARLO (Both Portfolios)
# ============================================================================

print('=' * 80)
print('--- 6.2.3) Crash Survival Monte Carlo: Original vs Rebalanced ---')
print('=' * 80)
print()

MC_HORIZON = 252 * 2
MC_N_PATHS = 500

CRASH_SCENARIOS = {
    'Moderate Crash (-30%)': {'mu_shock': -0.30, 'vol_mult': 2.0, 'duration': 120},
    'Severe Crash (-50%)': {'mu_shock': -0.50, 'vol_mult': 2.5, 'duration': 180},
    'Flash Crash (-35%)': {'mu_shock': -0.35, 'vol_mult': 3.0, 'duration': 45},
}

# Get daily params for both portfolios
orig_daily_mu = original_m / 252
orig_daily_vol = original_s / np.sqrt(252)
rebal_daily_mu = rebalanced_m / 252
rebal_daily_vol = rebalanced_s / np.sqrt(252)

rng_mc = np.random.default_rng(RANDOM_SEED + 63)

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
forecast_stats = []

for idx, (scenario, params) in enumerate(CRASH_SCENARIOS.items()):
    ax = axes3[idx]
    crash_mu = params['mu_shock'] / params['duration']
    crash_vol = params['vol_mult']
    crash_dur = params['duration']
    
    # Simulate both portfolios
    orig_paths = np.zeros((MC_N_PATHS, MC_HORIZON + 1))
    rebal_paths = np.zeros((MC_N_PATHS, MC_HORIZON + 1))
    orig_paths[:, 0] = 1.0
    rebal_paths[:, 0] = 1.0
    
    crash_starts = rng_mc.integers(0, MC_HORIZON // 4, size=MC_N_PATHS)
    
    for t in range(MC_HORIZON):
        z = rng_mc.standard_normal(MC_N_PATHS)  # Same random shock for fair comparison
        for p in range(MC_N_PATHS):
            in_crash = (t >= crash_starts[p]) and (t < crash_starts[p] + crash_dur)
            
            # Original portfolio
            mu_t_orig = orig_daily_mu + crash_mu if in_crash else orig_daily_mu
            vol_t_orig = orig_daily_vol * crash_vol if in_crash else orig_daily_vol
            ret_orig = mu_t_orig + vol_t_orig * z[p]
            orig_paths[p, t + 1] = orig_paths[p, t] * (1 + ret_orig)
            
            # Rebalanced portfolio
            mu_t_rebal = rebal_daily_mu + crash_mu if in_crash else rebal_daily_mu
            vol_t_rebal = rebal_daily_vol * crash_vol if in_crash else rebal_daily_vol
            ret_rebal = mu_t_rebal + vol_t_rebal * z[p]
            rebal_paths[p, t + 1] = rebal_paths[p, t] * (1 + ret_rebal)
    
    days = np.arange(MC_HORIZON + 1)
    
    # Plot median paths for both
    orig_median = np.median(orig_paths, axis=0)
    rebal_median = np.median(rebal_paths, axis=0)
    
    # Confidence bands for rebalanced
    p5_rebal, p95_rebal = np.percentile(rebal_paths, [5, 95], axis=0)
    ax.fill_between(days, p5_rebal, p95_rebal, color='green', alpha=0.15, label='Rebal 90% CI')
    
    ax.plot(days, orig_median, color='red', lw=2, ls='--', label='Original (median)')
    ax.plot(days, rebal_median, color='green', lw=2.5, label='Rebalanced (median)')
    ax.axhline(y=1.0, color='black', ls='--', lw=1, alpha=0.5)
    ax.set_title(f'{scenario}', fontweight='bold')
    ax.set_xlabel('Days')
    ax.set_ylabel('Portfolio Value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Stats
    orig_final = orig_paths[:, -1]
    rebal_final = rebal_paths[:, -1]
    forecast_stats.append({
        'Scenario': scenario,
        'Original Median': f'${np.median(orig_final):.2f}',
        'Rebalanced Median': f'${np.median(rebal_final):.2f}',
        'Orig Survival (>$0.70)': f'{np.mean(orig_final > 0.70) * 100:.0f}%',
        'Rebal Survival (>$0.70)': f'{np.mean(rebal_final > 0.70) * 100:.0f}%',
    })

plt.suptitle('Chart 3: Crash Survival Monte Carlo (2-Year Horizon)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

if forecast_stats:
    print('Crash Survival Comparison (Survival = ending above $0.70 from $1.00):')
    display(pd.DataFrame(forecast_stats))
print()

# ============================================================================
# SUMMARY
# ============================================================================

print('=' * 80)
print('SECTION 6.2 COMPLETE')
print('=' * 80)

# Determine if the rebalancing is providing value
risk_reduced = delta_risk < 0
return_reduced = delta_return < 0

if risk_reduced and return_reduced:
    verdict = 'DEFENSIVE REPOSITIONING ACTIVE: Lower return + Lower risk = Insurance premium paid'
    explanation = 'The portfolio is positioned to survive bubble crashes. Lower expected return is the cost of this protection.'
elif risk_reduced and not return_reduced:
    verdict = 'OPTIMAL REPOSITIONING: Same/higher return with LOWER risk'
    explanation = 'The rebalancing found a more efficient point on the frontier.'
elif not risk_reduced and return_reduced:
    verdict = 'REPOSITIONING REVIEW NEEDED: Check factor exposures'
    explanation = 'Unexpected outcome - may need parameter tuning.'
else:
    verdict = 'MINIMAL CHANGE: Portfolio already well-positioned'
    explanation = 'The rebalancing engine found no significant improvements needed.'

display(Markdown(f"""
### Section 6.2 Summary

**{verdict}**

{explanation}

| Metric | Original | Rebalanced | Change |
|--------|----------|------------|--------|
| Expected Return | {original_m:.2%} | {rebalanced_m:.2%} | {delta_return:+.2%} |
| Expected Risk | {original_s:.2%} | {rebalanced_s:.2%} | {delta_risk:+.2%} |

**Charts:** Risk/Return Frontier, Drawdown Protection, Crash Survival Monte Carlo

All charts update with live Yahoo Finance data.
"""))
'''

def main():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    
    # Find and replace cell 17 (Section 6.2)
    for i, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        if 'SECTION 6.2: POST-REBALANCING ANALYSIS DASHBOARD' in src[:500]:
            print(f"Found Section 6.2 at cell {i}, replacing...")
            
            # Convert new code to source list format
            new_lines = NEW_SECTION_62.split('\n')
            new_source = [line + '\n' for line in new_lines[:-1]]
            if new_lines:
                new_source.append(new_lines[-1])
            
            cell['source'] = new_source
            
            # Clear old outputs
            cell['outputs'] = []
            cell['execution_count'] = None
            
            break
    
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("✓ Section 6.2 rewritten successfully!")
    print("Changes:")
    print("  - Fixed axes: X=Risk, Y=Return (standard convention)")
    print("  - Removed pie charts")
    print("  - Changed backtest to show MAX DRAWDOWN (key metric for crash protection)")
    print("  - Monte Carlo now compares BOTH portfolios under crash scenarios")
    print("  - Added proper framing: lower return = insurance premium (EXPECTED)")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
