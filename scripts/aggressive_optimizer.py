#!/usr/bin/env python3
"""
AGGRESSIVE optimizer upgrade - designed to BEAT buy-and-hold
Key changes:
1. Raise severity threshold to 60% (was 40%)
2. Add return BOOST in low-risk periods (not just momentum tilt)
3. Only go defensive when probability > 70%
4. Reduce transaction costs by skipping small rebalances
"""
import json
import re

NB_PATH = '/Users/saahildey/Bubble Resilient Portfolio/Bubble_Resilient_Portfolio.ipynb'

with open(NB_PATH, 'r') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][19]['source'])

# ============================================================
# REPLACEMENT 1: _sev_from_prob - HIGHER threshold (60%)
# ============================================================
old1_pattern = r'def _sev_from_prob\(p: float\) -> float:.*?(?=\ndef |\nclass )'

new1 = '''def _sev_from_prob(p: float) -> float:
    # AGGRESSIVE: only trigger defensive mode when probability > 60%
    if p < 0.60:
        return 0.0  # LOW/MEDIUM risk = stay aggressive, capture upside
    elif p < 0.75:
        return float((p - 0.60) / 0.15)  # 0.60->0.75 maps to 0->1 (mild)
    else:
        return float(1.0 + (p - 0.75) / 0.25 * 1.5)  # 0.75->1.0 maps to 1->2.5

'''

match1 = re.search(old1_pattern, src, re.DOTALL)
if match1:
    src = src[:match1.start()] + new1 + src[match1.end():]
    print("✅ 1/3: Upgraded _sev_from_prob (threshold raised to 60%)")
else:
    print("❌ 1/3: Could not find _sev_from_prob")

# ============================================================
# REPLACEMENT 2: _bubble_adj_mu - BOOST returns in low risk
# ============================================================
old2_pattern = r'def _bubble_adj_mu\(train_rets, risk_sc, max_sv\):.*?(?=\ndef |\nclass )'

new2 = '''def _bubble_adj_mu(train_rets, risk_sc, max_sv):
    mu = train_rets.mean() * 252
    # AGGRESSIVE: boost returns in low-risk, only penalize in extreme risk
    if max_sv < 0.3:
        # VERY LOW RISK: strong momentum boost to capture upside
        mom_60d = train_rets.tail(60).mean() * 252
        mom_20d = train_rets.tail(20).mean() * 252
        # Boost winners more aggressively
        boost = ((mom_60d - mu) * 0.3 + (mom_20d - mu) * 0.2).clip(lower=-0.01, upper=0.05)
        return mu + boost
    elif max_sv < 1.0:
        # MEDIUM RISK: neutral (no penalty, no boost)
        return mu
    else:
        # HIGH RISK ONLY: apply defensive penalty
        penalty = risk_sc.fillna(0.0) * 0.015 * max_sv
        return mu - penalty

'''

match2 = re.search(old2_pattern, src, re.DOTALL)
if match2:
    src = src[:match2.start()] + new2 + src[match2.end():]
    print("✅ 2/3: Upgraded _bubble_adj_mu (BOOST in low risk, penalty only when severity >= 1.0)")
else:
    print("❌ 2/3: Could not find _bubble_adj_mu")

# ============================================================
# REPLACEMENT 3: _optimise_weights - MAXIMIZE returns, skip small trades
# ============================================================
old3_pattern = r'def _optimise_weights\(train_rets, base_w, risk_sc, max_sv, gross, maxw, signs\):.*?return pd\.Series\(w_out, index=cols\)'

new3 = '''def _optimise_weights(train_rets, base_w, risk_sc, max_sv, gross, maxw, signs):
    cols = list(base_w.index)
    R = train_rets.reindex(columns=cols).fillna(0.0)
    if R.shape[0] < 120:
        return base_w
    
    # AGGRESSIVE: Stay at base weights when risk is low (reduce turnover/costs)
    if max_sv < 0.8:
        # LOW-MEDIUM RISK: keep base weights, don't churn
        return base_w
    
    mu_adj = _bubble_adj_mu(R, risk_sc.reindex(cols).fillna(0.0), max_sv).values
    cov = R.cov().values * 252
    w0 = base_w.reindex(cols).fillna(0.0).values
    
    n = len(cols)
    bnds = []
    for i in range(n):
        if signs.get(cols[i], 1) >= 0:
            bnds.append((0.0, maxw))
        else:
            bnds.append((-maxw, 0.0))
    
    # Only reduce risk when severity is VERY HIGH (>= 1.5)
    orig_var = float(w0 @ cov @ w0)
    if max_sv >= 1.5:
        target_var = orig_var * 0.7  # reduce risk by 30%
    else:
        target_var = orig_var * 0.9  # mild reduction
    
    cons = [
        {'type': 'ineq', 'fun': lambda w: target_var - float(w @ cov @ w)},
        {'type': 'eq',   'fun': lambda w: float(np.abs(w).sum()) - gross},
    ]
    # MAXIMIZE adjusted return (minimize negative)
    res = minimize(lambda w: -float(w @ mu_adj), w0, method='SLSQP',
                   jac=lambda w: -mu_adj, bounds=bnds, constraints=cons,
                   options={'maxiter': 800, 'ftol': 1e-10})
    w_out = res.x if res.success else w0
    
    # Skip rebalance if change is tiny (saves transaction costs)
    delta = np.abs(w_out - w0).sum()
    if delta < 0.05:  # less than 5% total change = not worth the cost
        return base_w
    
    g = float(np.abs(w_out).sum())
    if g > 0:
        w_out *= gross / g
    return pd.Series(w_out, index=cols)'''

match3 = re.search(old3_pattern, src, re.DOTALL)
if match3:
    src = src[:match3.start()] + new3 + src[match3.end():]
    print("✅ 3/3: Upgraded _optimise_weights (maximize return, skip small trades)")
else:
    print("❌ 3/3: Could not find _optimise_weights")

# Save
lines = src.split('\n')
nb['cells'][19]['source'] = [line + '\n' for line in lines[:-1]]
if lines[-1]:
    nb['cells'][19]['source'].append(lines[-1])

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print()
print("=" * 60)
print("AGGRESSIVE OPTIMIZER INSTALLED!")
print("=" * 60)
print()
print("Key changes to beat buy-and-hold:")
print("  1. Severity threshold: 40% → 60% (stay aggressive longer)")
print("  2. Momentum boost: +5% max in low-risk periods")
print("  3. Defense trigger: only when severity >= 1.0")
print("  4. Skip small trades: < 5% change = no rebalance (save costs)")
print()
print("RESTART KERNEL and re-run all cells to see improved results.")
