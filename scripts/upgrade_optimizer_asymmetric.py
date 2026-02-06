#!/usr/bin/env python3
"""
Upgrade Section 7 optimizer to ASYMMETRIC mode:
- Only go defensive when bubble risk is HIGH (severity >= 1.0)
- In LOW risk regimes, stay at original weights (reduce unnecessary turnover)
- Add slight momentum boost in low-risk periods
"""
import json
import re

NB_PATH = '/Users/saahildey/Bubble Resilient Portfolio/Bubble_Resilient_Portfolio.ipynb'

with open(NB_PATH, 'r') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][19]['source'])

# ============================================================
# REPLACEMENT 1: _sev_from_prob - asymmetric thresholding
# ============================================================
old1 = """def _sev_from_prob(p: float) -> float:
    return float(np.clip(p * 2.0, 0, 2.5))"""

new1 = """def _sev_from_prob(p: float) -> float:
    # ASYMMETRIC: only significant severity when probability > 40%
    if p < 0.40:
        return 0.0  # LOW risk = no defensive action needed
    elif p < 0.60:
        return float((p - 0.40) / 0.20)  # 0.40->0.60 maps to 0->1 (mild)
    else:
        return float(1.0 + (p - 0.60) / 0.40 * 1.5)  # 0.60->1.0 maps to 1->2.5"""

if old1 in src:
    src = src.replace(old1, new1)
    print("✅ 1/3: Replaced _sev_from_prob (asymmetric thresholding)")
else:
    print("❌ 1/3: Could not find _sev_from_prob (may already be updated)")

# ============================================================
# REPLACEMENT 2: _bubble_adj_mu - asymmetric penalty + momentum
# ============================================================
old2 = """def _bubble_adj_mu(train_rets, risk_sc, max_sv):
    mu = train_rets.mean() * 252
    penalty = risk_sc.fillna(0.0) * 0.02 * max_sv
    return mu - penalty"""

new2 = """def _bubble_adj_mu(train_rets, risk_sc, max_sv):
    mu = train_rets.mean() * 252
    # ASYMMETRIC: only penalize when severity is HIGH
    if max_sv < 0.5:
        # LOW RISK: no penalty, slight momentum boost for top performers
        mom_60d = train_rets.tail(60).mean() * 252
        boost = (mom_60d - mu).clip(lower=-0.02, upper=0.03)  # small momentum tilt
        return mu + boost * 0.5
    elif max_sv < 1.0:
        # MEDIUM RISK: mild penalty only
        penalty = risk_sc.fillna(0.0) * 0.01 * max_sv
        return mu - penalty
    else:
        # HIGH RISK: full defensive penalty
        penalty = risk_sc.fillna(0.0) * 0.02 * max_sv
        return mu - penalty"""

if old2 in src:
    src = src.replace(old2, new2)
    print("✅ 2/3: Replaced _bubble_adj_mu (asymmetric penalty + momentum boost)")
else:
    print("❌ 2/3: Could not find _bubble_adj_mu (may already be updated)")

# ============================================================
# REPLACEMENT 3: _optimise_weights - asymmetric optimizer
# ============================================================
# Find and replace the optimizer function
old3_pattern = r'def _optimise_weights\(train_rets, base_w, risk_sc, max_sv, gross, maxw, signs\):.*?return pd\.Series\(w_out, index=cols\)'

new3 = """def _optimise_weights(train_rets, base_w, risk_sc, max_sv, gross, maxw, signs):
    cols = list(base_w.index)
    R = train_rets.reindex(columns=cols).fillna(0.0)
    if R.shape[0] < 120:
        return base_w
    
    # ASYMMETRIC: Skip optimization when risk is low (reduce turnover/costs)
    if max_sv < 0.5:
        # LOW RISK: stay at base weights (don't churn - save transaction costs)
        return base_w
    
    mu_adj = _bubble_adj_mu(R, risk_sc.reindex(cols).fillna(0.0), max_sv).values
    cov = R.cov().values * 252
    w0 = base_w.reindex(cols).fillna(0.0).values
    orig_ret = float(w0 @ mu_adj)

    n = len(cols)
    bnds = []
    for i in range(n):
        if signs.get(cols[i], 1) >= 0:
            bnds.append((0.0, maxw))
        else:
            bnds.append((-maxw, 0.0))

    # MAXIMIZE adjusted return subject to not exceeding original risk
    orig_var = float(w0 @ cov @ w0)
    # Only reduce risk when severity is HIGH (>= 1.0)
    target_var = orig_var * (1.0 if max_sv < 1.0 else 0.85)

    cons = [
        {'type': 'ineq', 'fun': lambda w: target_var - float(w @ cov @ w)},  # variance <= target
        {'type': 'eq',   'fun': lambda w: float(np.abs(w).sum()) - gross},
    ]
    # Maximize adjusted return (minimize negative)
    res = minimize(lambda w: -float(w @ mu_adj), w0, method='SLSQP',
                   jac=lambda w: -mu_adj, bounds=bnds, constraints=cons,
                   options={'maxiter': 800, 'ftol': 1e-10})
    w_out = res.x if res.success else w0
    g = float(np.abs(w_out).sum())
    if g > 0:
        w_out *= gross / g
    return pd.Series(w_out, index=cols)"""

match = re.search(old3_pattern, src, re.DOTALL)
if match:
    src = src[:match.start()] + new3 + src[match.end():]
    print("✅ 3/3: Replaced _optimise_weights (asymmetric optimizer)")
else:
    print("❌ 3/3: Could not find _optimise_weights (may already be updated)")

# Save
lines = src.split('\n')
nb['cells'][19]['source'] = [line + '\n' for line in lines[:-1]]
if lines[-1]:
    nb['cells'][19]['source'].append(lines[-1])

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print()
print("=" * 60)
print("UPGRADE COMPLETE!")
print("=" * 60)
print()
print("Key changes:")
print("  1. Severity threshold raised: Only defensive when prob > 40%")
print("  2. Momentum boost: Slight tilt toward recent winners in low-risk periods")
print("  3. Skip rebalancing: No trades when max_severity < 0.5 (saves costs)")
print("  4. Asymmetric risk: Only reduce variance when severity >= 1.0")
print()
print("Re-run Cell 20 (Section 7) to see improved results.")
