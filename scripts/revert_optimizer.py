#!/usr/bin/env python3
"""
REVERT to ORIGINAL optimizer functions - the conservative defaults that worked.
Then we can make MINIMAL tweaks.
"""
import json
import re

NB_PATH = '/Users/saahildey/Bubble Resilient Portfolio/Bubble_Resilient_Portfolio.ipynb'

with open(NB_PATH, 'r') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][19]['source'])

# ============================================================
# REVERT 1: _sev_from_prob - back to original
# ============================================================
old1_pattern = r'def _sev_from_prob\(p: float\) -> float:.*?(?=\ndef |\nclass )'

new1 = '''def _sev_from_prob(p: float) -> float:
    return float(np.clip(p * 2.0, 0, 2.5))

'''

match1 = re.search(old1_pattern, src, re.DOTALL)
if match1:
    src = src[:match1.start()] + new1 + src[match1.end():]
    print("✅ 1/3: Reverted _sev_from_prob to ORIGINAL")
else:
    print("❌ 1/3: Could not find _sev_from_prob")

# ============================================================
# REVERT 2: _bubble_adj_mu - back to original
# ============================================================
old2_pattern = r'def _bubble_adj_mu\(train_rets, risk_sc, max_sv\):.*?(?=\ndef |\nclass )'

new2 = '''def _bubble_adj_mu(train_rets, risk_sc, max_sv):
    mu = train_rets.mean() * 252
    penalty = risk_sc.fillna(0.0) * 0.02 * max_sv
    return mu - penalty

'''

match2 = re.search(old2_pattern, src, re.DOTALL)
if match2:
    src = src[:match2.start()] + new2 + src[match2.end():]
    print("✅ 2/3: Reverted _bubble_adj_mu to ORIGINAL")
else:
    print("❌ 2/3: Could not find _bubble_adj_mu")

# ============================================================
# REVERT 3: _optimise_weights - back to original (minimize variance)
# ============================================================
old3_pattern = r'def _optimise_weights\(train_rets, base_w, risk_sc, max_sv, gross, maxw, signs\):.*?return pd\.Series\(w_out, index=cols\)'

new3 = '''def _optimise_weights(train_rets, base_w, risk_sc, max_sv, gross, maxw, signs):
    cols = list(base_w.index)
    R = train_rets.reindex(columns=cols).fillna(0.0)
    if R.shape[0] < 120:
        return base_w
    mu_adj = _bubble_adj_mu(R, risk_sc.reindex(cols).fillna(0.0), max_sv).values
    cov = R.cov().values * 252
    w0 = base_w.reindex(cols).fillna(0.0).values
    orig_var = float(w0 @ cov @ w0)

    n = len(cols)
    bnds = []
    for i in range(n):
        if signs.get(cols[i], 1) >= 0:
            bnds.append((0.0, maxw))
        else:
            bnds.append((-maxw, 0.0))

    cons = [
        {'type': 'ineq', 'fun': lambda w: orig_var - float(w @ cov @ w)},
        {'type': 'eq',   'fun': lambda w: float(np.abs(w).sum()) - gross},
    ]
    res = minimize(lambda w: float(w @ cov @ w), w0, method='SLSQP',
                   jac=lambda w: 2 * cov @ w, bounds=bnds, constraints=cons,
                   options={'maxiter': 800, 'ftol': 1e-10})
    w_out = res.x if res.success else w0
    g = float(np.abs(w_out).sum())
    if g > 0:
        w_out *= gross / g
    return pd.Series(w_out, index=cols)'''

match3 = re.search(old3_pattern, src, re.DOTALL)
if match3:
    src = src[:match3.start()] + new3 + src[match3.end():]
    print("✅ 3/3: Reverted _optimise_weights to ORIGINAL")
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
print("REVERTED TO ORIGINAL OPTIMIZER")
print("=" * 60)
print()
print("All 3 functions restored to working defaults.")
print("Restart kernel and re-run to confirm $112 result is back.")
