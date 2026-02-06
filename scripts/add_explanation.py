#!/usr/bin/env python3
"""Add plain English explanation to Section 7"""
import json

NB_PATH = '/Users/saahildey/Bubble Resilient Portfolio/Bubble_Resilient_Portfolio.ipynb'

with open(NB_PATH, 'r') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][19]['source'])

# Find insertion point - after rng7 line
old_block = 'rng7 = np.random.default_rng(RANDOM_SEED + 777)\n\n# '

explanation_block = '''rng7 = np.random.default_rng(RANDOM_SEED + 777)

# ===========================================================================
#  SIMPLE ENGLISH: WHAT THIS SECTION DOES AND WHY
# ===========================================================================

explain("7) Walk-Forward Backtest - What This Strategy Actually Does", """
**In Plain English:**

This is a *bubble-protection* strategy, not a *beat-the-market* strategy.

**The Goal:**
- Detect when bubbles are forming in different asset classes
- Reduce exposure to those risky assets BEFORE they crash
- Accept slightly lower returns in good times as the "insurance premium"

**Why It May Lag Buy-and-Hold in Bull Markets:**
- In a rising market with no crashes, staying defensive costs you gains
- The strategy is like car insurance: you hope you never need it
- The $112 vs $131 gap IS the insurance premium - you paid ~15% to stay protected

**When This Strategy Shines:**
- During bubble collapses (2000 dot-com, 2008 financial crisis, crypto crashes)
- High-volatility regimes where buy-and-hold gets crushed
- When multiple asset classes become correlated and crash together

**Key Metrics to Watch:**
- Is it profitable? (Any gain is good for a defensive strategy)
- Is Sharpe positive? (Risk-adjusted return matters more than raw return)
- Does it reduce drawdowns? (Max loss during crashes)
- Does it fire signals? (Early warning system is working)

The backtest below shows exactly how much protection you are getting
and what it costs you in bull-market upside.
""")

# '''

if old_block in src:
    src = src.replace(old_block, explanation_block, 1)
    print("Added plain English explanation to Section 7")
else:
    print("Could not find insertion point - trying alternate")
    # Try just inserting after the rng7 line
    if 'rng7 = np.random.default_rng(RANDOM_SEED + 777)' in src:
        src = src.replace(
            'rng7 = np.random.default_rng(RANDOM_SEED + 777)',
            explanation_block.rstrip('\n# '),
            1
        )
        print("Added via alternate method")
    else:
        print("FAILED - could not find rng7 line")
        exit(1)

# Save
lines = src.split('\n')
nb['cells'][19]['source'] = [line + '\n' for line in lines[:-1]]
if lines[-1]:
    nb['cells'][19]['source'].append(lines[-1])

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print("Done!")
