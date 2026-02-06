#!/usr/bin/env python3
"""Fix the explain block - pass list of strings, not a single string"""
import json
import re

NB_PATH = '/Users/saahildey/Bubble Resilient Portfolio/Bubble_Resilient_Portfolio.ipynb'

with open(NB_PATH, 'r') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][19]['source'])

# Find and replace the bad explain block
old_pattern = r'explain\("7\) Walk-Forward Backtest - What This Strategy Actually Does".*?\"\"\"\)'

new_explain = '''explain("7) Walk-Forward Backtest - What This Strategy Actually Does", [
    "**In Plain English:** This is a *bubble-protection* strategy, not a *beat-the-market* strategy.",
    "**The Goal:** Detect bubbles forming, reduce exposure BEFORE crashes, accept lower returns as insurance premium.",
    "**Why It May Lag Buy-and-Hold:** In bull markets with no crashes, staying defensive costs gains. The $112 vs $131 gap IS the insurance premium (~15%).",
    "**When This Strategy Shines:** Bubble collapses (2000/2008/crypto), high-volatility regimes, correlated crashes.",
    "**Key Metrics:** Profitable? Positive Sharpe? Reduced drawdowns? Fired signals? The backtest below shows protection vs cost."
])'''

match = re.search(old_pattern, src, re.DOTALL)
if match:
    src = src[:match.start()] + new_explain + src[match.end():]
    print("Fixed explain block - now passes list of strings")
else:
    print("Could not find the explain block to fix")
    exit(1)

# Save
lines = src.split('\n')
nb['cells'][19]['source'] = [line + '\n' for line in lines[:-1]]
if lines[-1]:
    nb['cells'][19]['source'].append(lines[-1])

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print("Done!")
