#!/usr/bin/env python3
"""Update Section 6.2 markdown cell."""

import json

NB_PATH = '/Users/saahildey/Bubble Resilient Portfolio/Bubble_Resilient_Portfolio.ipynb'

NEW_MD = '''---

## 📊 Section 6.2: Post-Rebalancing Analysis Dashboard

> **🎯 Purpose:** Analyze the bubble-aware rebalanced portfolio's **downside protection** — the key metric is crash survival, not raw returns.

**📊 What This Section Produces (3 Charts):**
1. **Risk vs Return Frontier** — Standard axes (X=Risk, Y=Return) showing defensive repositioning
2. **Drawdown Protection** — Max drawdown comparison during Dot-com, GFC, COVID crashes
3. **Crash Survival Monte Carlo** — Both portfolios simulated under bubble crash scenarios

**🛡️ Understanding the "Insurance Premium":**
A bubble-resilient portfolio **intentionally accepts lower expected return** in exchange for crash protection. If the rebalanced portfolio shows lower return AND lower risk, **the system is working correctly**. You're paying an insurance premium now to avoid catastrophic losses when bubbles pop.

**🔄 Live Data:** All charts update using Yahoo Finance prices and CSV roster weights.

---
'''

def main():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'markdown':
            src = ''.join(cell.get('source', []))
            if 'Section 6.2' in src and 'Post-Rebalancing' in src:
                print(f'Found markdown cell {i}, updating...')
                
                lines = NEW_MD.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]]
                if lines:
                    cell['source'].append(lines[-1])
                
                with open(NB_PATH, 'w', encoding='utf-8') as f:
                    json.dump(nb, f, indent=1, ensure_ascii=False)
                
                print('✓ Updated Section 6.2 markdown!')
                return 0
    
    print('Section 6.2 markdown not found.')
    return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
