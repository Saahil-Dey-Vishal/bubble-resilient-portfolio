#!/usr/bin/env python3
"""Verify Section 6.2 fixes."""

import json

NB_PATH = '/Users/saahildey/Bubble Resilient Portfolio/Bubble_Resilient_Portfolio.ipynb'

with open(NB_PATH) as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'SECTION 6.2' in src[:500]:
            print(f'Cell {i} - Section 6.2 code cell')
            print(f'Lines: {len(cell.get("source", []))}')
            
            # Check for key fixes
            if "set_xlabel('Risk" in src:
                print('OK: X-axis = Risk (fixed)')
            else:
                print('WARN: X-axis not set to Risk')
                
            if "set_ylabel('Expected Return" in src:
                print('OK: Y-axis = Return (fixed)')
            else:
                print('WARN: Y-axis not set to Return')
                
            if 'max_drawdown' in src:
                print('OK: Max drawdown function present')
            else:
                print('WARN: No max_drawdown function')
                
            if 'pie' not in src.lower():
                print('OK: Pie charts removed')
            else:
                print('WARN: Pie charts still present')
                
            if 'orig_paths' in src and 'rebal_paths' in src:
                print('OK: Monte Carlo compares both portfolios')
            else:
                print('WARN: Monte Carlo not comparing both')
            
            break
