#!/usr/bin/env python3
"""Fix corrupted explain blocks in Cells 18 and 20."""
import json
import re

NB_PATH = '/Users/saahildey/Bubble Resilient Portfolio/Bubble_Resilient_Portfolio.ipynb'

with open(NB_PATH, 'r') as f:
    nb = json.load(f)

def fix_cell(cell_idx, section_name, explain_title):
    """Fix a corrupted explain block in a cell."""
    src = ''.join(nb['cells'][cell_idx]['source'])
    
    # Try to find the markdown display ending and corruption
    # Pattern: """)) followed by garbage
    
    # Find the last occurrence of """))
    last_md_end = src.rfind('"""))')
    if last_md_end == -1:
        print(f"Cell {cell_idx+1}: No markdown display end found")
        return False
    
    # Find the first occurrence of "# Dynamic explanation for Section" after last_md_end
    dyn_idx = src.find('# Dynamic explanation for Section', last_md_end)
    if dyn_idx == -1:
        print(f"Cell {cell_idx+1}: No dynamic explanation marker found")
        return False
    
    # Clean source: everything up to and including """)) plus newlines
    clean_end = last_md_end + len('"""))')
    clean_src = src[:clean_end] + '\n\n'
    
    # Add proper explain block based on section
    if section_name == '6.2':
        clean_src += '''# Dynamic explanation for Section 6.2
if SHOW_EXPLANATIONS and 'explain' in globals():
    try:
        _bullets_62 = [
            f"Verdict: {verdict}",
            f"Return change: {delta_return:+.2%} (Original {original_m:.2%} → Rebalanced {rebalanced_m:.2%})",
            f"Risk change: {delta_risk:+.2%} (Original {original_s:.2%} → Rebalanced {rebalanced_s:.2%})",
            f"Drawdown protection: {'Improved' if dd_improved else 'Similar'} in historical crashes",
            "All metrics computed from live Yahoo Finance data.",
        ]
        explain('6.2) Dashboard summary', _bullets_62)
    except Exception:
        pass
'''
    elif section_name == '7':
        clean_src += '''# Dynamic explanation for Section 7
if SHOW_EXPLANATIONS and 'explain' in globals():
    try:
        _bullets_7 = [
            f"Verdict: {final_verdict}",
            f"Terminal value: ${terminal_rebal:.2f} from $100 starting capital",
            f"Total P&L: ${total_pnl:+.2f} (OOS, after {TCOST_BPS:.0f}bps transaction costs)",
            f"Sharpe ratio: {sharpe_rebal:.2f} (vs {sharpe_orig:.2f} original)",
            f"Regime performance: Wins {n_win}/{n_total} market regimes",
            f"Bubble signals detected: {n_signals} events across AI/Credit/Crypto",
        ]
        explain('7) Walk-forward backtest results', _bullets_7)
    except Exception:
        pass
'''
    
    # Convert to list of lines
    lines = clean_src.split('\n')
    nb['cells'][cell_idx]['source'] = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        nb['cells'][cell_idx]['source'].append(lines[-1])
    
    print(f"Cell {cell_idx+1} (Section {section_name}): Fixed explain block")
    return True

# Fix Cell 18 (index 17) - Section 6.2
fix_cell(17, '6.2', '6.2) Dashboard summary')

# Fix Cell 20 (index 19) - Section 7
fix_cell(19, '7', '7) Walk-forward backtest results')

# Save
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print("\nDone! Verifying syntax...")

# Verify
with open(NB_PATH, 'r') as f:
    nb = json.load(f)

for cell_idx in [17, 19]:
    src = ''.join(nb['cells'][cell_idx]['source'])
    try:
        compile(src, f'<cell{cell_idx+1}>', 'exec')
        print(f"Cell {cell_idx+1}: Syntax OK")
    except SyntaxError as e:
        print(f"Cell {cell_idx+1}: Syntax error at line {e.lineno}: {e.msg}")
