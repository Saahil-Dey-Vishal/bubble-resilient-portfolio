#!/usr/bin/env python3
"""Fix the corrupted explain block in Cell 18 (Section 6.2)."""
import json

with open('/Users/saahildey/Bubble Resilient Portfolio/Bubble_Resilient_Portfolio.ipynb', 'r') as f:
    nb = json.load(f)

# Cell 18 is index 17
src = ''.join(nb['cells'][17]['source'])

# Find the start of the corruption
marker = "# Dynamic explanation for Section 6.2\n\nif 'explain' in globals():        pass"

idx = src.find(marker)
if idx == -1:
    print('Marker not found - trying alternate pattern')
    # Try simpler pattern
    marker = "if 'explain' in globals():        pass"
    idx = src.find(marker)
    if idx != -1:
        # Go back to find # Dynamic 
        back_marker = "# Dynamic explanation for Section 6.2"
        back_idx = src.rfind(back_marker, 0, idx)
        if back_idx != -1:
            idx = back_idx

if idx == -1:
    print('Could not find corruption marker')
else:
    # Everything before the corruption
    clean_src = src[:idx]
    
    # Add proper explain block
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
    
    # Convert to list of lines - each line needs \n except possibly the last
    lines = clean_src.split('\n')
    nb['cells'][17]['source'] = [line + '\n' for line in lines[:-1]]
    if lines[-1]:  # If last line is non-empty
        nb['cells'][17]['source'].append(lines[-1])
    
    with open('/Users/saahildey/Bubble Resilient Portfolio/Bubble_Resilient_Portfolio.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    
    print('Fixed Cell 18 (Section 6.2) - explain block repaired')
