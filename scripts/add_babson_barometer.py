#!/usr/bin/env python3
"""
Add Babson Barometer and remove LPPL columns from EWS table.

Changes:
1. Add Babson Barometer function to classify bubble stages
2. Add 'Babson Stage' column to EWS table
3. Remove 'Backtest Confidence (0-100)', 'LPPL endgame (0-100)', 'LPPL tc (days)' columns
4. Update format dict and documentation bullets
"""

import json
import re
import sys

NB_PATH = '/Users/saahildey/Bubble Resilient Portfolio/Bubble_Resilient_Portfolio.ipynb'

def main():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    
    changes_made = 0
    
    for i, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        
        source = cell.get('source', [])
        if isinstance(source, str):
            source = [source]
        
        source_text = ''.join(source)
        modified = False
        new_source_text = source_text
        
        # 1. Add Babson Barometer function after prob_ci calculation and before rows.append
        if "prob_ci = ''" in source_text and "rows.append(" in source_text and "'LPPL endgame (0-100)'" in source_text:
            # Find the pattern: prob_ci block ending with rows.append(
            old_pattern = r"(    prob_ci = ''\n    try:\n.*?except Exception:\n        prob_ci = '')\n    rows\.append\("
            
            babson_code = '''
    # --- Babson Barometer: classify bubble stage based on probability + momentum ---
    # Stages: Stealth (<25%) -> Awareness (25-50%) -> Mania (50-80%) -> Blow-off (>80%) -> Return to Mean (falling fast)
    def _babson_stage(prob: float, delta_30d: float) -> str:
        if not np.isfinite(prob):
            return '—'
        if np.isfinite(delta_30d) and prob >= 60 and delta_30d < -10:
            return 'Return to Mean'
        if prob >= 80:
            return 'Blow-off'
        if prob >= 50:
            return 'Mania'
        if prob >= 25:
            return 'Awareness'
        return 'Stealth'

    babson_stage = _babson_stage(p_now, delta_pp)
'''
            
            # Replace pattern
            new_source_text = re.sub(
                old_pattern,
                r"\1" + babson_code + "\n    rows.append(",
                new_source_text,
                flags=re.DOTALL
            )
            
            if new_source_text != source_text:
                modified = True
                print(f"Added Babson Barometer function in cell {i}")
        
        # 2. Replace LPPL columns with Babson Stage in rows.append dict
        # Remove: 'Backtest Confidence (0-100)', 'LPPL endgame (0-100)', 'LPPL tc (days)'
        # Add: 'Babson Stage'
        if "'LPPL endgame (0-100)'" in new_source_text and "rows.append(" in new_source_text:
            # Remove Backtest Confidence line
            new_source_text = re.sub(
                r"\n\s*'Backtest Confidence \(0-100\)':[^\n]+,", 
                "", 
                new_source_text
            )
            # Remove LPPL endgame line
            new_source_text = re.sub(
                r"\n\s*'LPPL endgame \(0-100\)':[^\n]+,", 
                "", 
                new_source_text
            )
            # Remove LPPL tc line
            new_source_text = re.sub(
                r"\n\s*'LPPL tc \(days\)':[^\n]+,", 
                "", 
                new_source_text
            )
            
            # Add Babson Stage after Confidence (0-100)
            new_source_text = re.sub(
                r"('Confidence \(0-100\)': conf,)", 
                r"\1\n            'Babson Stage': babson_stage,", 
                new_source_text
            )
            
            if new_source_text != source_text:
                modified = True
                print(f"Updated rows.append to add Babson Stage and remove LPPL columns in cell {i}")
        
        # 3. Update format dict (fmt)
        if "'Backtest Confidence (0-100)': '{:.0f}'" in new_source_text:
            # Remove Backtest Confidence from fmt
            new_source_text = re.sub(
                r"\n\s*'Backtest Confidence \(0-100\)':\s*'\{:\.0f\}',?", 
                "", 
                new_source_text
            )
            # Remove LPPL endgame from fmt
            new_source_text = re.sub(
                r"\n\s*'LPPL endgame \(0-100\)':\s*'\{:\.0f\}',?", 
                "", 
                new_source_text
            )
            # Remove LPPL tc from fmt
            new_source_text = re.sub(
                r"\n\s*'LPPL tc \(days\)':\s*'\{:\.0f\}',?", 
                "", 
                new_source_text
            )
            
            if new_source_text != source_text:
                modified = True
                print(f"Removed LPPL columns from format dict in cell {i}")
        
        # 4. Update documentation bullets
        if "'Backtest Confidence (0–100):" in new_source_text or "LPPL endgame (0-100):" in new_source_text:
            # Remove Backtest Confidence bullet
            new_source_text = re.sub(
                r"\n\s*'Backtest Confidence \(0[-–]100\):[^']*',?", 
                "", 
                new_source_text
            )
            # Remove LPPL endgame bullet
            new_source_text = re.sub(
                r"\n\s*'LPPL endgame \(0-100\):[^']*',?", 
                "", 
                new_source_text
            )
            # Remove LPPL tc bullet
            new_source_text = re.sub(
                r"\n\s*'LPPL tc \(days\):[^']*',?", 
                "", 
                new_source_text
            )
            
            # Add Babson Stage bullet after Confidence bullet
            new_source_text = re.sub(
                r"('Confidence \(0[-–]100\):[^']*stability[^']*',)",
                r"\1\n                    'Babson Stage: bubble lifecycle stage (Stealth → Awareness → Mania → Blow-off → Return to Mean).',",
                new_source_text
            )
            
            if new_source_text != source_text:
                modified = True
                print(f"Updated documentation bullets in cell {i}")
        
        # 5. Also remove the backfill line for Backtest Confidence
        if "simple_tbl['Backtest Confidence (0-100)']" in new_source_text:
            new_source_text = re.sub(
                r"\n.*simple_tbl\['Backtest Confidence \(0-100\)'\].*\n", 
                "\n", 
                new_source_text
            )
            if new_source_text != source_text:
                modified = True
                print(f"Removed Backtest Confidence backfill in cell {i}")
        
        # 6. Remove backtest_confidence from audit log payload (optional cleanup)
        if "'backtest_confidence_0_100'" in new_source_text:
            new_source_text = re.sub(
                r"\n\s*'backtest_confidence_0_100':[^\n]+,?",
                "",
                new_source_text
            )
            if new_source_text != source_text:
                modified = True
                print(f"Removed backtest_confidence from audit log in cell {i}")
        
        if modified:
            # Convert back to source list format
            new_source_lines = new_source_text.split('\n')
            # Preserve trailing newlines properly
            new_source = [line + '\n' for line in new_source_lines[:-1]]
            if new_source_lines:
                new_source.append(new_source_lines[-1])
            cell['source'] = new_source
            changes_made += 1

    if changes_made > 0:
        with open(NB_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"\n✓ Made changes to {changes_made} cells. Notebook saved.")
    else:
        print("\nNo changes were made. Patterns not found.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
