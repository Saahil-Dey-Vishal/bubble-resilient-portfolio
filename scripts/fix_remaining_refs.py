#!/usr/bin/env python3
"""
Fix remaining Backtest Confidence references after main changes.
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
        
        # 1. Fix model card framing text
        if "Evidence (Confidence + Backtest Confidence)" in source_text:
            new_source_text = new_source_text.replace(
                "Evidence (Confidence + Backtest Confidence)",
                "Evidence (Confidence) + Stage (Babson Stage)"
            )
            modified = True
            print(f"Fixed model card framing in cell {i}")
        
        # 2. Remove bt = float(r.get('Backtest Confidence...
        if "bt = float(r.get('Backtest Confidence (0-100)'" in source_text:
            new_source_text = re.sub(
                r"\n\s*bt = float\(r\.get\('Backtest Confidence \(0-100\)'[^\n]+\n",
                "\n",
                new_source_text
            )
            modified = True
            print(f"Removed bt variable in cell {i}")
        
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
        print(f"\n✓ Fixed {changes_made} cells. Notebook saved.")
    else:
        print("\nNo additional fixes were needed.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
