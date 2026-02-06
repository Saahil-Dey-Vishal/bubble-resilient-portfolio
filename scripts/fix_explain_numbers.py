#!/usr/bin/env python3
"""Fix incorrect explain() section numbers in the notebook."""
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('Bubble_Resilient_Portfolio.ipynb', 'r') as f:
    nb = json.load(f)

fixes_made = 0

# Fix Cell 11 (index 11): '5) Portfolio' → '4) Portfolio'
for i, line in enumerate(nb['cells'][11]['source']):
    if "'5) Portfolio allocation" in line:
        nb['cells'][11]['source'][i] = line.replace("'5) Portfolio allocation", "'4) Portfolio allocation")
        print("Fixed Cell 11: '5) Portfolio allocation' -> '4) Portfolio allocation'")
        fixes_made += 1
        break

# Fix Cell 13 (index 13): '12) Bubble early-warning' → '5) Bubble early-warning'
for i, line in enumerate(nb['cells'][13]['source']):
    if "'12) Bubble early-warning" in line:
        nb['cells'][13]['source'][i] = line.replace("'12) Bubble early-warning", "'5) Bubble early-warning")
        print("Fixed Cell 13: '12) Bubble early-warning' -> '5) Bubble early-warning'")
        fixes_made += 1
        break

# Fix Cell 15 (index 15): '13) Rebalancing' → '6) Rebalancing'
for i, line in enumerate(nb['cells'][15]['source']):
    if "'13) Rebalancing" in line:
        nb['cells'][15]['source'][i] = line.replace("'13) Rebalancing", "'6) Rebalancing")
        print("Fixed Cell 15: '13) Rebalancing' -> '6) Rebalancing'")
        fixes_made += 1
        break

# Save
with open('Bubble_Resilient_Portfolio.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\nFixed {fixes_made} explain() titles")
