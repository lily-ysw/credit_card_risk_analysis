import json

# Paths
input_path = '../../frontend/public/credit_card_listings_full.json'
output_path = '../../frontend/public/credit_card_listings_initial.json'

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Take first 100 items
initial = data[:100]

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(initial, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(initial)} items to {output_path}") 