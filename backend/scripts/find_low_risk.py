#!/usr/bin/env python3
import json
import os

CRITICAL_KEYWORDS_FILE = 'backend/data/keywords/keywords_critical.json'

# Load the data
with open('frontend/public/credit_card_listings_full.json', 'r') as f:
    data = json.load(f)

# Load critical keywords from JSON file
if os.path.exists(CRITICAL_KEYWORDS_FILE):
    with open(CRITICAL_KEYWORDS_FILE, 'r', encoding='utf-8') as f:
        critical_keywords_dict = json.load(f)
    critical_keywords = [kw.lower() for kw in critical_keywords_dict.keys()]
else:
    critical_keywords = []

low_risk_items = []
for i, item in enumerate(data):
    text = (item.get('Title', '') + ' ' + item.get('Description', '') + ' ' + item.get('Category', '')).lower()
    if not any(kw in text for kw in critical_keywords):
        low_risk_items.append((i, item))
        if len(low_risk_items) >= 5:
            break

print(f'Found {len(low_risk_items)} potential low-risk items:')
for idx, item in low_risk_items:
    print(f'Item {idx}: {item["Title"]}')
    print(f'  Category: {item["Category"]}')
    print(f'  Description: {item["Description"][:100]}...')
    print() 