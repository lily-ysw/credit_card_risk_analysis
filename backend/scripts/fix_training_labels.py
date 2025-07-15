#!/usr/bin/env python3
"""
Fix training data labels to be consistent with keyword logic.
This script ensures that "Cards and CVV" items are labeled as "Critical" risk.
"""

import json
import pandas as pd
from pathlib import Path

def fix_training_labels():
    """Fix the training data labels to be consistent with keyword logic."""
    
    # Load the full dataset
    data_path = Path("../data/credit_card_listings_full.json")
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return
    
    print("Loading full dataset...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items")
    
    # Count items by category and current risk level
    category_risk_counts = {}
    for item in data:
        category = item.get('category', 'Unknown')
        risk = item.get('risk_level', 'Unknown')
        if category not in category_risk_counts:
            category_risk_counts[category] = {}
        if risk not in category_risk_counts[category]:
            category_risk_counts[category][risk] = 0
        category_risk_counts[category][risk] += 1
    
    print("\nCurrent risk level distribution by category:")
    for category, risks in category_risk_counts.items():
        print(f"\n{category}:")
        for risk, count in risks.items():
            print(f"  {risk}: {count}")
    
    # Fix "Cards and CVV" items to be Critical
    fixed_count = 0
    for item in data:
        if item.get('category') == 'Cards and CVV':
            if item.get('risk_level') != 'Critical':
                old_risk = item.get('risk_level', 'Unknown')
                item['risk_level'] = 'Critical'
                fixed_count += 1
                print(f"Fixed: {item.get('title', 'N/A')[:50]}... | {old_risk} -> Critical")
    
    print(f"\nFixed {fixed_count} 'Cards and CVV' items to Critical risk")
    
    # Save the fixed data
    output_path = Path("../data/credit_card_listings_fixed.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved fixed data to {output_path}")
    
    # Also update the CSV files for ML training
    print("\nUpdating CSV files for ML training...")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Split into train/test (80/20 split)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['risk_level'])
    
    # Save train and test CSV files
    train_path = Path("../data/credit_card_train.csv")
    test_path = Path("../data/credit_card_test.csv")
    
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    
    print(f"Saved train data ({len(train_df)} samples) to {train_path}")
    print(f"Saved test data ({len(test_df)} samples) to {test_path}")
    
    # Show new distribution
    print("\nNew risk level distribution by category:")
    new_category_risk_counts = {}
    for item in data:
        category = item.get('category', 'Unknown')
        risk = item.get('risk_level', 'Unknown')
        if category not in new_category_risk_counts:
            new_category_risk_counts[category] = {}
        if risk not in new_category_risk_counts[category]:
            new_category_risk_counts[category][risk] = 0
        new_category_risk_counts[category][risk] += 1
    
    for category, risks in new_category_risk_counts.items():
        print(f"\n{category}:")
        for risk, count in risks.items():
            print(f"  {risk}: {count}")
    
    # Show overall risk distribution
    print("\nOverall risk level distribution:")
    risk_counts = {}
    for item in data:
        risk = item.get('risk_level', 'Unknown')
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    for risk, count in sorted(risk_counts.items()):
        percentage = (count / len(data)) * 100
        print(f"  {risk}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    fix_training_labels() 