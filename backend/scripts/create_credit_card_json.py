#!/usr/bin/env python3
"""
Convert credit card CSV data to JSON for frontend
"""

import pandas as pd
import json
import os
import re

def load_keywords_from_files():
    keywords = {
        'critical': [],
        'high': [],
        'medium': [],
        'low': []
    }
    risk_files = {
        'critical': '../data/keywords/keywords_critical.json',
        'high': '../data/keywords/keywords_high.json',
        'medium': '../data/keywords/keywords_medium.json',
        'low': '../data/keywords/keywords_low.json'
    }
    for risk_level, file_path in risk_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    keywords[risk_level] = json.load(f)
            except Exception as e:
                print(f"Error loading {risk_level} keywords: {e}")
        else:
            print(f"Warning: {file_path} not found")
    return keywords

CREDIT_CARD_CATEGORIES = {
    'Cards and CVV': 'Critical',
    'Carding': 'Critical',
    'SSN': 'Critical',
    'Dump': 'Critical',
    'Drop Bank': 'High',
    'Physical Drop': 'High'
}

def create_credit_card_risk_labels(title, description, category, price, keywords):
    # Focus on the actual content, not the category
    text = f"{title} {description}".lower()
    
    # Count keyword matches for each risk level
    critical_count = sum(1 for keyword in keywords['critical'] if keyword.lower() in text)
    high_count = sum(1 for keyword in keywords['high'] if keyword.lower() in text)
    medium_count = sum(1 for keyword in keywords['medium'] if keyword.lower() in text)
    low_count = sum(1 for keyword in keywords['low'] if keyword.lower() in text)
    
    # Determine risk level based on keyword density and severity
    if critical_count >= 2:
        return 'Critical'
    elif critical_count == 1 or high_count >= 3:
        return 'High'
    elif high_count >= 1 or medium_count >= 2:
        return 'Medium'
    elif medium_count == 1 or low_count >= 2:
        return 'Low'
    else:
        return 'Low'  # Default to low if no keywords found

def create_credit_card_json():
    """Create JSON files from credit card CSV data"""
    
    # Read the credit card CSV data
    try:
        df = pd.read_csv('../data/credit_card_listings.csv')
        print(f"Loaded {len(df)} credit card listings from CSV")
        
        # Create a sample of 100 items for faster loading
        df_sample = df.head(100)
        
        # Convert to list of dictionaries for JSON
        data_list = []
        keywords = load_keywords_from_files()
        for _, row in df.iterrows():
            title = str(row['title']) if pd.notna(row['title']) else ''
            description = str(row['description']) if pd.notna(row['description']) else ''
            category = str(row['category']) if pd.notna(row['category']) else ''
            price = str(row['price']) if pd.notna(row['price']) else ''
            location = str(row['location']) if pd.notna(row['location']) else ''
            risk_level = create_credit_card_risk_labels(title, description, category, price, keywords)
            item = {
                'title': title,
                'description': description,
                'category': category,
                'price': price,
                'location': location,
                'risk_level': risk_level
            }
            data_list.append(item)
        
        # Save to frontend/public directory
        os.makedirs('frontend/public', exist_ok=True)
        
        # Save full dataset
        with open('frontend/public/credit_card_listings_full.json', 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        print(f"Created credit_card_listings_full.json with {len(data_list)} items")
        
        return True
        
    except Exception as e:
        print(f"Error creating JSON files: {e}")
        return False

if __name__ == "__main__":
    create_credit_card_json() 