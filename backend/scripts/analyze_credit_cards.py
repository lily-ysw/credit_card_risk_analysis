#!/usr/bin/env python3
"""
Comprehensive analysis of credit card related listings.
"""

import pandas as pd
import json
import re
from collections import Counter

STOP_WORDS_FILE = '../data/stop_words.txt'

def load_stop_words(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def analyze_credit_card_listings():
    """Analyze credit card listings for patterns and risk indicators."""
    
    # Load the credit card listings
    print("Loading credit card listings...")
    df = pd.read_csv('../data/credit_card_listings.csv', encoding='utf-8')
    print(f"Total credit card listings: {len(df)}")
    
    # Basic statistics
    print(f"\n=== BASIC STATISTICS ===")
    print(f"Categories: {df['Category'].value_counts().to_dict()}")
    
    # Price analysis
    try:
        prices = df['Price'].str.extract(r'(\d+\.?\d*)').astype(float)
        avg_price = prices.mean()
        min_price = prices.min()
        max_price = prices.max()
        print(f"Average price: ${avg_price:.2f}")
        print(f"Price range: ${min_price:.2f} - ${max_price:.2f}")
    except:
        print("Price analysis: Unable to parse prices")
    
    # Extract common terms from titles and descriptions
    print(f"\n=== COMMON TERMS ANALYSIS ===")
    
    # Combine titles and descriptions
    all_text = ' '.join(df['Title'].astype(str) + ' ' + df['Description'].astype(str))
    
    # Extract words (simple tokenization)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    
    # Load stop words from file
    stop_words = load_stop_words(STOP_WORDS_FILE)
    
    # Filter out common stop words
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    print("Top 20 most common terms:")
    for word, count in word_counts.most_common(20):
        print(f"  {word}: {count}")
    
    # Analyze by category
    print(f"\n=== CATEGORY ANALYSIS ===")
    for category in df['Category'].unique():
        category_df = df[df['Category'] == category]
        print(f"\n{category} ({len(category_df)} listings):")
        
        # Common terms in this category
        category_text = ' '.join(category_df['Title'].astype(str) + ' ' + category_df['Description'].astype(str))
        category_words = re.findall(r'\b[a-zA-Z]{3,}\b', category_text.lower())
        category_filtered = [word for word in category_words if word not in stop_words]
        category_counts = Counter(category_filtered)
        
        print("  Top terms:")
        for word, count in category_counts.most_common(10):
            print(f"    {word}: {count}")
    
    # Risk assessment based on keywords
    print(f"\n=== RISK ASSESSMENT ===")
    
    # Load keyword classifications
    with open('../data/keywords/keywords_critical.json', 'r') as f:
        critical_keywords = json.load(f)
    with open('../data/keywords/keywords_high.json', 'r') as f:
        high_keywords = json.load(f)
    
    # Analyze each listing for risk keywords
    risk_analysis = []
    
    for idx, row in df.iterrows():
        text = f"{row['Title']} {row['Description']}".lower()
        
        critical_matches = [kw for kw in critical_keywords.keys() if kw.lower() in text]
        high_matches = [kw for kw in high_keywords.keys() if kw.lower() in text]
        
        risk_score = len(critical_matches) * 3 + len(high_matches) * 2
        
        risk_analysis.append({
            'title': row['Title'],
            'category': row['Category'],
            'critical_keywords': critical_matches,
            'high_keywords': high_matches,
            'risk_score': risk_score
        })
    
    # Sort by risk score
    risk_analysis.sort(key=lambda x: x['risk_score'], reverse=True)
    
    print("Top 10 highest risk listings:")
    for i, item in enumerate(risk_analysis[:10]):
        print(f"\n{i+1}. {item['title']}")
        print(f"   Category: {item['category']}")
        print(f"   Risk Score: {item['risk_score']}")
        print(f"   Critical Keywords: {', '.join(item['critical_keywords'][:5])}")
        print(f"   High Keywords: {', '.join(item['high_keywords'][:5])}")
    
    # Save risk analysis
    with open('../data/credit_card_risk_analysis.json', 'w') as f:
        json.dump(risk_analysis, f, indent=2)
    
    print(f"\nRisk analysis saved to: ../data/credit_card_risk_analysis.json")
    
    return df, risk_analysis

if __name__ == "__main__":
    analyze_credit_card_listings() 