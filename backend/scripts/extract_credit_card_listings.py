#!/usr/bin/env python3
"""
Extract Credit Card related listings from dark market dataset.
"""

import pandas as pd
import json

def extract_credit_card_listings():
    """Extract all credit card related listings from the dataset."""
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('../data/dark_market_output_v2.csv', encoding='latin-1')
    print(f"Total listings: {len(df)}")
    
    # Define credit card related categories
    credit_card_categories = [
        'Cards and CVV',
        'Carding', 
        'SSN',
        'Dump',
        'Drop Bank',
        'Physical Drop'
    ]
    
    # Filter for credit card related listings
    credit_card_df = df[df['Category'].isin(credit_card_categories)]
    
    print(f"\nCredit card related listings found: {len(credit_card_df)}")
    print("\nBreakdown by category:")
    category_counts = credit_card_df['Category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    # Save to CSV
    output_file = '../data/credit_card_listings.csv'
    credit_card_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nSaved to: {output_file}")
    
    # Save sample to JSON for frontend
    sample_size = min(100, len(credit_card_df))
    sample_df = credit_card_df.head(sample_size)
    
    # Convert to JSON format
    sample_listings = []
    for _, row in sample_df.iterrows():
        listing = {
            'title': str(row['Title']),
            'price': str(row['Price']),
            'seller': str(row['Seller']),
            'category': str(row['Category']),
            'description': str(row['Description']),
            'quantity': str(row['Quantity in Stock']),
            'availability': str(row['Availability'])
        }
        sample_listings.append(listing)
    
    # Save sample to JSON
    sample_file = '../data/credit_card_listings_sample.json'
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_listings, f, indent=2, ensure_ascii=False)
    
    print(f"Saved sample to: {sample_file}")
    
    # Show some examples
    print(f"\nSample listings:")
    for i, listing in enumerate(sample_listings[:5]):
        print(f"\n{i+1}. {listing['title']}")
        print(f"   Category: {listing['category']}")
        print(f"   Price: {listing['price']}")
        print(f"   Seller: {listing['seller']}")
    
    return credit_card_df

if __name__ == "__main__":
    extract_credit_card_listings() 