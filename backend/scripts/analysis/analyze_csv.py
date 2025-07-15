#!/usr/bin/env python3
"""
CSV Data Analysis for Machine Learning Opportunities
Analyzes dark_market_output.csv to identify ML use cases
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def clean_price(price_str):
    """Clean price strings to extract numeric values"""
    if pd.isna(price_str):
        return np.nan
    
    # Remove common currency symbols and text
    price_str = str(price_str).upper()
    price_str = re.sub(r'[^\d.,]', '', price_str)
    
    # Handle different formats
    if ',' in price_str and '.' in price_str:
        # Format like "1,234.56"
        price_str = price_str.replace(',', '')
    elif ',' in price_str:
        # Format like "1,234" or "1,234,567"
        parts = price_str.split(',')
        if len(parts[-1]) == 3:  # Last part is 3 digits (thousands)
            price_str = price_str.replace(',', '')
        else:
            price_str = price_str.replace(',', '.')
    
    try:
        return float(price_str)
    except:
        return np.nan

def analyze_text_features(text_series):
    """Analyze text features for ML opportunities"""
    if text_series.empty:
        return {}
    
    # Combine all text
    all_text = ' '.join(text_series.dropna().astype(str))
    
    # Word frequency analysis
    words = re.findall(r'\b\w+\b', all_text.lower())
    word_freq = Counter(words)
    
    # Text length analysis
    text_lengths = text_series.dropna().str.len()
    
    return {
        'total_words': len(words),
        'unique_words': len(set(words)),
        'avg_text_length': text_lengths.mean(),
        'max_text_length': text_lengths.max(),
        'min_text_length': text_lengths.min(),
        'top_words': word_freq.most_common(10)
    }

def main():
    print("🔍 Analyzing dark_market_output_v2.csv for Machine Learning Opportunities")
    print("=" * 70)
    
    # Read CSV
    try:
        df = pd.read_csv('../data/dark_market_output_v2.csv', encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        df = pd.read_csv('../data/dark_market_output_v2.csv', encoding='latin-1', on_bad_lines='skip')
    
    print(f"📊 Dataset Overview:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    print(f"\n📋 Columns:")
    for i, col in enumerate(df.columns, 1):
        non_null = df[col].notna().sum()
        print(f"   {i:2d}. {col:<20} ({non_null:,} non-null values)")
    
    print(f"\n🔍 Data Quality Analysis:")
    missing_data = df.isnull().sum()
    print(f"   Missing values per column:")
    for col, missing in missing_data[missing_data > 0].items():
        percentage = (missing / len(df)) * 100
        print(f"     {col}: {missing:,} ({percentage:.1f}%)")
    
    # Analyze key columns
    print(f"\n💰 Price Analysis:")
    if 'Price' in df.columns:
        df['Price_clean'] = df['Price'].apply(clean_price)
        price_stats = df['Price_clean'].describe()
        print(f"   Price range: ${price_stats['min']:.2f} - ${price_stats['max']:.2f}")
        print(f"   Average price: ${price_stats['mean']:.2f}")
        print(f"   Median price: ${price_stats['50%']:.2f}")
        print(f"   Valid prices: {df['Price_clean'].notna().sum():,} out of {len(df):,}")
    
    print(f"\n🏷️ Category Analysis:")
    if 'Category' in df.columns:
        category_counts = df['Category'].value_counts()
        print(f"   Total categories: {len(category_counts)}")
        print(f"   Top 10 categories:")
        for i, (cat, count) in enumerate(category_counts.head(10).items(), 1):
            percentage = (count / len(df)) * 100
            print(f"     {i:2d}. {cat[:40]:<40} {count:,} ({percentage:.1f}%)")
    
    print(f"\n🌍 Location Analysis:")
    if 'Seller Location' in df.columns:
        location_counts = df['Seller Location'].value_counts()
        print(f"   Total locations: {len(location_counts)}")
        print(f"   Top 10 locations:")
        for i, (loc, count) in enumerate(location_counts.head(10).items(), 1):
            percentage = (count / len(df)) * 100
            print(f"     {i:2d}. {loc[:30]:<30} {count:,} ({percentage:.1f}%)")
    
    print(f"\n📝 Text Analysis:")
    if 'Title' in df.columns:
        title_analysis = analyze_text_features(df['Title'])
        print(f"   Title analysis:")
        print(f"     Total words: {title_analysis['total_words']:,}")
        print(f"     Unique words: {title_analysis['unique_words']:,}")
        print(f"     Average length: {title_analysis['avg_text_length']:.1f} characters")
        print(f"     Top words: {[word for word, _ in title_analysis['top_words'][:5]]}")
    
    if 'Description' in df.columns:
        desc_analysis = analyze_text_features(df['Description'])
        print(f"   Description analysis:")
        print(f"     Total words: {desc_analysis['total_words']:,}")
        print(f"     Unique words: {desc_analysis['unique_words']:,}")
        print(f"     Average length: {desc_analysis['avg_text_length']:.1f} characters")
    
    # ML Opportunities Analysis
    print(f"\n🤖 MACHINE LEARNING OPPORTUNITIES:")
    print("=" * 50)
    
    opportunities = []
    
    # 1. Risk Classification
    if 'Title' in df.columns and 'Description' in df.columns:
        opportunities.append({
            'task': 'Risk Classification',
            'description': 'Classify listings as High/Medium/Low risk based on content',
            'features': ['Title', 'Description', 'Category', 'Price'],
            'target': 'Risk Level (derived from keywords)',
            'algorithm': 'Text Classification (BERT, TF-IDF + ML)',
            'data_quality': 'High - Rich text content available'
        })
    
    # 2. Price Prediction
    if 'Price' in df.columns and df['Price_clean'].notna().sum() > 1000:
        opportunities.append({
            'task': 'Price Prediction',
            'description': 'Predict listing prices based on features',
            'features': ['Title', 'Description', 'Category', 'Seller Location', 'Quantity'],
            'target': 'Price (cleaned numeric)',
            'algorithm': 'Regression (XGBoost, Random Forest)',
            'data_quality': f'Good - {df["Price_clean"].notna().sum():,} valid prices'
        })
    
    # 3. Category Classification
    if 'Category' in df.columns and len(df['Category'].value_counts()) > 5:
        opportunities.append({
            'task': 'Category Classification',
            'description': 'Automatically categorize listings based on content',
            'features': ['Title', 'Description'],
            'target': 'Category',
            'algorithm': 'Multi-class Classification (BERT, CNN)',
            'data_quality': f'Good - {len(df["Category"].value_counts())} categories'
        })
    
    # 4. Fraud Detection
    if len(df) > 1000:
        opportunities.append({
            'task': 'Fraud Detection',
            'description': 'Identify suspicious or fraudulent listings',
            'features': ['Title', 'Description', 'Price', 'Seller', 'Location patterns'],
            'target': 'Fraud Score (derived from patterns)',
            'algorithm': 'Anomaly Detection (Isolation Forest, Autoencoder)',
            'data_quality': 'Good - Large dataset for pattern detection'
        })
    
    # 5. Sentiment Analysis
    if 'Description' in df.columns:
        opportunities.append({
            'task': 'Sentiment Analysis',
            'description': 'Analyze sentiment and urgency in listings',
            'features': ['Title', 'Description'],
            'target': 'Sentiment Score (Positive/Negative/Neutral)',
            'algorithm': 'Sentiment Analysis (BERT, VADER)',
            'data_quality': 'High - Rich text descriptions available'
        })
    
    # 6. Market Trend Analysis
    if 'Price' in df.columns and 'Category' in df.columns:
        opportunities.append({
            'task': 'Market Trend Analysis',
            'description': 'Analyze price trends and market dynamics',
            'features': ['Price', 'Category', 'Location', 'Availability'],
            'target': 'Market Trends (Time series analysis)',
            'algorithm': 'Time Series Analysis (ARIMA, Prophet)',
            'data_quality': 'Moderate - Would need temporal data'
        })
    
    # Print opportunities
    for i, opp in enumerate(opportunities, 1):
        print(f"\n{i}. {opp['task']}")
        print(f"   📝 {opp['description']}")
        print(f"   🎯 Target: {opp['target']}")
        print(f"   🔧 Algorithm: {opp['algorithm']}")
        print(f"   📊 Data Quality: {opp['data_quality']}")
        print(f"   📈 Features: {', '.join(opp['features'])}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 30)
    print("1. Start with Risk Classification - highest immediate value")
    print("2. Use BERT for text-based tasks (best performance)")
    print("3. Implement data preprocessing pipeline")
    print("4. Consider ensemble methods for better accuracy")
    print("5. Use cross-validation due to imbalanced classes")
    
    print(f"\n📈 NEXT STEPS:")
    print("=" * 20)
    print("1. Data preprocessing and cleaning")
    print("2. Feature engineering (text embeddings, price normalization)")
    print("3. Train/test split with stratification")
    print("4. Model development and evaluation")
    print("5. Integration with existing dashboard")

if __name__ == "__main__":
    main() 