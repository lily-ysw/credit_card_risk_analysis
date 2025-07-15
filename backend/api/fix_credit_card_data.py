import pandas as pd
import json
import os
import re
from sklearn.model_selection import train_test_split

# Credit card categories and their base risk levels
CREDIT_CARD_CATEGORIES = {
    'Cards and CVV': 'Critical',
    'Carding': 'Critical', 
    'SSN': 'Critical',
    'Dump': 'Critical',
    'Drop Bank': 'High',
    'Physical Drop': 'High'
}

def load_keywords_from_files():
    """Load keywords from JSON files"""
    keywords = {'critical': [], 'high': [], 'medium': [], 'low': []}
    risk_files = {
        'critical': 'data/keywords/keywords_critical.json',
        'high': 'data/keywords/keywords_high.json', 
        'medium': 'data/keywords/keywords_medium.json',
        'low': 'data/keywords/keywords_low.json'
    }
    
    for risk_level, file_path in risk_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    keywords[risk_level] = json.load(f)
                print(f"Loaded {len(keywords[risk_level])} {risk_level} risk keywords")
            except Exception as e:
                print(f"Error loading {risk_level} keywords: {e}")
        else:
            print(f"Warning: {file_path} not found")
    
    return keywords

def create_credit_card_risk_labels(title, description, category, price, keywords):
    """Create risk labels specifically for credit card related listings using keyword files"""
    text = f"{title} {description} {category}".lower()
    
    # Check if it's a known credit card category first
    if category in CREDIT_CARD_CATEGORIES:
        base_risk = CREDIT_CARD_CATEGORIES[category]
        
        # Check for critical risk keywords first
        for keyword in keywords['critical']:
            if keyword.lower() in text:
                return 'Critical'
        
        # Check for high risk keywords
        for keyword in keywords['high']:
            if keyword.lower() in text:
                return 'High' if base_risk == 'Medium' else base_risk
        
        return base_risk
    
    # For non-credit card categories, use keyword analysis
    # Check for critical risk keywords first
    for keyword in keywords['critical']:
        if keyword.lower() in text:
            return 'Critical'
    
    # Check for high risk keywords
    for keyword in keywords['high']:
        if keyword.lower() in text:
            return 'High'
    
    # Check for medium risk keywords
    for keyword in keywords['medium']:
        if keyword.lower() in text:
            return 'Medium'
    
    # Check for low risk keywords
    for keyword in keywords['low']:
        if keyword.lower() in text:
            return 'Low'
    
    return 'Low'

def clean_price(price_str):
    """Clean price strings to extract numeric values"""
    if not price_str:
        return 0
    
    price_str = str(price_str).upper()
    price_str = re.sub(r'[^\d.,]', '', price_str)
    
    if ',' in price_str and '.' in price_str:
        price_str = price_str.replace(',', '')
    elif ',' in price_str:
        parts = price_str.split(',')
        if len(parts[-1]) == 3:
            price_str = price_str.replace(',', '')
        else:
            price_str = price_str.replace(',', '.')
    
    try:
        return float(price_str)
    except:
        return 0

def extract_credit_card_features(title, description, category, price, location):
    """Extract features specifically for credit card risk analysis"""
    # Combine text features
    text_features = f"{title} {description} {category}"
    
    # Price feature (normalized)
    price_feature = clean_price(price)
    
    # Category-based features
    is_credit_card_category = 1 if category in CREDIT_CARD_CATEGORIES else 0
    category_risk_score = {
        'Cards and CVV': 4,
        'Carding': 4,
        'SSN': 4,
        'Dump': 4,
        'Drop Bank': 3,
        'Physical Drop': 3
    }.get(category, 1)
    
    # Location feature
    location_worldwide = 1 if 'worldwide' in str(location).lower() else 0
    
    # Text-based risk indicators
    text_lower = text_features.lower()
    has_cvv = 1 if 'cvv' in text_lower else 0
    has_pin = 1 if 'pin' in text_lower else 0
    has_fullz = 1 if 'fullz' in text_lower else 0
    has_dumps = 1 if 'dumps' in text_lower else 0
    has_track = 1 if any(x in text_lower for x in ['track1', 'track2']) else 0
    
    return {
        'text': text_features,
        'price': price_feature,
        'is_credit_card_category': is_credit_card_category,
        'category_risk_score': category_risk_score,
        'location_worldwide': location_worldwide,
        'has_cvv': has_cvv,
        'has_pin': has_pin,
        'has_fullz': has_fullz,
        'has_dumps': has_dumps,
        'has_track': has_track
    }

def fix_credit_card_data():
    """Fix the credit card data by regenerating with proper risk levels"""
    print("Loading original credit card data...")
    
    # Load the original CSV
    try:
        df = pd.read_csv('data/credit_card_listings.csv', encoding='utf-8', low_memory=False)
        print(f"Loaded {len(df)} rows from original CSV")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False
    
    # Load keywords
    keywords = load_keywords_from_files()
    
    print("Calculating proper risk levels...")
    
    # Calculate proper risk levels for each row
    risk_levels = []
    features_list = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(df)}")
        
        # Calculate risk level
        risk_level = create_credit_card_risk_labels(
            row['Title'], row['Description'], row['Category'], row['Price'], keywords
        )
        risk_levels.append(risk_level)
        
        # Extract features
        features = extract_credit_card_features(
            row['Title'], row['Description'], row['Category'], row['Price'], row['Seller Location']
        )
        features_list.append(features)
    
    # Add risk levels and features to dataframe
    df['risk_level'] = risk_levels
    df['price_numeric'] = [f['price'] for f in features_list]
    df['text_length'] = [len(f['text']) for f in features_list]
    df['has_cvv'] = [f['has_cvv'] for f in features_list]
    df['has_pin'] = [f['has_pin'] for f in features_list]
    df['has_fullz'] = [f['has_fullz'] for f in features_list]
    df['has_dumps'] = [f['has_dumps'] for f in features_list]
    
    # Show risk level distribution
    print("\nRisk level distribution:")
    print(df['risk_level'].value_counts())
    
    # Split into train and test
    print("\nSplitting into train and test sets...")
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['risk_level'])
    
    # Save the fixed datasets
    train_df.to_csv('data/credit_card_train.csv', index=False)
    test_df.to_csv('data/credit_card_test.csv', index=False)
    
    print(f"\nSaved {len(train_df)} training samples to data/credit_card_train.csv")
    print(f"Saved {len(test_df)} test samples to data/credit_card_test.csv")
    
    return True

if __name__ == "__main__":
    fix_credit_card_data() 