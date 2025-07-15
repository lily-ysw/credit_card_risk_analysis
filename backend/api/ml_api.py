#!/usr/bin/env python3
"""
Credit Card Focused ML API for risk classification
"""

import csv
import re
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
from flask import send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import json
from datetime import datetime
import math
import time
from flask import Flask, request, jsonify
from flask import send_file
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
import zipstream
from flask import Response
from datetime import timedelta

app = Flask(__name__)
CORS(app)

# Global variables for model and vectorizer
model = None
vectorizer = None
model_accuracy = 0.0
training_samples = 0
last_trained = None
cached_keywords = None
conf_matrix = None
class_report = None
top_features = None

# Credit card specific categories and their risk levels
CREDIT_CARD_CATEGORIES = {
    'Cards and CVV': 'Critical',      # Direct credit card data
    'Carding': 'Critical',            # Credit card fraud activities
    'SSN': 'Critical',                # Identity theft
    'Dump': 'Critical',               # Magnetic stripe data
    'Drop Bank': 'High',              # Money laundering
    'Physical Drop': 'High'           # Physical fraud operations
}

def load_keywords_from_files():
    """Load keywords from the JSON files you created"""
    keywords = {
        'critical': {},
        'high': {},
        'medium': {},
        'low': {}
    }
    
    # Load keywords from separate files
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
                print(f"Loaded {len(keywords[risk_level])} {risk_level} risk keywords")
            except Exception as e:
                print(f"Error loading {risk_level} keywords: {e}")
        else:
            print(f"Warning: {file_path} not found")
    
    return keywords

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

def create_credit_card_risk_labels(title, description, category, price):
    """Create risk labels based on content analysis rather than category"""
    global cached_keywords
    
    # Focus on the actual content, not the category
    text = f"{title} {description}".lower()
    
    # Load keywords from your JSON files (cached)
    if cached_keywords is None:
        cached_keywords = load_keywords_from_files()
    
    keywords = cached_keywords
    
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

def load_and_preprocess_credit_card_data():
    """Load and preprocess credit card specific data with random split each time"""
    # Load the full JSON dataset
    with open('../data/credit_card_listings_full.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Combine text features
    df['combined_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['category'].fillna('')
    # Use existing risk_level as label
    if 'risk_level' not in df.columns:
        raise ValueError('risk_level column missing in data')
    # Random split each time
    random_state = int(time.time()) % 100000
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=random_state, stratify=df['risk_level'])
    print(f"Random split with random_state={random_state}: Train {len(df_train)}, Test {len(df_test)}")
    return df_train, df_test

def train_credit_card_model():
    """Train the ML model specifically for credit card risk classification"""
    global model, vectorizer, model_accuracy, training_samples, last_trained, conf_matrix, class_report, top_features
    
    print("Training Credit Card ML model...")
    
    # Load and preprocess data
    df_train, df_test = load_and_preprocess_credit_card_data()
    if df_train is None or df_test is None:
        return False, "Failed to load data"
    
    # Prepare features and labels - use proper train/test split
    X_train = df_train['combined_text']
    y_train = df_train['risk_level']
    X_test = df_test['combined_text']
    y_test = df_test['risk_level']
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Create TF-IDF features with credit card specific parameters
    vectorizer = TfidfVectorizer(
        max_features=1500, 
        stop_words='english', 
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Random Forest with credit card specific parameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_tfidf, y_train)
    
    # Save model and vectorizer to disk
    with open('../data/credit_card_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('../data/credit_card_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Evaluate model on proper test set
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store model info
    model_accuracy = accuracy
    training_samples = len(df_train)
    last_trained = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Store confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['Critical', 'High', 'Medium', 'Low'])
    class_report = classification_report(y_test, y_pred, labels=['Critical', 'High', 'Medium', 'Low'], output_dict=True)
    
    # Get feature importance and top features
    feature_importance = model.feature_importances_
    feature_names = vectorizer.get_feature_names_out()
    feature_importance_pairs = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)
    top_features = [{'feature': name, 'importance': float(importance)} for name, importance in feature_importance_pairs[:20]]
    
    print(f"Model trained on {training_samples} samples")
    print(f"Model accuracy on held-out test set: {accuracy:.3f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return True, "Model trained successfully"

# Initialize model on startup
train_credit_card_model()

def sanitize_for_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    else:
        return obj

@app.route('/predict', methods=['POST'])
def predict():
    """Predict risk level for a single listing"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Transform text using the trained vectorizer
        text_features = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_features)[0]
        probabilities = model.predict_proba(text_features)[0]
        
        # Get feature importance for this prediction
        feature_names = vectorizer.get_feature_names_out()
        feature_importance = model.feature_importances_
        
        # Get top contributing features
        top_features = []
        if hasattr(model, 'feature_importances_'):
            indices = np.argsort(feature_importance)[::-1][:10]
            for idx in indices:
                if idx < len(feature_names):
                    top_features.append({
                        'feature': feature_names[idx],
                        'importance': float(feature_importance[idx])
                    })
        
        # Create probability mapping
        classes = model.classes_
        prob_mapping = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        
        response = {
            'prediction': prediction,
            'probabilities': prob_mapping,
            'confidence': float(max(probabilities)),
            'top_features': top_features[:5],
            'model_accuracy': model_accuracy,
            'training_samples': training_samples
        }
        response = sanitize_for_json(response)
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict risk levels for multiple listings"""
    try:
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Data must be a list of items'}), 400
        
        if len(data) == 0:
            return jsonify({'error': 'Empty data list provided'}), 400
        
        # Process each item in the batch
        results = []
        for item in data:
            try:
                # Extract text features from the item
                title = str(item.get('title', ''))
                description = str(item.get('description', ''))
                category = str(item.get('category', ''))
                
                # Combine text features
                text = f"{title} {description} {category}".strip()
                
                if not text:
                    # If no text, use a default prediction
                    results.append({
                        **item,
                        'mlPrediction': 'Low',
                        'ml_confidence': 0.0
                    })
                    continue
                
                # Transform text using the trained vectorizer
                text_features = vectorizer.transform([text])
                
                # Make prediction
                prediction = model.predict(text_features)[0]
                probabilities = model.predict_proba(text_features)[0]
                confidence = float(max(probabilities))
                
                # Add ML prediction to the item
                results.append({
                    **item,
                    'mlPrediction': prediction,
                    'ml_confidence': confidence
                })
                
            except Exception as e:
                # If prediction fails for an item, use default
                results.append({
                    **item,
                    'mlPrediction': 'Low',
                    'ml_confidence': 0.0
                })
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model"""
    try:
        success, message = train_credit_card_model()
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'accuracy': model_accuracy,
                'samples': training_samples,
                'last_trained': last_trained
            })
        else:
            return jsonify({'success': False, 'error': message}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    global model, model_accuracy, training_samples, last_trained, top_features
    if model is None:
        return jsonify({'error': 'Model not trained'}), 400
    return jsonify({
        'accuracy': model_accuracy,
        'training_samples': training_samples,
        'last_trained': last_trained,
        'model_type': 'Random Forest',
        'features': 'TF-IDF + Credit Card Specific Features',
        'confusion_matrix': conf_matrix.tolist() if conf_matrix is not None else None,
        'classification_report': class_report if class_report is not None else None,
        'credit_card_categories': CREDIT_CARD_CATEGORIES,
        'top_features': top_features if top_features is not None else []
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/download_model', methods=['GET'])
def download_model():
    """Download the trained model file"""
    model_path = '../data/credit_card_model.pkl'
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 404
    return send_file(model_path, as_attachment=True)

@app.route('/download_vectorizer', methods=['GET'])
def download_vectorizer():
    """Download the trained vectorizer file"""
    vectorizer_path = '../data/credit_card_vectorizer.pkl'
    if not os.path.exists(vectorizer_path):
        return jsonify({'error': 'Vectorizer file not found'}), 404
    return send_file(vectorizer_path, as_attachment=True)

@app.route('/download_report_pdf', methods=['GET'])
def download_report_pdf():
    """Generate and download a PDF report with ML Model Information, Classification Report, and Confusion Matrix"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # ML Model Information
    info_data = [
        ["Model Type", 'Random Forest'],
        ["Accuracy", f"{model_accuracy:.2%}" if model_accuracy else "N/A"],
        ["Training Samples", str(training_samples)],
        ["Last Trained", last_trained or "N/A"]
    ]
    info_table = Table(info_data, hAlign='LEFT')
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    elements.append(KeepTogether([
        Paragraph("<b>ML Model Information</b>", styles['Heading2']),
        info_table
    ]))
    elements.append(Spacer(1, 16))

    # Top Feature Importances
    if top_features and len(top_features) > 0:
        tf_header = ['Feature', 'Importance']
        tf_rows = [[f["feature"], f'{f["importance"]:.4f}'] for f in top_features[:10]]
        tf_table = Table([tf_header] + tf_rows, hAlign='LEFT')
        tf_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightyellow),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        elements.append(KeepTogether([
            Paragraph("<b>Top Feature Importances</b>", styles['Heading2']),
            tf_table
        ]))
        elements.append(Spacer(1, 16))

    # Classification Report
    if class_report:
        metrics = ['precision', 'recall', 'f1-score', 'support']
        header = ['Class'] + [m.title() for m in metrics]
        rows = []
        for k, v in class_report.items():
            if k in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            row = [k] + [f"{v[m]:.2f}" if isinstance(v[m], float) else str(v[m]) for m in metrics]
            rows.append(row)
        if 'macro avg' in class_report:
            v = class_report['macro avg']
            rows.append(['Macro Avg'] + [f"{v[m]:.2f}" if isinstance(v[m], float) else str(v[m]) for m in metrics])
        if 'weighted avg' in class_report:
            v = class_report['weighted avg']
            rows.append(['Weighted Avg'] + [f"{v[m]:.2f}" if isinstance(v[m], float) else str(v[m]) for m in metrics])
        if 'accuracy' in class_report:
            rows.append(['Accuracy', f"{class_report['accuracy']:.2f}", '', '', ''])
        table_data = [header] + rows
        cr_table = Table(table_data, hAlign='LEFT')
        cr_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        elements.append(KeepTogether([
            Paragraph("<b>Classification Report</b>", styles['Heading2']),
            cr_table
        ]))
    else:
        elements.append(KeepTogether([
            Paragraph("<b>Classification Report</b>", styles['Heading2']),
            Paragraph("No classification report available.", styles['Normal'])
        ]))
    elements.append(Spacer(1, 16))

    # Confusion Matrix
    if conf_matrix is not None:
        labels = ['Critical', 'High', 'Medium', 'Low']
        header = ['Actual \\ Predicted'] + labels
        rows = []
        for i, label in enumerate(labels):
            row = [label] + [str(conf_matrix[i][j]) for j in range(len(labels))]
            rows.append(row)
        table_data = [header] + rows
        cm_table = Table(table_data, hAlign='LEFT')
        cm_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgreen),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        elements.append(KeepTogether([
            Paragraph("<b>Confusion Matrix</b>", styles['Heading2']),
            cm_table
        ]))
    else:
        elements.append(KeepTogether([
            Paragraph("<b>Confusion Matrix</b>", styles['Heading2']),
            Paragraph("No confusion matrix available.", styles['Normal'])
        ]))

    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='ml_model_report.pdf', mimetype='application/pdf')

@app.route('/download_all_zip', methods=['GET'])
def download_all_zip():
    """Download Model, Vectorizer, and PDF Report as a zip file"""
    model_path = '../data/credit_card_model.pkl'
    vectorizer_path = '../data/credit_card_vectorizer.pkl'
    # Generate PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    # ML Model Information
    info_data = [
        ["Model Type", 'Random Forest'],
        ["Accuracy", f"{model_accuracy:.2%}" if model_accuracy else "N/A"],
        ["Training Samples", str(training_samples)],
        ["Last Trained", last_trained or "N/A"]
    ]
    info_table = Table(info_data, hAlign='LEFT')
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    elements.append(KeepTogether([
        Paragraph("<b>ML Model Information</b>", styles['Heading2']),
        info_table
    ]))
    elements.append(Spacer(1, 16))

    # Top Feature Importances
    if top_features and len(top_features) > 0:
        tf_header = ['Feature', 'Importance']
        tf_rows = [[f["feature"], f'{f["importance"]:.4f}'] for f in top_features[:10]]
        tf_table = Table([tf_header] + tf_rows, hAlign='LEFT')
        tf_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightyellow),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        elements.append(KeepTogether([
            Paragraph("<b>Top Feature Importances</b>", styles['Heading2']),
            tf_table
        ]))
        elements.append(Spacer(1, 16))

    # Classification Report
    if class_report:
        metrics = ['precision', 'recall', 'f1-score', 'support']
        header = ['Class'] + [m.title() for m in metrics]
        rows = []
        for k, v in class_report.items():
            if k in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            row = [k] + [f"{v[m]:.2f}" if isinstance(v[m], float) else str(v[m]) for m in metrics]
            rows.append(row)
        if 'macro avg' in class_report:
            v = class_report['macro avg']
            rows.append(['Macro Avg'] + [f"{v[m]:.2f}" if isinstance(v[m], float) else str(v[m]) for m in metrics])
        if 'weighted avg' in class_report:
            v = class_report['weighted avg']
            rows.append(['Weighted Avg'] + [f"{v[m]:.2f}" if isinstance(v[m], float) else str(v[m]) for m in metrics])
        if 'accuracy' in class_report:
            rows.append(['Accuracy', f"{class_report['accuracy']:.2f}", '', '', ''])
        table_data = [header] + rows
        cr_table = Table(table_data, hAlign='LEFT')
        cr_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        elements.append(KeepTogether([
            Paragraph("<b>Classification Report</b>", styles['Heading2']),
            cr_table
        ]))
    else:
        elements.append(KeepTogether([
            Paragraph("<b>Classification Report</b>", styles['Heading2']),
            Paragraph("No classification report available.", styles['Normal'])
        ]))
    elements.append(Spacer(1, 16))

    # Confusion Matrix
    if conf_matrix is not None:
        labels = ['Critical', 'High', 'Medium', 'Low']
        header = ['Actual \\ Predicted'] + labels
        rows = []
        for i, label in enumerate(labels):
            row = [label] + [str(conf_matrix[i][j]) for j in range(len(labels))]
            rows.append(row)
        table_data = [header] + rows
        cm_table = Table(table_data, hAlign='LEFT')
        cm_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgreen),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        elements.append(KeepTogether([
            Paragraph("<b>Confusion Matrix</b>", styles['Heading2']),
            cm_table
        ]))
    else:
        elements.append(KeepTogether([
            Paragraph("<b>Confusion Matrix</b>", styles['Heading2']),
            Paragraph("No confusion matrix available.", styles['Normal'])
        ]))
    doc.build(elements)
    buffer.seek(0)
    # Prepare zipstream
    z = zipstream.ZipFile(mode='w', compression=zipstream.ZIP_DEFLATED)
    if os.path.exists(model_path):
        z.write(model_path, arcname='model.pkl')
    if os.path.exists(vectorizer_path):
        z.write(vectorizer_path, arcname='vectorizer.pkl')
    z.write_iter('ml_model_report.pdf', [buffer.getvalue()])
    # Format zip filename with last_trained datetime (local time, no +8 adjustment)
    if last_trained:
        try:
            dt_obj = datetime.strptime(last_trained, '%Y-%m-%d %H:%M:%S')
            dt_str = dt_obj.strftime('%Y%m%d_%H%M%S')
            zip_name = f'ml_model_bundle_{dt_str}.zip'
        except Exception:
            from datetime import datetime as dt
            now = dt.now()
            zip_name = f'ml_model_bundle_{now.strftime("%Y%m%d_%H%M%S")}.zip'
    else:
        from datetime import datetime as dt
        now = dt.now()
        zip_name = f'ml_model_bundle_{now.strftime("%Y%m%d_%H%M%S")}.zip'
    return Response(z, mimetype='application/zip', headers={
        'Content-Disposition': f'attachment; filename={zip_name}'
    })

if __name__ == '__main__':
    print("Starting Credit Card ML API server on http://localhost:9000")
    app.run(host='0.0.0.0', port=9000, debug=True) 