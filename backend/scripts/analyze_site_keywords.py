#!/usr/bin/env python3
"""
Analyze Site Data Using Configurable Keyword Risk Levels

This script analyzes ONLY the data/sites/ files and uses a JSON config file
(data/keyword_risk_classification.json) for keywords and risk levels.
"""

import json
import os
import re
import pandas as pd
from collections import Counter

CONFIG_JSON = 'data/keyword_risk_classification.json'

TEMPLATE = {
    "critical": {
        "fullz": "Complete identity package (SSN, DOB, address) - enables identity theft",
        "dumps": "Magnetic stripe data for card cloning",
        "cloned": "Duplicated physical cards - enables ATM withdrawals",
        "cashout": "Converting stolen data to cash - active fraud process"
    },
    "high": {
        "cvv": "Card verification value - card fraud",
        "ssn": "Social Security Number - identity theft",
        "paypal": "PayPal accounts - payment fraud",
        "visa": "Visa cards - payment fraud"
    },
    "medium": {
        "card": "Credit/debit cards - payment method",
        "bank": "Banking - financial institution",
        "money": "Currency - financial",
        "shop": "Shopping - commerce"
    },
    "low": {
        "info": "Information - data",
        "data": "Information - data",
        "file": "File - data",
        "tool": "Tool - utility"
    }
}

def ensure_config_json():
    """Ensure the JSON config file exists, create template if not."""
    if not os.path.exists(CONFIG_JSON):
        print(f"Config file {CONFIG_JSON} not found. Creating template.")
        os.makedirs(os.path.dirname(CONFIG_JSON), exist_ok=True)
        with open(CONFIG_JSON, 'w') as f:
            json.dump(TEMPLATE, f, indent=2)
        print(f"Please edit {CONFIG_JSON} to add your keywords and reasons.")
        return False
    return True

def load_keywords():
    """Load keywords from JSON config file."""
    with open(CONFIG_JSON, 'r') as f:
        config = json.load(f)
    
    # Flatten the structure for easier processing
    keywords = []
    risk_map = {}
    reason_map = {}
    
    for risk_level, keyword_dict in config.items():
        for keyword, reason in keyword_dict.items():
            keywords.append(keyword.lower())
            risk_map[keyword.lower()] = risk_level
            reason_map[keyword.lower()] = reason
    
    return keywords, risk_map, reason_map

def analyze_site_keywords():
    """Analyze the actual site data to show where keywords came from."""
    if not ensure_config_json():
        return
    
    keywords, risk_map, reason_map = load_keywords()
    
    sites_dir = 'data/sites'
    all_names = []
    all_descs = []
    
    # Gather all listing names and descriptions
    for site_name in os.listdir(sites_dir):
        site_path = os.path.join(sites_dir, site_name)
        if os.path.isdir(site_path):
            for fname in os.listdir(site_path):
                if fname.endswith('.json'):
                    with open(os.path.join(site_path, fname), 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            for item in data:
                                if 'name' in item:
                                    all_names.append(item['name'])
                                if 'description' in item:
                                    all_descs.append(item['description'])
                        except Exception as e:
                            print(f"Error reading {fname}: {e}")
    
    all_text = ' '.join(all_names + all_descs).lower()
    
    # Count keyword occurrences
    keyword_counts = {}
    for kw in keywords:
        count = len(re.findall(r'\b' + re.escape(kw) + r'\b', all_text))
        if count > 0:
            keyword_counts[kw] = count
    
    print("\n=== KEYWORD OCCURRENCES IN SITE DATA ===")
    
    # Group by risk level
    for risk_level in ['critical', 'high', 'medium', 'low']:
        risk_keywords = {kw: count for kw, count in keyword_counts.items() 
                        if risk_map.get(kw) == risk_level}
        
        if risk_keywords:
            print(f"\n{risk_level.upper()} RISK:")
            for kw, count in sorted(risk_keywords.items(), key=lambda x: -x[1]):
                print(f"  {kw}: {count} occurrences - {reason_map[kw]}")
    
    print(f"\nEdit the JSON at {CONFIG_JSON} to add/remove keywords and reasons.")
    print("Structure: risk_level -> keyword -> reason")

if __name__ == "__main__":
    analyze_site_keywords() 