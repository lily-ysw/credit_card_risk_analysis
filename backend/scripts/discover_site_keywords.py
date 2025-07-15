#!/usr/bin/env python3
"""
Discover Keywords from Site Data (All Keys)

This script analyzes all JSON files in data/sites/*/*.json to find keywords,
count their occurrences, and rank them by frequency. It extracts all string values
from all keys, recursively.
"""

import json
import os
import re
from collections import Counter

def load_keywords_from_files():
    """Load keywords from separate JSON files for each risk level."""
    keywords = {
        'critical': {},
        'high': {},
        'medium': {},
        'low': {}
    }
    
    # Load keywords from separate files
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
                print(f"Loaded {len(keywords[risk_level])} {risk_level} risk keywords from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"Warning: {file_path} not found")
    
    return keywords

def extract_all_strings_from_json(obj, strings=None):
    """Recursively extract all string values from JSON object."""
    if strings is None:
        strings = []
    
    if isinstance(obj, dict):
        for value in obj.values():
            extract_all_strings_from_json(value, strings)
    elif isinstance(obj, list):
        for item in obj:
            extract_all_strings_from_json(item, strings)
    elif isinstance(obj, str):
        strings.append(obj.lower())
    
    return strings

def analyze_keywords_in_sites():
    """Analyze keyword occurrences in site JSON files."""
    keywords = load_keywords_from_files()
    
    # Get all site directories
    sites_dir = 'data/sites'
    if not os.path.exists(sites_dir):
        print(f"Sites directory {sites_dir} not found")
        return
    
    site_dirs = [d for d in os.listdir(sites_dir) if os.path.isdir(os.path.join(sites_dir, d))]
    
    all_occurrences = {}
    site_analysis = {}
    
    for site_dir in site_dirs:
        site_path = os.path.join(sites_dir, site_dir)
        json_files = [f for f in os.listdir(site_path) if f.endswith('.json')]
        
        site_occurrences = {}
        site_strings = []
        
        for json_file in json_files:
            file_path = os.path.join(site_path, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract all strings from the JSON
                strings = extract_all_strings_from_json(data)
                site_strings.extend(strings)
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Analyze keywords in this site
        for risk_level, risk_keywords in keywords.items():
            for keyword, description in risk_keywords.items():
                # Count occurrences of this keyword in the site's strings
                count = sum(1 for s in site_strings if keyword.lower() in s)
                if count > 0:
                    if keyword not in site_occurrences:
                        site_occurrences[keyword] = {'count': 0, 'risk_level': risk_level, 'description': description}
                    site_occurrences[keyword]['count'] += count
                    
                    # Add to global occurrences
                    if keyword not in all_occurrences:
                        all_occurrences[keyword] = {'count': 0, 'risk_level': risk_level, 'description': description}
                    all_occurrences[keyword]['count'] += count
        
        site_analysis[site_dir] = site_occurrences
    
    # Save results
    with open('data/keyword_occurrences.json', 'w', encoding='utf-8') as f:
        json.dump(all_occurrences, f, indent=2, ensure_ascii=False)
    
    with open('data/site_keyword_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(site_analysis, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\nAnalysis complete!")
    print(f"Found {len(all_occurrences)} unique keywords across all sites")
    print(f"Results saved to:")
    print(f"  - data/keyword_occurrences.json (global occurrences)")
    print(f"  - data/site_keyword_analysis.json (per-site analysis)")
    
    # Print top keywords by risk level
    for risk_level in ['critical', 'high', 'medium', 'low']:
        risk_keywords = {k: v for k, v in all_occurrences.items() if v['risk_level'] == risk_level}
        if risk_keywords:
            print(f"\nTop {risk_level.upper()} risk keywords:")
            sorted_keywords = sorted(risk_keywords.items(), key=lambda x: x[1]['count'], reverse=True)
            for keyword, info in sorted_keywords[:10]:
                print(f"  {keyword}: {info['count']} occurrences - {info['description']}")

if __name__ == "__main__":
    analyze_keywords_in_sites() 