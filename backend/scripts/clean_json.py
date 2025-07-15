#!/usr/bin/env python3
import json
import re

def clean_json_file(input_file, output_file):
    """Clean JSON file by replacing NaN values with empty strings"""
    print(f"Cleaning {input_file}...")
    
    # Read the file as text first
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace NaN values with empty strings
    # This regex matches "NaN" that appears as a value in JSON
    cleaned_content = re.sub(r':\s*NaN\s*([,}])', r': ""\1', content)
    
    # Also handle NaN without quotes
    cleaned_content = re.sub(r':\s*NaN\s*([,}])', r': ""\1', cleaned_content)
    
    # Write the cleaned content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"Cleaned file saved as {output_file}")
    
    # Verify the JSON is valid
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ JSON is valid. Contains {len(data)} items.")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ JSON is still invalid: {e}")
        return False

if __name__ == "__main__":
    # Clean the credit card listings file
    input_file = "frontend/public/credit_card_listings_100.json"
    output_file = "frontend/public/credit_card_listings_100_clean.json"
    
    if clean_json_file(input_file, output_file):
        # Replace the original file with the cleaned version
        import shutil
        shutil.move(output_file, input_file)
        print(f"✅ Replaced {input_file} with cleaned version")
    else:
        print("❌ Failed to clean JSON file")
    
    # Clean the test data file
    input_file = "frontend/public/credit_card_test.json"
    output_file = "frontend/public/credit_card_test_clean.json"
    
    if clean_json_file(input_file, output_file):
        # Replace the original file with the cleaned version
        import shutil
        shutil.move(output_file, input_file)
        print(f"✅ Replaced {input_file} with cleaned version")
    else:
        print("❌ Failed to clean test JSON file") 