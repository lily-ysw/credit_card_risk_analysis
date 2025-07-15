# Data Directory

This directory contains all the data files for the Onion Market Dashboard project. It serves as the central repository for raw, processed, and analysis-ready data used throughout the backend and frontend systems.

## Directory Structure

```
data/
├── dark_market_output_v2.csv              # Main CSV dataset (80,178 records)
├── credit_card_listings.csv               # Filtered credit card-related listings
├── credit_card_train.csv                  # Labeled training data for ML (raw listings + risk_level)
├── credit_card_test.csv                   # Labeled test data for ML (raw listings + risk_level)
├── onion_list_page.json                   # Onion site listings (JSON)
├── ml_results.json                        # ML model performance metrics
└── sites/                                 # Individual site data (HTML & JSON)
    └── ...
```

## Data Files Overview

### Main Dataset
- **dark_market_output_v2.csv**: The primary dataset containing over 80,000 marketplace listings scraped from multiple onion sites. Includes fields such as Title, Price, Seller, Location, Category, and Description. Used for general analysis and as the source for downstream filtering.

### Credit Card Listings
- **credit_card_listings.csv**: A filtered subset of the main dataset, containing only listings related to credit cards. Extracted using keyword and category filtering. Used for risk analysis and ML labeling.

### ML Training & Test Data
- **credit_card_train.csv**: A labeled dataset of credit card-related listings. Each row contains:
  - `title`: Listing title
  - `description`: Listing description
  - `category`: Marketplace category
  - `price`: Price (as string, may include currency)
  - `location`: Seller or item location
  - `risk_level`: Human- or rule-assigned risk label (e.g., Low, Medium, High, Critical)

  This file is **not** a preprocessed ML feature matrix. It is used as the input for feature extraction, vectorization, and model training in the ML pipeline.

- **credit_card_test.csv**: Same structure as `credit_card_train.csv`, but used for model evaluation and reporting.

### Site-Specific Data
- **sites/**: Contains subdirectories for each onion market site. Each subdirectory includes:
  - `*_list_page.html`: Raw HTML scraped from the site
  - `*_list_page.json`: Parsed and normalized JSON data

### Analysis Results
- **ml_results.json**: Stores machine learning model performance metrics, such as accuracy, classification report, confusion matrix, and feature importances. Used by the frontend dashboard for analytics and reporting.

## Data Pipeline & Usage

1. **Raw Data Collection**: HTML pages are scraped from onion sites and stored in `sites/`.
2. **Parsing & Normalization**: HTML is converted to structured JSON, then standardized across sites.
3. **Filtering**: Credit card-related listings are extracted into `credit_card_listings.csv`.
4. **Labeling**: Listings are labeled with risk levels (manual or rule-based) and split into `credit_card_train.csv` and `credit_card_test.csv` for ML.
5. **ML Pipeline**: Feature extraction, vectorization, model training, and evaluation are performed using the labeled CSVs as input.
6. **Results Storage**: Model performance metrics are saved in `ml_results.json` for frontend display and reporting.

## How to Use These Data Files

### For Data Processing & ML
```bash
# Extract credit card data from the main dataset
python3 backend/scripts/ml/extract_credit_cards_v2.py

# Split into train/test sets
python3 backend/scripts/ml/split_credit_card_data.py

# Analyze the dataset
python3 backend/scripts/analysis/analyze_csv.py
```

### For ML API
```bash
# Start the ML API (uses credit_card_train.csv as input)
python3 backend/api/ml_api.py

# Or use the unified startup script (starts both backend and frontend)
./start_with_ml.sh
```
- The ML API supports downloading the trained model, vectorizer, and a PDF report (with model info, classification report, confusion matrix, and top feature importances).
- See backend/api/README.md for API details and endpoints.

### For Frontend
- The frontend loads data and analytics via the backend API. No manual copying of data files is required.
- Use the unified startup script to launch both backend and frontend:
```bash
./start_with_ml.sh
```

## Data Quality Notes
- The main dataset contains over 80,000 records with varying data quality and completeness.
- Credit card filtering identifies a subset of relevant listings for risk analysis.
- Labeled training and test sets are used as input for ML, not as preprocessed feature matrices.
- Test set provides unbiased evaluation of model performance.

## Modern Dashboard & Reporting
- The dashboard provides a modern, visually appealing interface for exploring risk analysis results, model performance, and feature importances.
- Export and reporting features allow you to download the trained model, vectorizer, and a comprehensive PDF report directly from the dashboard.
- All data and scripts are organized under the `backend/` directory for clarity and maintainability. 