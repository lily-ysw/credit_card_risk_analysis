# Backend Scripts Directory

## 🐍 Python Virtual Environment (venv) & Startup

The backend venv is **automatically created and managed** when you run the main startup script:

```bash
./start_with_ml.sh
```

- This script will:
  - Kill any process using port 3000 (frontend) or 9000 (backend API) to prevent address-in-use errors.
  - Create `.venv` in `backend/` if it doesn't exist.
  - Install all dependencies from `requirements.txt`.
  - Run the backend API using the venv.
- **You do not need to manually set up or activate the venv** unless you want to run backend scripts independently.
- If you do want to use the venv manually:
  ```bash
  cd backend
  source .venv/bin/activate
  # ... run your scripts ...
  deactivate
  ```

This directory contains all backend scripts organized by function.

## 📁 Directory Structure

```
backend/scripts/
├── README.md              # This file
├── crawling/              # Web crawling scripts
│   ├── crawl_onion.py     # Selenium-based onion site crawler
│   └── selenium_tor.py    # Tor browser automation utilities
├── analysis/              # Data analysis scripts
│   └── analyze_csv.py     # Advanced CSV analysis with pandas/matplotlib
├── ml/                    # Machine learning scripts
│   ├── extract_credit_cards_v2.py
│   ├── split_credit_card_data.py
│   └── ...
├── analyze_credit_cards.py
├── create_credit_card_json.py
├── extract_credit_card_listings.py
├── fix_training_labels.py
├── test_model_vectorizer.py
├── clean_json.py
├── discover_site_keywords.py
├── explain_risk_levels.py
├── analyze_site_keywords.py
└── find_low_risk.py
```

## 🕷️ Crawling Scripts

### `crawling/crawl_onion.py`
- **Purpose**: Crawl .onion sites using Selenium routed through Tor
- **Features**: 
  - Tor Browser integration
  - Headless/headed mode options
  - SOCKS proxy configuration
- **Usage**: `python3 backend/scripts/crawling/crawl_onion.py`

### `crawling/selenium_tor.py`
- **Purpose**: Tor browser automation utilities
- **Features**:
  - Firefox profile configuration
  - Proxy settings
  - WebDriver management

## 📊 Analysis Scripts

### `analysis/analyze_csv.py`
- **Purpose**: Advanced CSV data analysis for ML opportunities
- **Features**:
  - Pandas-based data analysis
  - Statistical summaries
  - Text feature analysis
  - ML opportunity identification
  - Visualization capabilities
- **Usage**: `python3 backend/scripts/analysis/analyze_csv.py`
- **Dependencies**: pandas, numpy, matplotlib, seaborn

## 🤖 Machine Learning Scripts

### `ml/extract_credit_cards_v2.py`, `ml/split_credit_card_data.py`, etc.
- **Purpose**: ML data preparation, training, and evaluation
- **Features**:
  - Data extraction and splitting
  - Model training and evaluation
  - Feature engineering
- **Usage**: `python3 backend/scripts/ml/extract_credit_cards_v2.py`

## 🚀 Running Scripts

### From Project Root
```bash
# Run analysis
python3 backend/scripts/analysis/analyze_csv.py

# Run ML data prep
python3 backend/scripts/ml/extract_credit_cards_v2.py

# Run crawler
python3 backend/scripts/crawling/crawl_onion.py
```

### Using Startup Scripts
```bash
# Start full stack (backend + frontend)
./start_with_ml.sh
```

## 📋 Dependencies

### Python Packages
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `flask` - Web API framework (for backend/api/ml_api.py)
- `selenium` - Web automation
- `beautifulsoup4` - HTML parsing
- `matplotlib` - Data visualization
- `seaborn` - Statistical visualization

### System Requirements
- Python 3.8+
- Tor Browser (for crawling)
- Firefox (for Selenium)
- Node.js (for frontend)

## 🔧 Configuration

### Data Paths
All scripts are configured to read from the `backend/data/` directory:
- CSV data: `../data/dark_market_output_v2.csv`
- Site data: `../data/sites/`

### Port Configuration
- ML API: Port 9000 (configurable in backend/api/ml_api.py)
- Frontend: Port 3000 (Next.js default)

## 🛠️ Development

### Adding New Scripts
1. Create script in appropriate subdirectory
2. Update this README with documentation
3. Ensure proper path references
4. Test from project root

### Script Guidelines
- Use relative paths from project root
- Include proper error handling
- Add logging for debugging
- Document dependencies
- Follow Python PEP 8 style 