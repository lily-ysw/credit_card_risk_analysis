# Credit Card Risk Analysis

A comprehensive platform for crawling, analyzing, and classifying dark market credit card listings, featuring both keyword-based and machine learning risk assessment, a modern dashboard, and robust export capabilities.

---

## ⚡ Quick Start Script

To start the backend, frontend, and automatically kill any processes using the required ports, run:

```bash
./start_with_ml.sh
```

This script will:
- Kill any processes using the necessary ports
- Set up and use a Python virtual environment (venv) for the backend automatically
- Install backend Python dependencies if needed
- Start the backend (ML API)
- Start the frontend (dashboard)

> **Note:** Ensure `backend/data/dark_market_output_v2.csv` exists before running the script.

---

## 🛠️ Requirements

- **Python:** 3.8+
- **Node.js:** 16+
- **npm:** 8+
- **OS:** macOS, Linux, or WSL recommended

---

## 🗂️ Project Structure

```
white_paper/
├── backend/
│   ├── api/                    # Backend API (ML, data processing)
│   ├── data/                   # All data files and subfolders
│   │   ├── sites/              # Crawled site data (per-market JSON)
│   │   ├── keywords/           # Risk-level keyword files (editable)
│   │   ├── keyword_occurrences.json
│   │   ├── site_keyword_analysis.json
│   │   ├── credit_card_listings.csv
│   │   └── stop_words.txt
│   ├── requirements.txt
│   └── scripts/                # Analysis, crawling, and utility scripts
├── frontend/                   # Next.js dashboard (UI)
├── docs/                       # Documentation
└── start_with_ml.sh            # Quick start script
```

---

## 🏁 Quick Start (Manual/Advanced)

If you prefer to start components manually:

### 1. Start the Backend (ML API)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd api
python ml_api.py
```
- Runs on [http://localhost:9000](http://localhost:9000)

### 2. Start the Frontend (Dashboard)
```bash
cd frontend
npm install
npm run dev
```
- Runs on [http://localhost:3000](http://localhost:3000)

---

## 🧠 ML & Risk Classification

- **Random Forest Classifier** with TF-IDF features
- Trains on credit card listings (70/30 split)
- Shows accuracy, confusion matrix, classification report, and top feature importances
- **Keyword-based risk**: Fast, interpretable, and editable via JSON
- **Dual comparison**: Instantly see where ML and keyword risk agree/disagree

---

## 📊 Dashboard Highlights

- **Header**: Modern gradient, bold title, and subtitle
- **Risk Comparison Table**: Paginated, filterable, with CSV export
- **Feature Importance**: Horizontal bar chart of top features
- **Classification Report & Confusion Matrix**: Side by side, visually enhanced
- **Risk Prediction Demo**: Input title/description/category, get instant ML risk prediction
- **Risk Guide**: Color-coded, clear risk level explanations
- **Download All (ZIP)**: Model, vectorizer, and PDF report (with all metrics and feature importances)
- **Responsive**: Looks great on desktop and mobile

---

## 📝 API Endpoints (ML Backend)

- `GET /health` — API status
- `GET /model_info` — Model metrics, confusion matrix, feature importances
- `POST /predict` — Predict risk for a single text
- `POST /predict_batch` — Batch risk prediction
- `POST /retrain` — Retrain model
- `GET /download_model` — Download model (.pkl)
- `GET /download_vectorizer` — Download vectorizer (.pkl)
- `GET /download_report_pdf` — Download PDF report (with all metrics)
- `GET /download_all_zip` — Download ZIP (model, vectorizer, PDF)

---

## 🛠️ Customization & Data

- **Edit risk keywords:** `backend/data/keywords/keywords_*.json`
- **Add/clean stop words:** `backend/data/stop_words.txt`
- **Retrain model:** Use dashboard or `POST /retrain`
- **Update/expand data:** Place new JSON in `backend/data/sites/`

---

## 🧩 Technical Details

- **Crawling**: Selenium + Tor, robust to dynamic content
- **Keyword Discovery**: Recursively extracts, tokenizes, and ranks all keywords
- **ML Pipeline**: TF-IDF vectorization, Random Forest, feature importance, confidence scores
- **PDF Export**: All report sections (model info, feature importances, classification report, confusion matrix) are kept together for clarity
- **Frontend**: Next.js, React, modern CSS, fully responsive

---

## 📚 Example Workflow

1. **Crawl** dark market sites → JSON
2. **Discover** keywords → `data/keyword_occurrences.json`
3. **Classify** keywords by risk (edit JSON)
4. **Train** ML model (auto or via dashboard)
5. **Compare** ML vs. keyword risk in dashboard
6. **Predict** risk for new listings
7. **Export** results, model, and reports

---

## 🤝 Credits

- Data science, crawling, and ML: [Your Name/Team]
- UI/UX and dashboard: [Your Name/Team]
- Open source libraries: Selenium, scikit-learn, pandas, Flask, Next.js, React, ReportLab, etc.

---

## 📄 License

MIT License. For research and educational use only. 
This script:
- Scans all JSON files in `data/sites/*/*.json`
- Extracts all string values from all fields
- Tokenizes and counts keyword occurrences
- Filters out stop words and common terms
- Outputs ranked keyword list to `data/keyword_occurrences.json`

### 4. Keyword Risk Classification
Keywords are organized into separate JSON files by risk level for better management:

**Critical Risk (`data/keywords/keywords_critical.json` - 140 keywords):**
- Identity theft terms: fullz, dumps, cloned, ssn, dob, identity, stolen
- Cyber attack terms: hack, exploit, malware, ransomware, phishing, ddos
- Financial fraud: fraud, theft, laundering, extortion, blackmail
- Card data: cvv, pin, track1, track2, bin, expiry, valid, fresh, live
- Account credentials: login, password, username, email, phone, address
- Service accounts: paypal, amazon, ebay, netflix, uber, airbnb
- Geographic indicators: worldwide, asia, latin, europe, africa, america

**High Risk (`data/keywords/keywords_high.json` - 93 keywords):**
- Payment methods: visa, mastercard, amex, btc, crypto, wallet
- Financial services: paypal, neteller, skrill, western, moneygram
- Digital payments: venmo, cashapp, zelle, apple, google, stripe
- Cryptocurrencies: ethereum, litecoin, monero, dash, ripple
- Trading platforms: coinbase, binance, kraken, robinhood
- Business services: shipping, delivery, logistics, marketplace, vendor

**Medium Risk (`data/keywords/keywords_medium.json` - 94 keywords):**
- General financial: card, bank, money, shop, prepaid, gift
- Payment processing: stripe, square, payoneer, wise, revolut
- Digital banking: monzo, chime, sofi, business, enterprise
- Trading services: broker, dealer, reseller, distributor
- Geographic regions: usa, asia, eurozone, latin, europe, africa

**Low Risk (`data/keywords/keywords_low.json` - 83 keywords):**
- File formats: webp, png, jpg, pdf, doc, txt, csv, json, xml, zip
- General terms: info, data, file, tool, product, content, uploads
- Documentation: help, contact, about, terms, privacy, faq, support
- Account services: login, register, signup, password, recovery
- Technical terms: http, configuration, settings, preferences, profile

### 5. Machine Learning Classification
- Uses Random Forest classifier with TF-IDF features
- Trains on credit card related listings (70% train, 30% test split)
- Provides confidence scores and feature importance
- Achieves ~95% accuracy on test data

## Project Structure

```
white_paper/
├── data/
│   ├── sites/                    # Crawled site data
│   │   ├── 55niks/
│   │   ├── bitcards/
│   │   ├── cardht/
│   │   └── ...
│   ├── keywords/                  # Keyword classification files
│   │   ├── keywords_critical.json # Critical risk keywords
│   │   ├── keywords_high.json     # High risk keywords
│   │   ├── keywords_medium.json   # Medium risk keywords
│   │   └── keywords_low.json      # Low risk keywords
│   ├── keyword_occurrences.json  # Discovered keywords
│   ├── site_keyword_analysis.json # Per-site keyword analysis
│   ├── credit_card_listings.csv  # Extracted credit card data
│   └── stop_words.txt           # Stop words for filtering
├── scripts/
│   ├── crawling/                # Web crawling scripts
│   └── ml/                      # Machine learning scripts
├── frontend/                    # Next.js dashboard
└── discover_site_keywords.py    # Keyword discovery script
```

## Key Features

### Automated Keyword Discovery
- **Recursive Analysis**: Extracts keywords from all JSON fields
- **Frequency Ranking**: Orders keywords by occurrence count
- **Stop Word Filtering**: Removes common, non-meaningful terms
- **Flexible Configuration**: Easy to modify keyword categories
- **Separate Risk Files**: Organized keyword management by risk level

### Dual Classification System
- **Rules-Based**: Fast, interpretable keyword matching
- **Machine Learning**: Sophisticated pattern recognition
- **Comparison Dashboard**: Side-by-side risk assessment

### Real-Time Processing
- **Live API**: RESTful endpoints for predictions
- **Batch Processing**: Handle multiple listings efficiently
- **Model Retraining**: Update ML model with new data

## Usage

### 1. Start the Backend
```bash
# Start ML API server
python3 scripts/ml/ml_api.py
```

### 2. Start the Frontend
```bash
cd frontend
npm run dev
```

### 3. Access the Dashboard
- Frontend: http://localhost:3000
- ML API: http://localhost:9000

### 4. Discover Keywords
```bash
# Analyze all site data and find keywords
python3 discover_site_keywords.py

# View results
cat data/keyword_occurrences.json
cat data/site_keyword_analysis.json
```

### 5. Update Risk Classification
```bash
# Edit individual risk level files
nano data/keywords/keywords_critical.json
nano data/keywords/keywords_high.json
nano data/keywords/keywords_medium.json
nano data/keywords/keywords_low.json

# Keywords are organized by risk level:
# - critical: Most dangerous terms (identity theft, cyber attacks)
# - high: High-risk financial terms (payment methods, crypto)
# - medium: Moderate risk terms (business services, regions)
# - low: Low-risk technical terms (file formats, documentation)
```

## API Endpoints

### ML API (Port 9000)
- `GET /health` - Check API status
- `GET /model_info` - Get model performance metrics and confusion matrix
- `POST /predict` - Predict risk for single text (expects `{"text": "..."}`)
- `POST /predict_batch` - Batch prediction for multiple items (expects array of objects)
- `POST /retrain` - Retrain model with current data

### Frontend (Port 3000)
- Main dashboard with risk comparison
- Sample data loading and pagination
- Real-time ML predictions
- Model performance visualization
- **Enhanced ML metrics display** with readable confusion matrices and classification reports
- **Model retraining interface** with live updates
- **Improved table styling** with dark text colors and proper contrast
- **Dynamic performance metrics** showing accuracy, precision, recall, and F1-scores

## Data Flow

1. **Crawling** → HTML files from .onion sites
2. **Parsing** → Structured JSON data
3. **Keyword Discovery** → Frequency-ranked keyword list
4. **Manual Classification** → Risk levels assigned to keywords
5. **ML Training** → Model learns from classified data
6. **Prediction** → Real-time risk assessment
7. **Visualization** → Dashboard comparison

## Configuration

### Stop Words
Edit `data/stop_words.txt` to customize keyword filtering.

### Risk Classification
Modify individual keyword files to adjust risk levels:
- `data/keywords_critical.json` - Most dangerous terms
- `data/keywords_high.json` - High-risk financial terms
- `data/keywords_medium.json` - Moderate risk terms
- `data/keywords_low.json` - Low-risk technical terms

### ML Model
- Training data: `data/credit_card_listings_train.csv`
- Test data: `data/credit_card_listings_test.csv`
- Model parameters in `scripts/ml/ml_api.py`

## Technical Details

### Keyword Discovery Algorithm
1. Recursively extract all string values from JSON objects
2. Tokenize strings into individual words
3. Apply stop word filtering
4. Count frequency of each keyword
5. Rank by occurrence count
6. Output to JSON for manual classification

### Risk Classification Logic
1. Check for critical keywords first (highest priority)
2. Check for high-risk keywords
3. Check for medium-risk keywords  
4. Check for low-risk keywords
5. Default to "no_risk" if no keywords match

### Machine Learning Pipeline
1. TF-IDF vectorization of text features
2. Random Forest classification
3. Cross-validation for model selection
4. Feature importance analysis
5. Confidence score calculation

### Frontend Styling Improvements
1. **Confusion Matrix**: Dark text colors (`#1f2937`) with proper font weights (600 for headers, 500 for data)
2. **Classification Report**: Consistent dark styling with improved readability
3. **Table Styling**: Explicit color definitions and proper contrast ratios
4. **Visual Hierarchy**: Enhanced font sizes and weights for better information hierarchy
5. **Responsive Design**: Tables adapt to different screen sizes while maintaining readability

### ML API Enhancements
1. **Batch Prediction Endpoint**: New `/predict_batch` endpoint for processing multiple items at once
2. **Individual Prediction Endpoint**: `/predict` endpoint for single text predictions
3. **Real-time Model Retraining**: Dynamic model updates with live performance metrics
4. **Error Handling**: Graceful fallbacks when predictions fail
5. **Confidence Scores**: ML predictions include confidence levels for better decision making

## Recent Updates

### Training Data Correction (Latest - July 13, 2025)
- **Fixed critical data inconsistency** between keyword logic and ML training data
- **Relabeled 154 "Cards and CVV" items** from Low/High/Medium → Critical risk
- **Updated CSV training files** with corrected labels for proper ML training
- **Improved risk distribution**: Critical now 55.1% (was 32.5%), better balanced dataset
- **Enhanced model accuracy** with consistent labeling between keyword and ML approaches
- **Created `fix_training_labels.py`** script for automated data correction and validation

### Batch Prediction System (Latest)
- **New `/predict_batch` endpoint** for processing multiple items efficiently
- **Automatic ML predictions** for all loaded data in frontend
- **Real-time prediction updates** when loading different datasets
- **Improved error handling** with graceful fallbacks when ML predictions fail
- **Enhanced frontend integration** with automatic prediction fetching
- **Fixed API endpoint mismatch** between frontend expectations and backend implementation

### UI/UX Improvements
- **Enhanced confusion matrix readability** with dark text colors (`#1f2937`) and improved contrast
- **Better classification report styling** with consistent dark text and font weights (600 for headers, 500 for data)
- **Improved table styling** for ML metrics with explicit color definitions and proper contrast ratios
- **Enhanced visual hierarchy** with proper font weights and sizes for better information hierarchy
- **Responsive design** with tables that adapt to different screen sizes while maintaining readability

### ML Model Retraining Features
- **Real-time model retraining** via frontend interface with live updates
- **Dynamic confusion matrix updates** showing latest model performance metrics
- **Live classification report updates** with precision, recall, and F1-scores
- **Automatic data reloading** after model retraining to show updated predictions
- **Model performance tracking** with accuracy improvements from 80% to 82-85%
- **Random state variation** for different training splits and performance metrics

### Keyword Organization
- **Separated keyword files** by risk level for better management (410+ keywords total)
- **Enhanced keyword discovery** with per-site analysis and frequency ranking
- **Improved coded language detection** for dark web communication patterns
- **Comprehensive coverage** across all risk levels with detailed categorization

### Analysis Results
The latest keyword discovery found **107 unique keywords** across all sites:
- **Critical Risk**: car (157), pin (77), login (58), gold (40), account (40)
- **High Risk**: btc (100), visa (87), onion (80), usd (54), prepaid (46)
- **Medium Risk**: card (155), bank (10)
- **Low Risk**: http (40), data (29), product (28), pro (28), day (25)

### System Architecture Improvements
- **Fixed port conflicts** and service management issues
- **Enhanced error handling** for missing data files and API failures
- **Improved data synchronization** between frontend and backend
- **Better logging and debugging** for troubleshooting system issues
- **Streamlined startup scripts** for easier service management

## Data Correction and Validation

### Training Data Consistency
The system now includes automated data correction to ensure consistency between keyword logic and ML training data:

```bash
# Fix training data labels to match keyword logic
python3 fix_training_labels.py
```

This script:
- **Analyzes current risk distribution** by category
- **Identifies inconsistencies** between keyword logic and training labels
- **Relabels problematic items** (e.g., "Cards and CVV" → Critical)
- **Updates CSV files** for ML training
- **Validates the correction** with new distribution statistics

### Troubleshooting Common Issues

#### ML Predictions Don't Match Keyword Logic
**Problem**: ML model predicts different risk levels than keyword-based approach
**Solution**: 
1. Check if training data is consistent: `python3 fix_training_labels.py`
2. Retrain the model: Use the "Retrain Model" button in frontend
3. Verify the model is using updated data

#### Batch Predictions Not Working
**Problem**: Frontend shows "No ML Prediction" for all items
**Solution**:
1. Ensure ML API is running: `curl http://localhost:9000/health`
2. Check if `/predict_batch` endpoint exists
3. Verify frontend is using correct endpoint (`/api/ml/predict_batch`)

#### Port Conflicts
**Problem**: Services fail to start due to port conflicts
**Solution**:
1. Kill existing processes: `pkill -f "python3 ml_api.py"` or `pkill -f "next dev"`
2. Check port usage: `lsof -i :9000` or `lsof -i :3000`
3. Use different ports if needed

#### Missing Data Files
**Problem**: ML API fails to load training data
**Solution**:
1. Ensure CSV files exist: `ls data/credit_card_*.csv`
2. Regenerate from JSON: Use the data correction script
3. Check file paths in ML API configuration

## Contributing

To add new keywords or modify classifications:
1. Run keyword discovery to find new terms: `python3 discover_site_keywords.py`
2. Review `data/keyword_occurrences.json` for new keywords
3. Add relevant keywords to appropriate risk level files:
   - `data/keywords_critical.json` for identity theft/cyber attack terms
   - `data/keywords_high.json` for financial/payment terms
   - `data/keywords_medium.json` for business/service terms
   - `data/keywords_low.json` for technical/documentation terms
4. Test classification on sample data
5. Update documentation if needed

### Data Quality Guidelines
- **Consistency**: Ensure keyword logic matches training data labels
- **Balance**: Maintain reasonable distribution across risk levels
- **Validation**: Test predictions on real data samples
- **Documentation**: Update README when making significant changes

## Security Notes

- This system is for research and analysis purposes only
- All data is anonymized and used for risk assessment
- No actual financial transactions or personal data are processed
- Follow ethical guidelines when analyzing dark market data
- Keywords include coded language and dark web terminology for comprehensive detection 