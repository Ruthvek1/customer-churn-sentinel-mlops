<div align="center">

# 🔮 Customer Churn Prediction System

### End-to-End ML Pipeline with Production Drift Monitoring

[![CI Pipeline](https://github.com/YOUR_USERNAME/churn-prediction-system/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/churn-prediction-system/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Not just a model — a production-grade ML system that monitors, alerts, and self-heals.*

</div>

---

## 🎯 What Makes This Different

Most ML projects stop at `model.fit()`. This one goes further:

| Typical ML Project | This Project |
|:---|:---|
| ✅ Train a model | ✅ Train a model |
| ❌ Serve predictions | ✅ FastAPI REST API with Swagger docs |
| ❌ Monitor in production | ✅ Data drift detection with Evidently AI |
| ❌ Handle model decay | ✅ Automated retraining triggers |
| ❌ Dashboard | ✅ Multi-page Streamlit dashboard |
| ❌ Deployment-ready | ✅ Docker Compose + CI/CD |

> **The key insight:** Models decay in production when data distributions shift. This system detects that shift and triggers retraining — closing the MLOps loop.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  Telco Churn Dataset → Preprocessing → Feature Engineering      │
│                              ↓                                  │
│                    Reference Dataset (for drift)                │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                         ML LAYER                                │
│  XGBoost Classifier ← Hyperparameter Tuning (RandomizedSearchCV)│
│         ↓                                                       │
│  Model Artifact (.joblib) + Experiment Logs (JSON)              │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                              │
│  FastAPI Backend                                                │
│  ├── POST /predictions/predict     → Single prediction          │
│  ├── POST /predictions/predict/batch → Batch predictions        │
│  ├── GET  /drift/report            → Run drift detection        │
│  ├── POST /retrain                 → Trigger retraining         │
│  └── GET  /health                  → System health check        │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING LAYER                              │
│  Prediction Logger (SQLite) → Drift Detector (Evidently AI)     │
│         ↓                              ↓                        │
│  Prediction History           Alert Manager → Retrain Trigger   │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                     DASHBOARD LAYER                             │
│  Streamlit Multi-Page App                                       │
│  ├── 📊 Live Predictions    (input form + gauge chart)          │
│  ├── 🔍 Drift Monitor       (per-feature drift analysis)       │
│  ├── 📈 Model Performance   (ROC curve + confusion matrix)     │
│  └── 🔧 Retrain Control     (manual retrain + version history) │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/churn-prediction-system.git
cd churn-prediction-system

# Start everything with one command
docker-compose up --build
```

- **API:** http://localhost:8000/docs (Swagger UI)
- **Dashboard:** http://localhost:8501

### Option 2: Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset (place in data/raw/telco_churn.csv)
# Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# 4. Train the model
PYTHONPATH=. python -m src.model.train

# 5. Start the API
PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000

# 6. Start the dashboard (new terminal)
PYTHONPATH=. streamlit run dashboard/app.py
```

---

## 🧪 Tech Stack

| Component | Technology | Purpose |
|:---|:---|:---|
| **ML Model** | XGBoost + Scikit-learn | Churn classification with feature importance |
| **Backend API** | FastAPI + Uvicorn | Async REST API with auto-generated Swagger docs |
| **Dashboard** | Streamlit + Plotly | Interactive monitoring UI |
| **Drift Detection** | Evidently AI | KS test, chi-squared, PSI for distribution monitoring |
| **Data Storage** | SQLite | Zero-config prediction logging |
| **Containerization** | Docker + Docker Compose | One-command deployment |
| **CI/CD** | GitHub Actions | Automated linting + testing |
| **Config** | YAML + Pydantic Settings | Type-safe configuration management |
| **Logging** | Structlog | JSON-formatted structured logs |

> **Deep dive:** See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed tech stack comparisons and design decisions.

---

## 📁 Project Structure

```
├── src/
│   ├── data/          # Data preprocessing & feature engineering
│   ├── model/         # Training, prediction, evaluation
│   ├── monitoring/    # Drift detection, logging, alerts
│   ├── api/           # FastAPI routes & schemas
│   └── utils/         # Logging utilities
├── dashboard/         # Streamlit multi-page dashboard
├── config/            # YAML configuration
├── tests/             # Pytest test suite
├── .github/workflows/ # CI/CD pipeline
├── Dockerfile         # Multi-stage Docker build
├── docker-compose.yml # One-command deployment
└── DOCUMENTATION.md   # Detailed technical documentation
```

---

## 🔍 Key Features

### 1. Data Drift Detection ⭐
The headline feature — monitors incoming data for distributional shifts:
- **Kolmogorov-Smirnov test** for numerical features
- **Chi-squared test** for categorical features
- Per-feature drift scoring with visual heatmaps
- Automatic retraining triggers when drift exceeds thresholds

### 2. Production API
- Auto-generated Swagger documentation at `/docs`
- Request validation with Pydantic
- Model loaded once on startup (not per-request)
- CORS enabled for dashboard integration

### 3. Interactive Dashboard
- Real-time prediction with churn probability gauge
- Drift monitoring with trend analysis
- Model performance metrics (ROC-AUC, F1, confusion matrix)
- One-click retraining with before/after comparison

### 4. MLOps Practices
- Experiment logging with version tracking
- Model versioning with timestamps
- Docker containerization
- GitHub Actions CI pipeline
- Structured JSON logging

---

## 🧠 Skills Demonstrated

- **Machine Learning:** Classification, feature engineering, hyperparameter tuning, cross-validation
- **Backend Development:** REST API design, async Python, request validation
- **MLOps:** Model monitoring, drift detection, automated retraining, experiment tracking
- **Data Engineering:** ETL pipeline, data validation, schema enforcement
- **DevOps:** Docker, CI/CD, health checks, structured logging
- **Frontend:** Dashboard design, data visualization, real-time monitoring

---

## 📊 Model Performance

| Metric | Score |
|:---|:---|
| **Accuracy** | ~0.80 |
| **ROC-AUC** | ~0.84 |
| **F1 Score** | ~0.60 |
| **Precision** | ~0.65 |
| **Recall** | ~0.55 |

*Metrics from XGBoost with RandomizedSearchCV tuning on the Telco Customer Churn dataset.*

---

## 🧪 Running Tests

```bash
# Run all tests
PYTHONPATH=. python -m pytest tests/ -v

# Run specific test suites
PYTHONPATH=. python -m pytest tests/test_model.py -v
PYTHONPATH=. python -m pytest tests/test_api.py -v
PYTHONPATH=. python -m pytest tests/test_drift.py -v
```

---

## 📖 Documentation

For detailed technical documentation including:
- Complete tech stack comparisons ("why X over Y")
- System architecture deep dive
- Drift detection methodology
- API design patterns
- Deployment strategies

👉 See **[DOCUMENTATION.md](DOCUMENTATION.md)**

---

## 🚀 Future Improvements

- [ ] Model A/B testing framework
- [ ] Kubernetes deployment with Helm charts
- [ ] Real-time streaming predictions with Kafka
- [ ] Feature store integration (Feast)
- [ ] Advanced alerting (Slack/email notifications)
- [ ] Model explainability (SHAP values per prediction)
- [ ] Data versioning with DVC

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for ML Engineering**

*"Not just a model — a system."*

</div>
