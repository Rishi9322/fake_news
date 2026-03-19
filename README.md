# 🛡️ AI-Based Fake News Detection System

An AI-powered web application that uses **Natural Language Processing**, **Machine Learning**, and **OpenRouter LLMs** to detect fake news articles. Built with a storytelling cinematic UI.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1-green?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange?logo=scikit-learn)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple?logo=bootstrap)
![Playwright](https://img.shields.io/badge/Playwright-Testing-red?logo=playwright)

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Setup Instructions](#-setup-instructions)
- [Dataset Download](#-dataset-download)
- [Train the Model](#-train-the-model)
- [Run the App](#-run-the-app)
- [Secure API Keys](#-secure-api-keys)
- [Deploy Live API (Render + GitHub Actions)](#-deploy-live-api-render--github-actions)
- [Testing Suites (Playwright & Selenium)](#-testing-suites)
- [Tech Stack](#-tech-stack)

---

## ✨ Features

- **Cinematic Landing Page** — Interactive scroll-animations telling the story of misinformation
- **Dual Verification Modes** — Toggles between Local ML and OpenRouter Deep AI Reasoning (`arcee-ai/trinity-large-preview:free`)
- **Dual Model Training** — Logistic Regression (primary) + Naive Bayes (comparison)
- **NLP Preprocessing Pipeline** — Tokenization, stopword removal, stemming
- **Real-Time API Endpoints** — Flask API providing predictions & confidence metrics
- **Beautiful Unified UI** — Dark theme, glassmorphism, responsive Bootstrap 5, and floating viral node animations
- **Comprehensive E2E Testing** — Robust backend and frontend UI tests in Pytest using both Playwright and Selenium WebDriver.

---

## 📁 Project Structure

```
fake-news-detection/
│
├── dataset/                  # ← Place CSV files here
├── templates/
│   ├── landing.html          # Cinematic landing page
│   └── index.html            # Core detector tool
├── tests/
│   ├── test_backend.py       # API, Model, and Route tests
│   ├── test_selenium.py      # Selenium automated UI testing
│   └── test_ui.py            # Playwright automated UI testing
│
├── train_model.py            # ML training pipeline
├── app.py                    # Flask web server
├── .env                      # API keys (OpenRouter)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Internet connection (for NLTK downloads and OpenRouter API)

### 1. Clone or Download the Project

```bash
cd fake-news-detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Download

This project uses the **Fake and Real News Dataset** from Kaggle.

### Steps:

1. Visit: [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Click **Download** (you will need a free Kaggle account)
3. Extract the ZIP file
4. Place `Fake.csv` and `True.csv` inside the `dataset/` folder:

```
dataset/
├── Fake.csv    (~23,000 fake news articles)
└── True.csv    (~21,000 real news articles)
```

---

## 🧠 Train the Model

Once the dataset is in place, run the training script:

```bash
python train_model.py
```

This will:
- Load and preprocess all articles
- Train Logistic Regression and Naive Bayes models
- Print evaluation metrics and comparison table
- Save `model.pkl` and `vectorizer.pkl` in the project root

**Expected training time:** 2–5 minutes depending on hardware.

---

## 🚀 Run the App

After training is complete:

```bash
python app.py
```

Then open your browser and navigate to:

```
http://127.0.0.1:5000
```

1. You will be greeted by the **Cinematic Landing Page**. Scroll down to read the context.
2. Click **"Try the Detector"** to enter the core app at `/detector`.
3. Paste any news article and toggle between **Fast ML** and **Deep AI reasoning**. Click **"Analyze News"**.

---

## 🔐 Secure API Keys

This project reads all sensitive values from environment variables.

Use a local `.env` file (not committed) with:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openai/gpt-4o-mini
APP_API_KEY=your_private_api_key   # optional
```

Notes:
- If `OPENROUTER_API_KEY` is missing, `/predict-ai` is disabled.
- If `APP_API_KEY` is set, API routes `/predict` and `/predict-ai` require header `X-API-Key: <APP_API_KEY>`.
- A template is included in `.env.example`.

---

## 🌐 Deploy Live API (Render + GitHub Actions)

This repo includes a CI/CD workflow at `.github/workflows/python-app.yml` that:
1. Runs lint + backend tests on each push/PR.
2. Triggers Render deployment on push to `main` (when `RENDER_DEPLOY_HOOK` is configured).

### One-time setup

1. Create a **Render Web Service** connected to this GitHub repo.
2. Render will auto-detect the build and start commands from `Procfile`:
	- **Procfile** specifies: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
	- **build.sh** runs: `pip install --upgrade pip && pip install -r requirements.txt`
3. Render will auto-use Python 3.14.3 (default). All dependencies in `requirements.txt` have pre-built wheels for Python 3.14:
	- Flask 3.1.0, scikit-learn 1.7.2, pandas 2.3.2, numpy 2.2.3, nltk 3.9.1, joblib 1.4.2, requests 2.32.3, python-dotenv 1.0.1, gunicorn 23.0.0
4. In Render **Environment Variables**, set:
	- `OPENROUTER_API_KEY` (get from https://openrouter.ai/keys)
	- `OPENROUTER_MODEL` (example: `arcee-ai/trinity-large-preview:free`)
	- `APP_API_KEY` (optional, for header-based API security)
	- `FLASK_DEBUG=False` (for production)
5. In Render, copy your **Deploy Hook URL** from Settings.
6. In GitHub repo settings → Secrets and variables → Actions, add:
	- `RENDER_DEPLOY_HOOK` = your deploy hook URL

### Deploy behavior

- GitHub Actions CI runs **lint + tests** on all pushes and PRs.
- On push to `main` with `RENDER_DEPLOY_HOOK` configured, CI automatically triggers Render deployment.
- If `RENDER_DEPLOY_HOOK` is not set, tests still run and pass; deploy step is silently skipped.
- Render deployment typically completes in 1-2 minutes (all wheels pre-built for Python 3.14).

### Go live

Push to `main` (or click **Manual Deploy** in Render).

After deployment, verify:

```bash
GET /health
GET /status
POST /predict
POST /predict-ai
```

### Troubleshooting

- **503 on all routes (`/`, `/health`, `/status`)**:
	The service failed to boot due to missing environment variables or API key issues. Check Render deploy logs for the actual error.
- **Build stuck or taking 60+ minutes**:
	This was a Python 3.14 wheel compatibility issue (resolved). All current dependencies have pre-built wheels; if it happens again, check that `requirements.txt` matches the versions in this commit.
- **AI button disabled on live UI**:
	The `/status` endpoint returns `ai_available: false`. Causes:
	  - Missing or invalid `OPENROUTER_API_KEY` in Render environment variables
	  - Incorrect `OPENROUTER_MODEL` name in environment
	  - Render hasn't redeployed after env var changes (manually trigger deploy in Render dashboard)
- **Local testing returns 503 on /predict-ai**:
	Missing `OPENROUTER_API_KEY` in local `.env` file. Add it and restart `app.py`.

---

## 🧪 Testing Suites

This project contains 14+ robust automated test cases. You can execute these test files using `pytest` to guarantee the integrity of the ML logic, API endpoints, and End-to-End User UI interactions.

### Run All Tests:
Ensure Flask is running on port `5000` via another terminal first.
```bash
python -m pytest tests/ -v
```

### 1. Backend & ML Logic
Validates the actual local ML model, prediction accuracy probabilities, and the API endpoints.
```bash
pytest tests/test_backend.py -v
```

### 2. Playwright UI Tests
Runs an isolated, extremely fast chromium headless browser mapping dynamic User flows (input typing verification, error cases).
```bash
pytest tests/test_ui.py -v
```

### 3. Selenium UI Tests
Traditional native WebDriver test cases using the built-in Selenium Manager to interact with the DOM elements synchronously.
```bash
pytest tests/test_selenium.py -v
```

---

## 💻 Tech Stack

### Backend
- **Framework**: Flask 3.1.0 (lightweight Python web framework)
- **ML Pipeline**: scikit-learn 1.7.2 (Logistic Regression, TF-IDF vectorizer)
- **NLP**: nltk 3.9.1 (tokenization, stopword removal, stemming)
- **Data Processing**: pandas 2.3.2, numpy 2.2.3
- **API Calls**: requests 2.32.3 (OpenRouter LLM integration)
- **Server**: gunicorn 23.0.0 (production WSGI server)
- **Environment**: python-dotenv 1.0.1

### Frontend
- **Templating**: Jinja2, HTML5, CSS3, Bootstrap 5.3
- **Styling**: Glassmorphism, animations, dark theme
- **JavaScript**: Vanilla JS for interactivity (fetch API, DOM manipulation)

### Testing
- **pytest**: Test framework
- **Playwright**: Fast, headless browser testing
- **Selenium**: Traditional WebDriver-based UI testing

### Deployment
- **Platform**: Render.com (Python 3.14.3 runtime)
- **CI/CD**: GitHub Actions
- **Version Control**: Git + GitHub

---

## 📝 Quick Start (End-to-End)

```bash
# 0. Clone repo
git clone https://github.com/Rishi9322/fake_news.git && cd fake_news

# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # or: source venv/bin/activate

# 2. Install requirements
pip install -r requirements.txt

# 3. Download dataset (optional for UI, required for training)
# From kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
# Extract Fake.csv and True.csv to ./dataset/

# 4. Train model (optional for API only)
python train_model.py

# 5. Start the Flask server
python app.py

# 6. Open http://127.0.0.1:5000 in your browser
```

---

> 🎓 **Final Year College Project** — AI-Based Fake News Detection using NLP & Machine Learning
