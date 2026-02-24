# ── FILE: README.md ──

# 🛡️ AI-Based Fake News Detection System

An AI-powered web application that uses **Natural Language Processing**, **Machine Learning**, and **OpenRouter LLMs** to detect fake news articles. Built with a storytelling cinematic UI.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
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

- Python 3.9 or higher
- pip (Python package manager)
- Internet connection (for NLTK downloads)

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

## 🛠️ Tech Stack

| Component     | Technology                           |
|---------------|--------------------------------------|
| Language      | Python 3.13                          |
| GenAI LLM     | OpenRouter API                       |
| ML Library    | scikit-learn 1.3                     |
| Web Framework | Flask 3.0                            |
| Frontend      | HTML5, Bootstrap 5.3, Vanilla JS     |
| E2E Testing   | Playwright, Selenium                 |
| Verification  | Pytest                               |

---

## 📝 Quick Start (5 Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from Kaggle and place CSVs in dataset/ folder

# 3. Train the models
python train_model.py

# 4. Start the Flask server
python app.py

# 5. Open http://127.0.0.1:5000 in your browser
```

---

> 🎓 **Final Year College Project** — AI-Based Fake News Detection using NLP & Machine Learning
