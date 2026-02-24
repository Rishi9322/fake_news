# ── FILE: app.py ──
# AI-Based Fake News Detection — Flask Web Application
# Serves the frontend and exposes /predict (ML) and /predict-ai (OpenRouter) endpoints.

import os
import re
import json
import string
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# ─── Load environment variables from .env file ─────────────────────────────────
load_dotenv()

# ─── Download required NLTK data ───────────────────────────────────────────────
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
except PermissionError:
    print("⚠️  NLTK data download skipped (file locked). Using existing data.")

# ─── Initialize Flask app ──────────────────────────────────────────────────────
app = Flask(__name__)

# ─── Initialize NLP tools ──────────────────────────────────────────────────────
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ─── Load model and vectorizer on startup (optional — app works in API-only mode) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

model = None
vectorizer = None

try:
    import joblib
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")
    print(f"✅ Vectorizer loaded from: {VECTORIZER_PATH}")
except FileNotFoundError:
    print("⚠️  model.pkl / vectorizer.pkl not found — ML prediction disabled.")
    print("   Run 'python train_model.py' first, or use AI (OpenRouter) mode.")

# ─── OpenRouter Configuration ──────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

if OPENROUTER_API_KEY:
    print(f"✅ OpenRouter API key loaded (model: {OPENROUTER_MODEL})")
else:
    print("⚠️  OPENROUTER_API_KEY not set — AI prediction disabled.")
    print("   Set it in a .env file or as an environment variable.")


def preprocess_text(text):
    """
    Apply full NLP preprocessing pipeline to a single text string.

    Steps:
        1. Convert to lowercase
        2. Remove punctuation and numbers
        3. Tokenize into words
        4. Remove English stopwords
        5. Apply Porter Stemmer

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned and stemmed text ready for vectorization.
    """
    # Step 1: Lowercase
    text = str(text).lower()

    # Step 2: Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Step 3: Tokenize
    tokens = word_tokenize(text)

    # Step 4: Remove stopwords  &  Step 5: Stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)


def call_openrouter(text):
    """
    Call the OpenRouter API to analyze news text using an LLM.

    Sends the article to the configured LLM model with a system prompt
    that instructs it to classify the news as REAL or FAKE with a
    confidence score and explanation.

    Args:
        text (str): Raw news article text to analyze.

    Returns:
        dict: Parsed response with label, confidence, probabilities, and reasoning.

    Raises:
        ValueError: If the API key is not configured.
        requests.RequestException: If the API call fails.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OpenRouter API key is not configured.")

    # System prompt instructing the LLM to act as a fake news detector
    system_prompt = """You are an expert fake news detection AI. Analyze the given news article and determine whether it is REAL or FAKE.

You MUST respond with ONLY a valid JSON object in this exact format (no markdown, no extra text):
{
    "label": "REAL" or "FAKE",
    "confidence": <number between 50 and 99>,
    "fake_prob": <number>,
    "real_prob": <number>,
    "reasoning": "<brief 2-3 sentence explanation of why you classified it this way>"
}

Rules:
- fake_prob + real_prob must equal 100
- confidence must match the probability of your chosen label
- Be objective and analyze writing style, claims, sources, and tone
- Consider sensationalism, emotional language, and factual consistency"""

    # Build the API request
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Fake News Detector",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this news article:\n\n{text[:4000]}"},
        ],
        "temperature": 0.3,
        "max_tokens": 300,
    }

    # Make the API call
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    # Parse the LLM response
    result = response.json()
    content = result["choices"][0]["message"]["content"].strip()

    # Clean markdown code fences if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    parsed = json.loads(content)

    return {
        "label": parsed.get("label", "UNKNOWN"),
        "confidence": float(parsed.get("confidence", 50)),
        "fake_prob": float(parsed.get("fake_prob", 50)),
        "real_prob": float(parsed.get("real_prob", 50)),
        "reasoning": parsed.get("reasoning", ""),
        "model_used": OPENROUTER_MODEL,
    }


# ─── Routes ─────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    """
    Serve the cinematic storytelling landing page.

    Returns:
        Rendered landing.html template.
    """
    return render_template("landing.html")


@app.route("/detector")
def detector():
    """
    Serve the main news detector application page.

    Returns:
        Rendered index.html template.
    """
    return render_template("index.html")


@app.route("/status")
def status():
    """
    Return the availability status of both prediction backends.

    Returns:
        JSON with ml_available and ai_available booleans.
    """
    return jsonify({
        "ml_available": model is not None and vectorizer is not None,
        "ai_available": bool(OPENROUTER_API_KEY),
        "ai_model": OPENROUTER_MODEL,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    ML-based prediction: preprocess, vectorize, and classify using
    the trained scikit-learn Logistic Regression model.

    Expected JSON input:
        { "text": "some news article text..." }

    Returns:
        JSON: {
            "label": "FAKE" or "REAL",
            "confidence": float,
            "fake_prob": float,
            "real_prob": float
        }
    """
    try:
        # Check if ML model is loaded
        if model is None or vectorizer is None:
            return jsonify({
                "error": "ML model not available. Run 'python train_model.py' first, or use AI mode."
            }), 503

        # Parse input
        data = request.get_json(force=True)
        text = data.get("text", "").strip()

        # Validate non-empty input
        if not text:
            return jsonify({"error": "Please provide some news text to analyze."}), 400

        # Preprocess using the identical pipeline from training
        cleaned_text = preprocess_text(text)

        # Vectorize
        text_vector = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]

        # Build response — class order is [0=FAKE, 1=REAL]
        fake_prob = round(probabilities[0] * 100, 2)
        real_prob = round(probabilities[1] * 100, 2)
        label = "REAL" if prediction == 1 else "FAKE"
        confidence = real_prob if prediction == 1 else fake_prob

        return jsonify(
            {
                "label": label,
                "confidence": confidence,
                "fake_prob": fake_prob,
                "real_prob": real_prob,
            }
        )

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/predict-ai", methods=["POST"])
def predict_ai():
    """
    AI-based prediction: send the news text to an LLM via OpenRouter API
    for analysis. Returns label, confidence, probabilities, and reasoning.

    Expected JSON input:
        { "text": "some news article text..." }

    Returns:
        JSON: {
            "label": "FAKE" or "REAL",
            "confidence": float,
            "fake_prob": float,
            "real_prob": float,
            "reasoning": str,
            "model_used": str
        }
    """
    try:
        # Parse input
        data = request.get_json(force=True)
        text = data.get("text", "").strip()

        # Validate non-empty input
        if not text:
            return jsonify({"error": "Please provide some news text to analyze."}), 400

        # Validate API key
        if not OPENROUTER_API_KEY:
            return jsonify({
                "error": "OpenRouter API key is not configured. Add OPENROUTER_API_KEY to your .env file."
            }), 503

        # Call OpenRouter API
        result = call_openrouter(text)
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse AI response. Please try again."}), 502
    except requests.exceptions.Timeout:
        return jsonify({"error": "AI API request timed out. Please try again."}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"AI API error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# ─── Run the app ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
