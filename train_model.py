# ── FILE: train_model.py ──
# AI-Based Fake News Detection — Model Training Pipeline
# This script loads the dataset, preprocesses text, trains two ML models,
# evaluates them, and saves the best model + vectorizer as .pkl files.

import os
import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

# ─── Download required NLTK data ───────────────────────────────────────────────
print("[1/8] Downloading NLTK resources...")
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ─── Initialize NLP tools ──────────────────────────────────────────────────────
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


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


def load_dataset():
    """
    Load fake.csv and true.csv from the dataset/ folder, label them,
    combine, and shuffle.

    Returns:
        pd.DataFrame: Shuffled dataframe with 'text' and 'label' columns.
    """
    # Build paths relative to this script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")

    fake_path = os.path.join(dataset_dir, "Fake.csv")
    true_path = os.path.join(dataset_dir, "True.csv")

    print(f"[2/8] Loading datasets from: {dataset_dir}")

    # Load CSVs
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    print(f"     → Fake news articles : {len(fake_df):,}")
    print(f"     → Real news articles : {len(true_df):,}")

    # Assign labels: fake = 0, real = 1
    fake_df["label"] = 0
    true_df["label"] = 1

    # Combine and shuffle
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"     → Total articles     : {len(df):,}")
    return df


def apply_preprocessing(df):
    """
    Apply the NLP preprocessing function to every row in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe with a 'text' column.

    Returns:
        pd.DataFrame: Dataframe with cleaned 'text' column.
    """
    print("[3/8] Preprocessing text data (this may take a few minutes)...")
    df["text"] = df["text"].apply(preprocess_text)
    print("     → Preprocessing complete.")
    return df


def vectorize_text(X_train, X_test):
    """
    Fit a TF-IDF vectorizer on training data and transform both splits.

    Args:
        X_train (pd.Series): Training text data.
        X_test  (pd.Series): Testing text data.

    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, vectorizer)
    """
    print("[4/8] Vectorizing text with TF-IDF (max_features=5000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"     → Vocabulary size: {len(vectorizer.vocabulary_):,}")
    return X_train_tfidf, X_test_tfidf, vectorizer


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model and print detailed metrics.

    Args:
        model: Trained sklearn classifier.
        X_test: TF-IDF test features.
        y_test: True labels.
        model_name (str): Name for display purposes.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    return {
        "Model": model_name,
        "Accuracy": f"{accuracy:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1 Score": f"{f1:.4f}",
    }


def print_comparison_table(results):
    """
    Print a side-by-side comparison table of two models.

    Args:
        results (list[dict]): List of metric dictionaries from evaluate_model.
    """
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON TABLE")
    print(f"{'='*60}")
    header = f"  {'Metric':<15} | {results[0]['Model']:<22} | {results[1]['Model']:<22}"
    print(header)
    print(f"  {'-'*15}-+-{'-'*22}-+-{'-'*22}")

    for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        row = f"  {metric:<15} | {results[0][metric]:<22} | {results[1][metric]:<22}"
        print(row)

    print(f"{'='*60}\n")


def save_artifacts(model, vectorizer):
    """
    Save the trained model and TF-IDF vectorizer as .pkl files.

    Args:
        model: Trained sklearn classifier to save.
        vectorizer: Fitted TfidfVectorizer to save.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"[8/8] Artifacts saved:")
    print(f"     → Model      : {model_path}")
    print(f"     → Vectorizer  : {vectorizer_path}")


def main():
    """
    Main training pipeline — orchestrates every step from data loading
    through model evaluation and artifact saving.
    """
    print("\n" + "=" * 60)
    print("  AI-BASED FAKE NEWS DETECTION — MODEL TRAINING")
    print("=" * 60 + "\n")

    # Step 1–2: Load dataset
    df = load_dataset()

    # Step 3: Preprocess
    df = apply_preprocessing(df)

    # Step 4: Train/test split
    print("[5/8] Splitting dataset (80% train / 20% test, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )
    print(f"     → Train samples: {len(X_train):,}")
    print(f"     → Test  samples: {len(X_test):,}")

    # Step 5: Vectorize
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)

    # Step 6: Train Logistic Regression
    print("[6/8] Training Logistic Regression (max_iter=1000)...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    print("     → Logistic Regression training complete.")

    # Step 7: Train Multinomial Naive Bayes
    print("[7/8] Training Multinomial Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    print("     → Naive Bayes training complete.")

    # Evaluate both models
    lr_results = evaluate_model(lr_model, X_test_tfidf, y_test, "Logistic Regression")
    nb_results = evaluate_model(nb_model, X_test_tfidf, y_test, "Multinomial Naive Bayes")

    # Print comparison
    print_comparison_table([lr_results, nb_results])

    # Save the primary model (Logistic Regression) + vectorizer
    save_artifacts(lr_model, vectorizer)

    print("\n✅ Training pipeline complete! You can now run the Flask app.\n")


if __name__ == "__main__":
    main()
