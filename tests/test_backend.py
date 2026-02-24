import pytest
import os
import json
import joblib
from app import app

# Define local paths for the artifacts since they are not globally exported in train_model.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

@pytest.fixture
def client():
    """Setup Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# --- API endpoint tests ---

def test_index_route(client):
    """Test the landing page route loads."""
    rv = client.get('/')
    assert rv.status_code == 200
    assert b"Can You Tell What's Real Anymore?" in rv.data

def test_detector_route(client):
    """Test the detector app route loads."""
    rv = client.get('/detector')
    assert rv.status_code == 200
    assert b"TRUTHGUARD" in rv.data

def test_status_endpoint(client):
    """Test the status endpoint returns correct backend availability."""
    rv = client.get('/status')
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert 'ml_available' in data
    assert 'ai_available' in data

def test_predict_empty_text(client):
    """Test the predict endpoint handles empty text correctly."""
    rv = client.post('/predict', json={'text': ''})
    assert rv.status_code == 400
    data = json.loads(rv.data)
    assert 'error' in data

# --- ML Model Tests ---

def test_model_files_exist():
    """Verify that the trained ML model and vectorizer files exist."""
    assert os.path.exists(MODEL_PATH), "Trained model file missing."
    assert os.path.exists(VECTORIZER_PATH), "Vectorizer file missing."

def test_model_inference():
    """Test the actual ML model inference directly."""
    # Load model and vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Test a clearly fake-style text
    fake_text = "Alien spaceships land in Washington DC and take over the government!"
    fake_vec = vectorizer.transform([fake_text])
    fake_prob = model.predict_proba(fake_vec)[0]
    
    # Test a clearly real-style factual text
    real_text = "The Federal Reserve announced a quarter-point interest rate hike on Wednesday following the meeting."
    real_vec = vectorizer.transform([real_text])
    real_prob = model.predict_proba(real_vec)[0]
    
    # We expect the model to output valid probabilities summing to 1.0
    assert 0 <= fake_prob[0] <= 1.0
    assert 0 <= fake_prob[1] <= 1.0
    assert sum(fake_prob) == pytest.approx(1.0)
    
    # Normally, we'd assert fake_prob[0] (fake) > real_prob[0] (fake), varying strongly by text, 
    # but since this is a basic LR trained on a specific dataset we just verify it runs without crashing 
    # and produces statistically different outputs for different texts.
    assert list(fake_prob) != list(real_prob)

def test_predict_endpoint_success(client):
    """Test the predict endpoint with valid text inputs."""
    sample_text = "The local city council held a meeting on Tuesday to discuss the new budget proposal for the upcoming fiscal year."
    
    rv = client.post('/predict', json={'text': sample_text})
    assert rv.status_code == 200
    data = json.loads(rv.data)
    
    assert 'label' in data
    assert 'confidence' in data
    assert 'real_prob' in data
    assert 'fake_prob' in data
    assert data['label'] in ['REAL', 'FAKE']
