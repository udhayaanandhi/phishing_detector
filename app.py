from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin

class UrlKeywordCounter(BaseEstimator, TransformerMixin):
    """Custom transformer to extract structural features from text."""
    def __init__(self):
        self.keywords = ['urgent', 'password', 'verify', 'click', 'account', 'suspended', 'login', 'bank']
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        features = np.zeros((len(X), 2))
        for i, text in enumerate(X):
            text_str = str(text).lower()
            # Feature 0: Count of URLs
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text_str)
            features[i, 0] = len(urls)
            # Feature 1: Keyword match count
            kw_count = sum(text_str.count(kw) for kw in self.keywords)
            features[i, 1] = kw_count
        return features

app = Flask(__name__)

# Load Model Pipeline and Label Encoder
try:
    pipeline = joblib.load('phishing_pipeline.pkl')
    le = joblib.load('label_encoder.pkl')
    # Determine the target index for Phishing
    if 'phishing' in str(le.classes_[1]).lower() or 'spam' in str(le.classes_[1]).lower():
        phishing_class_index = 1
    else:
        phishing_class_index = 0
except Exception as e:
    print(f"Error loading model: {e}")
    pipeline = None
    le = None
    phishing_class_index = 1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    metrics = {"accuracy": "N/A", "samples": "N/A"}
    try:
        with open('metrics.txt', 'r') as f:
            lines = f.readlines()
            if len(lines) >= 1:
                acc, count = lines[0].strip().split(',')
                metrics['accuracy'] = acc
                metrics['samples'] = count
    except:
        pass
    return render_template('dashboard.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    if not pipeline:
        return jsonify({'error': 'Model not loaded properly.'}), 500
        
    data = request.get_json()
    email_text = data.get('email_text', '')
    
    if not email_text.strip():
        return jsonify({'error': 'Email text cannot be empty!'}), 400

    # Predict
    prediction = pipeline.predict([email_text])[0]
    probabilities = pipeline.predict_proba([email_text])[0]
    
    label = le.inverse_transform([prediction])[0]
    phishing_prob = probabilities[phishing_class_index]
    safe_prob = probabilities[1 - phishing_class_index]
    
    confidence = max(phishing_prob, safe_prob)

    response = {
        'prediction': str(label),
        'confidence': f"{confidence * 100:.1f}%",
        'is_phishing': bool(prediction == phishing_class_index)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
