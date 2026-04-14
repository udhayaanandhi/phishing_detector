import pandas as pd
import numpy as np
import os
import joblib
import re
import matplotlib
matplotlib.use('Agg') # For headless saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

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

def train_and_export():
    dataset_path = os.path.join('data', 'phishing.csv')
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Please run download_kaggle.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Clean data (handle NaNs and find right columns)
    text_col = 'Email Text' if 'Email Text' in df.columns else df.columns[1] # fallback
    label_col = 'Email Type' if 'Email Type' in df.columns else df.columns[-1] # fallback
    
    df = df.dropna(subset=[text_col, label_col])
    
    X = df[text_col]
    y = df[label_col]
    
    # Encode Labels: map phishing to 1, safe to 0
    # Phishing datasets usually indicate Phishing or Safe. We need binary labels for ROC.
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # figure out which one is phishing (usually "Phishing Email" -> contains phishing or spam)
    phishing_class_index = 1 if 'phishing' in str(le.classes_[1]).lower() or 'spam' in str(le.classes_[1]).lower() else 0

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print("Building model pipeline...")
    # Scikit-learn FeatureUnion connects custom features + TF-IDF
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('custom', UrlKeywordCounter())
        ])),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print("Training model (this may take a minute)...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if phishing_class_index == 1 else pipeline.predict_proba(X_test)[:, 0]

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}%")
    
    # Mapping back to actual class names for classification report
    target_names = [str(cls) for cls in le.classes_]
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

    # ---- GENERATE CHARTS ----
    images_dir = os.path.join('static', 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    plt.style.use('dark_background')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'confusion_matrix.png'), transparent=True)
    plt.close()

    # 2. ROC Curve
    # Ensure binary classification for ROC
    y_test_binary = (y_test == phishing_class_index).astype(int)
    fpr, tpr, _ = roc_curve(y_test_binary, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='cyan', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'roc_curve.png'), transparent=True)
    plt.close()

    # 3. Feature Importance (Top 10 Text Features)
    # The RandomForest classifier has feature_importances_
    # Feature names: TFIDF features + 2 custom features
    try:
        tfidf = pipeline.named_steps['features'].transformer_list[0][1]
        text_feature_names = tfidf.get_feature_names_out()
        all_feature_names = np.concatenate([text_feature_names, ['custom_url_count', 'custom_kw_count']])
        
        rf = pipeline.named_steps['clf']
        importances = rf.feature_importances_
        indices = np.argsort(importances)[-10:] # Top 10
        
        plt.figure(figsize=(8,5))
        plt.barh(range(10), importances[indices], color='mediumspringgreen', align='center')
        plt.yticks(range(10), [all_feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'feature_importance.png'), transparent=True)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot feature importances: {e}")

    # Export model and label encoder
    print("Exporting model pipeline...")
    joblib.dump(pipeline, 'phishing_pipeline.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    # Save a small metrics file for the dashboard
    with open('metrics.txt', 'w') as f:
        f.write(f"{acc * 100:.2f},{len(X)}\n")
        f.write(target_names[0] + "," + target_names[1])
        
    print("Training complete! Model and charts are saved.")

if __name__ == '__main__':
    train_and_export()
