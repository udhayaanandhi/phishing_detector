# PhishShield AI: Advanced Email Threat Detection

PhishShield AI is a fully-fledged Machine Learning application designed to classify emails as either "Safe" or "Phishing" in real-time. It features an automated data training pipeline and a beautifully designed, modern web interface.

## 🚀 Features

- **High-Accuracy Classification**: Uses a Random Forest model trained alongside a custom TF-IDF feature pipeline that looks for suspicious links and malicious keywords.
- **Real-Time Scanner**: Predict threats instantly by pasting any email or text snippet into the scanner.
- **Confidence Scoring**: Returns a percentage probability for the prediction (e.g., 98.4% Confidence).
- **Interactive Dashboard**: Dynamically surfaces visualized metrics generated during training, such as the Confusion Matrix, ROC Curve, and Top Feature Importances.
- **Premium Aesthetics**: The UI is built entirely with high-quality vanilla CSS featuring glassmorphism, modern gradients, and micro-animations.

## 📂 Project Structure

```text
phishing_detector/
├── app.py                  # The Flask web server powering the interface and APIs
├── phishing_model.py       # The machine learning pipeline for processing data and training the model
├── download_kaggle.py      # Script to generate or download the dataset over Kaggle API
├── requirements.txt        # Required Python packages
├── data/                   # Directory containing the phishing.csv dataset
├── templates/              
│   ├── index.html          # Main Scanner Interface
│   └── dashboard.html      # Model Dashboard Interface
└── static/                 
    ├── css/style.css       # Dynamic stylesheets
    └── images/             # Automated charts (Confusion Matrix, ROC Curve, etc.) generated during model training
```

## 🛠️ Setup & Installation

**1. Navigate to the project directory**
```bash
cd phishing_detector
```

**2. Install dependencies**
Install the necessary python modules via pip:
```bash
pip install -r requirements.txt
```

**3. (Optional) Re-train the model**
The project comes with a pre-trained `phishing_pipeline.pkl`. If you wish to re-train the model to generate slightly different charts or use new data:
```bash
python download_kaggle.py   # Ensure data/phishing.csv is ready
python phishing_model.py    # Trains the model and exports the .pkl files and the dashboard images
```

**4. Start the Application**
Run the Flask server:
```bash
python app.py
```

Navigate to `http://127.0.0.1:5000` in your web browser to enjoy the application!
