# Bangla Text Sentiment Analysis

This repository implements a classical-machine-learning pipeline for multi-label emotion classification of Bangla social-media comments. We address three main challenges—class imbalance, rich morphology, and subtle contextual cues—by combining five traditional classifiers in a soft-vote ensemble with per-label threshold calibration.


## 🔧 Features

- **Preprocessing**  
  - Unicode normalization (NFC)  
  - HTML, URL, user-mention removal  
  - Custom Bangla tokenizer  
  - Stop-word removal & suffix-stripping stemming  

- **Feature Engineering**  
  - Hybrid TF-IDF:  
    - Word n-grams (1–3 grams)  
    - Character n-grams (3–5 grams)  
  - Combined → Min-Max scaling to [0,1]  

- **Models**  
  - Linear SVM, Logistic Regression, Random Forest, Naive Bayes, k-Nearest Neighbors  
  - One-vs-Rest multi-label strategy  
  - Hyperparameter tuning via 5-fold `GridSearchCV`  

- **Ensemble & Calibration**  
  - Soft-vote ensemble of all five classifiers  
  - Per-label probability threshold sweep (0.10–0.90, step 0.05) on validation set  
  - Achieves **micro-F1 0.5827** and **macro-F1 0.4531** on the test set  

## 🚀 Quick Start

1. **Clone the repo**  
   ```bash
   git clone https://github.com/<your-username>/bangla-sentiment-analysis.git
   cd bangla-sentiment-analysis
Create and activate a virtual environment

python3 -m venv venv
source venv/bin/activate
Install dependencies


pip install -r requirements.txt
Prepare your data
Place your CSV splits in data/ (same format as the paper).

Run preprocessing


python src/preprocess.py --input data/train.csv --output data/train_cleaned.csv
Train & calibrate

python src/train.py --config src/config.yml
python src/calibrate.py --model models/ensemble.pkl --val data/val_cleaned.csv
Evaluate


python src/evaluate.py --model models/ensemble_calibrated.pkl --test data/test_cleaned.csv

📈 Results
Micro-F1: 0.5827

Macro-F1: 0.4531

Joy & Sadness F1 > 0.70; Surprise & Fear remain below 0.25 (class imbalance)

🛠️ Future Work
Integrate Bangla-BERT for contextual embeddings

Data augmentation (oversampling, back-translation)

Neural multi-label architectures (classifier chains, label correlations)

Lexicon-enhanced features
