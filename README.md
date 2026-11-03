# Bangla Emotion Classification

Multi-label emotion classification of Bangla social-media comments with classical machine learning. Team project by Nidan Alam, Nazifa Tabassum Tandra and Fatema Akhter (2025).

## Problem

Each comment can carry any of six emotions: Love, Joy, Surprise, Anger, Sadness and Fear. The labels are imbalanced, with few Fear and Surprise examples.

## Data

`BanglaSentimentProject/data/`:

| File | Rows |
| --- | --- |
| `Train.csv` | 18,420 |
| `Val.csv` | 2,047 |
| `Test.csv` | 2,272 |

Columns: ID, Data (comment text), the six emotion labels, Topic, Domain and is_admin.

## Method

- **Cleaning** (`utils/preprocess.py`): HTML and URL removal, a regex tokenizer, stopword removal and light suffix stemming.
- **Training** (`models/train.py`): admin, duplicate and unlabeled rows are dropped. Fear is oversampled to 1,000 examples, then each label is randomly oversampled. Features are word 1 to 3-gram TF-IDF (min_df 5, max_df 0.8), character 3 to 5-gram TF-IDF (5,000 features) and six standardised count features (negation words, adjective and adverb suffixes, lexicon polarity). The top 10,000 features are kept by chi-squared score.
- **Models:** Linear SVM, Logistic Regression, Random Forest, k-Nearest Neighbours and Multinomial Naive Bayes, one binary classifier per label (`MultiOutputClassifier`). Each is grid-searched with 5-fold cross-validation on macro-F1 (grids in `utils/config.py`).
- **Ensemble** (`test_no_mlp.py`): the per-label probabilities of the five models are averaged (a sigmoid of the decision function for the SVM). A threshold for each label is swept from 0.10 to 0.90 in steps of 0.01 on `Val.csv`, then the ensemble is scored on `Test.csv`. `test.py` does the same with an extra PyTorch MLP (`models/train_torch_mlp.py`).

## Results

On the test split, as reported by the team: micro-F1 0.5827, macro-F1 0.4531. Joy and Sadness reach F1 above 0.70; Surprise and Fear stay below 0.25. The run outputs were not saved in this repository.

## How to run

```bash
cd BanglaSentimentProject
pip install numpy pandas scipy scikit-learn imbalanced-learn joblib tqdm torch
python models/train.py            # tunes the five models and saves them to models/
python models/train_torch_mlp.py  # optional MLP
python test_no_mlp.py             # thresholds on Val.csv, micro and macro F1 on Test.csv
python test.py                    # same, with the MLP in the ensemble
```

`test_no_mlp.py` imports `models/train_torch_mlp.py`, so PyTorch is needed even without the MLP.

## Known issues

- The tokenizer's character range leaves out Bangla vowel signs, so words are split into fragments, and the negation, suffix and lexicon features never fire on this data. Fixing the range and re-running is expected to change the scores above.

## Layout

- `BanglaSentimentProject/data/`: train, validation and test splits.
- `BanglaSentimentProject/utils/`: configuration and preprocessing.
- `BanglaSentimentProject/models/`: training scripts and the saved feature scaler.
- `BanglaSentimentProject/test.py`, `test_no_mlp.py`: threshold calibration and evaluation.

## License

MIT (see `LICENSE.txt`).
