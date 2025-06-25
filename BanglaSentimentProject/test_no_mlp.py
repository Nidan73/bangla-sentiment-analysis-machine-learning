#!/usr/bin/env python
"""
test_no_mlp.py — Calibrate per-label thresholds on Val.csv, then evaluate soft-vote ensemble
(of only scikit-learn models) on Test.csv, handling undefined metrics.
"""

import os
import sys
import json
import warnings

from sklearn.exceptions import UndefinedMetricWarning

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# Suppress warnings for undefined metrics in samples average
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ── ensure we can import from utils/ and models/ ─────────────────────────────
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))

import utils.config as config
# vectorise stays the same as in your train_torch_mlp (does **not** depend on torch)
from models.train_torch_mlp import vectorise

def get_ensemble_probs(texts):
    """
    Load all classical models, get per-label probabilities, and average them.
    """
    # 1) vectorise into (n_samples × n_features) numpy array
    X = vectorise(texts)

    # 2) load all saved scikit-learn models
    sk_models = joblib.load(config.MODELS_PKL)
    probs = []
    for name, mdl in sk_models.items():
        # For probabilistic classifiers
        if hasattr(mdl, "predict_proba"):
            # each estimator_ within the MultiOutputClassifier is a binary classifier
            m = np.vstack([est.predict_proba(X)[:,1] for est in mdl.estimators_]).T
        else:
            # use decision_function + sigmoid
            df_mat = np.vstack([est.decision_function(X) for est in mdl.estimators_]).T
            m = 1.0 / (1.0 + np.exp(-df_mat))
        probs.append(m)

    # average across all models
    return sum(probs) / len(probs)

def main():
    # ── 1) calibrate thresholds on validation set ─────────────────────────────────
    val_df = pd.read_csv(config.VAL_FILE)
    # drop admin if present, drop nulls
    val_df = val_df[val_df.get("is_admin", 0) == 0].dropna(subset=["Data"])
    y_val = val_df[config.LABELS].values
    texts_val = val_df["Data"].astype(str).tolist()

    print("[*] Computing ensemble probabilities on Val.csv…")
    val_probs = get_ensemble_probs(texts_val)

    # load or find best thresholds per label
    thresh_path = config.THRESHOLDS_JSON
    if os.path.exists(thresh_path):
        print(f"[✓] Loading existing thresholds from {thresh_path}")
        thresholds = json.load(open(thresh_path, "r", encoding="utf8"))
    else:
        print("[*] Calibrating thresholds per label…")
        thresholds = {}
        for idx, lbl in enumerate(config.LABELS):
            best_f1, best_t = -1.0, 0.5
            for t in np.linspace(0.1, 0.9, 81):
                pred = (val_probs[:, idx] >= t).astype(int)
                f1 = f1_score(y_val[:, idx], pred)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            thresholds[lbl] = best_t
            print(f"  {lbl}: best F1={best_f1:.3f} at t={best_t:.2f}")
        # save for reuse
        with open(thresh_path, "w", encoding="utf8") as f:
            json.dump(thresholds, f, indent=2, ensure_ascii=False)
        print(f"[✓] Saved thresholds to {thresh_path}")

    # build threshold vector
    thr_vec = np.array([thresholds[lbl] for lbl in config.LABELS])

    # ── 2) evaluate on Test.csv ─────────────────────────────────────────────────
    test_df = pd.read_csv(config.TEST_FILE)
    test_df = test_df[test_df.get("is_admin", 0) == 0].dropna(subset=["Data"])
    y_test = test_df[config.LABELS].values
    texts_test = test_df["Data"].astype(str).tolist()

    print("[*] Computing ensemble probabilities on Test.csv…")
    test_probs = get_ensemble_probs(texts_test)

    print("[*] Applying calibrated thresholds…")
    y_pred = (test_probs >= thr_vec).astype(int)

    # compute and print final metrics, handling zero_division cases
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    print(f"\nEnsemble w/o MLP — micro-F1: {micro:.4f}   macro-F1: {macro:.4f}\n")
    print(classification_report(
        y_test,
        y_pred,
        target_names=config.LABELS,
        zero_division=0  # assign 0 for undefined precision/recall
    ))

if __name__ == "__main__":
    main()
