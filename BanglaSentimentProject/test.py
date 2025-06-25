#!/usr/bin/env python
"""
test.py — Calibrate per-label thresholds on Val.csv, then evaluate soft-vote ensemble on Test.csv.
"""

import os, sys, json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# ── ensure we can import from utils/ and models/ ─────────────────────────────
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))

import utils.config as config
from models.train_torch_mlp import MLP, vectorise

# ── helper to load ensemble probabilities for a given set of texts ─────────
def get_ensemble_probs(texts):
    X = vectorise(texts)  # numpy array (n_samples × n_features)

    # 1) scikit-learn
    sk = joblib.load(config.MODELS_PKL)
    probs = []
    for name, mdl in sk.items():
        if hasattr(mdl, "predict_proba"):
            m = np.vstack([e.predict_proba(X)[:,1] for e in mdl.estimators_]).T
        else:
            df_mat = np.vstack([e.decision_function(X) for e in mdl.estimators_]).T
            m = 1/(1+np.exp(-df_mat))
        probs.append(m)

    # 2) GPU-MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = MLP(X.shape[1], len(config.LABELS)).to(device)
    mlp.load_state_dict(torch.load(
        os.path.join(config.MODELS_DIR, "mlp_torch_best.pth"),
        map_location=device
    ))
    mlp.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X).float().to(device)
        m = mlp(Xt).cpu().numpy()
    probs.append(m)

    # average
    return sum(probs)/len(probs)

# ── 1) calibrate thresholds on validation set ─────────────────────────────────
val_df = pd.read_csv(config.VAL_FILE)
val_df = val_df[val_df.get("is_admin",0)==0]  # drop admin if present
val_df = val_df.dropna(subset=["Data"])
y_val  = val_df[config.LABELS].values
texts_val = val_df["Data"].astype(str).tolist()

print("[*] Computing ensemble probabilities on Val.csv…")
val_probs = get_ensemble_probs(texts_val)

# if thresholds already exist, load them
thresh_path = config.THRESHOLDS_JSON
if os.path.exists(thresh_path):
    print(f"[✓] Loading existing thresholds from {thresh_path}")
    thresholds = json.load(open(thresh_path))
else:
    print("[*] Calibrating thresholds per label…")
    thresholds = {}
    # for each label, sweep threshold 0.1→0.9 in 0.01 increments
    for idx, lbl in enumerate(config.LABELS):
        best_f1, best_t = -1, 0.5
        scores = []
        for t in np.linspace(0.1, 0.9, 81):
            y_pred = (val_probs[:,idx] >= t).astype(int)
            f1 = f1_score(y_val[:,idx], y_pred)
            scores.append((f1,t))
            if f1>best_f1:
                best_f1, best_t = f1, t
        thresholds[lbl] = best_t
        print(f"  {lbl}: best F1={best_f1:.3f} at t={best_t:.2f}")
    # save
    with open(thresh_path, "w", encoding="utf8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)
    print(f"[✓] Saved thresholds to {thresh_path}")

# build threshold vector
thr_vec = np.array([thresholds[lbl] for lbl in config.LABELS])

# ── 2) evaluate on Test.csv ─────────────────────────────────────────────────
test_df = pd.read_csv(config.TEST_FILE)
test_df = test_df[test_df.get("is_admin",0)==0]
test_df = test_df.dropna(subset=["Data"])
y_test = test_df[config.LABELS].values
texts_test = test_df["Data"].astype(str).tolist()

print("[*] Computing ensemble probabilities on Test.csv…")
test_probs = get_ensemble_probs(texts_test)

print("[*] Applying calibrated thresholds…")
y_pred = (test_probs >= thr_vec).astype(int)

micro = f1_score(y_test, y_pred, average="micro")
macro = f1_score(y_test, y_pred, average="macro")
print(f"\nEnsemble w/ calibration — micro-F1: {micro:.4f}   macro-F1: {macro:.4f}\n")
print(classification_report(y_test, y_pred, target_names=config.LABELS))
