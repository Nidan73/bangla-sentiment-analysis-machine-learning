#!/usr/bin/env python
"""models/train.py — end-to-end training of classical ML + MLP, 5-fold CV, checkpoints."""

import os
import sys
import json
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import utils.config as config
from utils.preprocess import preprocess, random_oversample_texts

# ──────────────────────────────────────────────────────────────────────────────
# Dense‐feature helpers
NEGATION_WORDS = {"না","নয়","নয়","নাই","হয়নি","হয়নি","হয়না","হয়না"}
ADJ_SUFFIXES   = {"কি","তি","হীন","পূর্ণ","ময়","ময়"}
ADV_SUFFIXES   = {"ভাবে","মতো","করেই","সাথে","দিয়ে","প্রতি"}
LEXICON        = {w: float(v) for w, v in config.LEXICON.items()}

def fake_pos_counts(toks: List[str]) -> np.ndarray:
    adj = adv = neg = 0
    for t in toks:
        if t in NEGATION_WORDS: neg += 1
        if any(t.endswith(s) for s in ADJ_SUFFIXES): adj += 1
        if any(t.endswith(s) for s in ADV_SUFFIXES): adv += 1
    return np.array([adj, adv, neg], dtype=float)

def lexicon_feats(toks: List[str]) -> np.ndarray:
    pos_sum = neg_sum = hits = 0.0
    for t in toks:
        if t in LEXICON:
            val = LEXICON[t]; hits += 1
            if val > 0: pos_sum += val
            else:       neg_sum += val
    return np.array([pos_sum, neg_sum, hits], dtype=float)

# ──────────────────────────────────────────────────────────────────────────────
# Data cleanup & oversampling
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if 'is_admin' in df.columns:
        df = df[df['is_admin']==0]
    df = df.drop_duplicates(subset=['Data']).dropna(subset=['Data'])
    return df[df[config.LABELS].sum(axis=1)>0]

def oversample_fear(texts: List[str], y: np.ndarray) -> Tuple[List[str], np.ndarray]:
    idx  = np.where(y[:, config.LABELS.index('Fear')]==1)[0]
    need = max(0, 1000 - len(idx))
    if need:
        dup = np.random.choice(idx, size=need, replace=True)
        texts += [texts[i] for i in dup]
        y      = np.vstack([y, y[dup]])
    return texts, y

# ──────────────────────────────────────────────────────────────────────────────
# Build dense POS+lexicon features
def build_dense(tok_lists: List[List[str]]) -> csr_matrix:
    rows   = [np.hstack([fake_pos_counts(t), lexicon_feats(t)]) for t in tok_lists]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.vstack(rows))
    joblib.dump(scaler, os.path.join(config.MODELS_DIR, 'extra_scaler.pkl'))
    return csr_matrix(scaled)

# ──────────────────────────────────────────────────────────────────────────────
# Vectorise helper for validation
def vectorise(texts: List[str], wv, cv, sel_idx) -> csr_matrix:
    toks    = [preprocess(t) for t in texts]
    joined  = [' '.join(t) for t in toks]
    Xw      = wv.transform(joined)
    Xc      = cv.transform(joined)
    scaler  = joblib.load(os.path.join(config.MODELS_DIR, 'extra_scaler.pkl'))
    dense   = csr_matrix(scaler.transform([np.hstack([fake_pos_counts(t), lexicon_feats(t)]) 
                                           for t in toks]))
    return hstack([Xw, Xc, dense])[:, sel_idx].tocsr().copy()

# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # reproducibility
    random.seed(42)
    np.random.seed(42)

    # 1️⃣ Load & clean
    df     = clean_df(pd.read_csv(config.TRAIN_FILE))
    texts  = df['Data'].astype(str).tolist()
    y      = df[config.LABELS].values

    # 2️⃣ Oversample & shuffle
    texts, y = oversample_fear(texts, y)
    texts, y = random_oversample_texts(texts, y, random_state=42)
    texts, y = shuffle(texts, y, random_state=42)

    # 3️⃣ Tokenise & TF–IDF
    tok_lists = [preprocess(t) for t in texts]
    joined    = [' '.join(t) for t in tok_lists]
    wv        = TfidfVectorizer(**config.TFIDF_VECT_PARAMS, tokenizer=str.split)
    cv        = TfidfVectorizer(**config.CHAR_VECT_PARAMS)
    Xw, Xc    = wv.fit_transform(joined), cv.fit_transform(joined)
    Xd        = build_dense(tok_lists)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump({'word_vect': wv, 'char_vect': cv}, config.VECTORIZERS_PKL)

    # 4️⃣ χ² top-k feature selection
    X_full  = hstack([Xw, Xc, Xd])
    k       = config.FEATURE_SELECTION['k_best']
    chi     = sum(chi2(X_full, y[:, i])[0] for i in range(y.shape[1]))
    sel_idx = np.argsort(chi)[-k:]
    X_sel   = X_full[:, sel_idx].tocsr().copy()
    joblib.dump(sel_idx, os.path.join(config.MODELS_DIR, 'selector_idx.pkl'))

    # 5️⃣ Define & tune models (with checkpoint resume)
    model_classes = {
        'LogisticRegression': LogisticRegression,
        'RandomForest'     : RandomForestClassifier,
        'KNeighbors'       : KNeighborsClassifier,
        'MultinomialNB'    : MultinomialNB,
        'SVM'              : LinearSVC,
    }

    best_models, best_params = {}, {}
    for name, Cls in model_classes.items():
        ckpt = os.path.join(config.MODELS_DIR, f"{name}_best.pkl")
        if os.path.exists(ckpt):
            print(f"[✓] {name} checkpoint found — loading")
            best_models[name] = joblib.load(ckpt)
            continue

        print(f"\n=== Tuning {name} ===")
        # small grid for MLP
        if name == 'MLP':
            grid = list(ParameterGrid({
                'hidden_layer_sizes':[ (100,), (200,) ],
                'activation':['relu'],
                'alpha':[1e-3,1e-2],
                'learning_rate_init':[1e-3]
            }))
        else:
            grid = list(ParameterGrid(config.HYPERPARAM_GRIDS.get(name, [{}])))

        best_f1, best_cfg = -1.0, None
        for cfg in tqdm(grid, desc=f"{name} grid", unit="cfg"):
            # instantiate base
            if name == 'SVM':
                base = LinearSVC(C=cfg.get('C',1.0),
                                 class_weight=cfg.get('class_weight'),
                                 random_state=42,
                                 max_iter=5000)
            elif name == 'LogisticRegression':
                base = LogisticRegression(**cfg, random_state=42, max_iter=1000)
            else:
                base = Cls(**cfg)

            clf_cv = MultiOutputClassifier(clone(base), n_jobs=-1)
            kf     = KFold(5, shuffle=True, random_state=42)
            scores = []
            for tr, te in kf.split(X_sel):
                clf_cv.fit(X_sel[tr].copy(), y[tr])
                preds = clf_cv.predict(X_sel[te].copy())
                scores.append(f1_score(y[te], preds, average='macro'))
            m = float(np.mean(scores))
            if m > best_f1:
                best_f1, best_cfg = m, cfg

        # retrain best on all data
        if name == 'SVM':
            base_best = LinearSVC(C=best_cfg.get('C',1.0),
                                  class_weight=best_cfg.get('class_weight'),
                                  random_state=42,
                                  max_iter=5000)
        elif name == 'LogisticRegression':
            base_best = LogisticRegression(**best_cfg, random_state=42, max_iter=1000)
        else:
            base_best = Cls(**best_cfg)

        clf_best = MultiOutputClassifier(clone(base_best), n_jobs=-1)
        clf_best.fit(X_sel.copy(), y)
        print(f"--> {name}: macro-F1 = {best_f1:.4f} | cfg = {best_cfg}")

        joblib.dump(clf_best, ckpt)
        best_models[name], best_params[name] = clf_best, best_cfg

    # 6️⃣ Save all artefacts
    joblib.dump(best_models, config.MODELS_PKL)
    joblib.dump(best_params, config.HYPERPARAMS_PKL)
    with open(config.HYPERPARAMS_PKL.replace('.pkl','.json'), 'w', encoding='utf8') as fp:
        json.dump(best_params, fp, indent=2, ensure_ascii=False)

    # 7️⃣ Quick validation on Val.csv
    print("\n[*] Validation on Val.csv")
    dv  = clean_df(pd.read_csv(config.VAL_FILE))
    Xv  = vectorise(dv['Data'].astype(str).tolist(), wv, cv, sel_idx)
    yv  = dv[config.LABELS].values
    for n,m in best_models.items():
        print(f"{n}: {f1_score(yv, m.predict(Xv), average='macro'):.4f}")

    print("\n✅ Training complete.")

if __name__ == '__main__':
    main()
