#!/usr/bin/env python
"""GPU‐accelerated MLP using PyTorch on TF–IDF + dense features, with tqdm bars."""

import os
import sys
import random
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── project utils ─────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import utils.config as config
from utils.preprocess import preprocess

# <-- pull helpers from your scikit-learn training script ─────────────────
from models.train import clean_df, fake_pos_counts, lexicon_feats

# ── Dataset for PyTorch ───────────────────────────────────────────────────
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ── Vectorise function (NumPy output) ─────────────────────────────────────
def vectorise(texts):
    toks   = [preprocess(t) for t in texts]
    joined = [" ".join(t) for t in toks]

    wv_cv = joblib.load(config.VECTORIZERS_PKL)
    wv, cv = wv_cv['word_vect'], wv_cv['char_vect']
    sel_idx = joblib.load(os.path.join(config.MODELS_DIR, 'selector_idx.pkl'))
    scaler  = joblib.load(os.path.join(config.MODELS_DIR, 'extra_scaler.pkl'))

    Xw = wv.transform(joined)
    Xc = cv.transform(joined)
    dense_feats = scaler.transform([
        np.hstack([fake_pos_counts(t), lexicon_feats(t)]) for t in toks
    ])

    from scipy.sparse import csr_matrix, hstack
    X_full = hstack([Xw, Xc, csr_matrix(dense_feats)])
    X_sel  = X_full[:, sel_idx]
    return X_sel.toarray().astype(np.float32)

# ── MLP model definition (module-level) ──────────────────────────────────
class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, out_dim*2 if False else 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, out_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ── Main training entry point ──────────────────────────────────────────────
def main():
    random.seed(42)
    np.random.seed(42)

    # 1️⃣ Load & clean
    df    = clean_df(pd.read_csv(config.TRAIN_FILE))
    texts = df['Data'].astype(str).tolist()
    y_all = df[config.LABELS].values

    # 2️⃣ Vectorise
    print("[*] Vectorising data…")
    X_all = vectorise(texts)

    # 3️⃣ Split
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    # 4️⃣ DataLoaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train_dl = DataLoader(TabularDataset(X_tr, y_tr), batch_size=256,
                          shuffle=True, num_workers=0, pin_memory=True)
    val_dl   = DataLoader(TabularDataset(X_val, y_val), batch_size=256,
                          shuffle=False, num_workers=0, pin_memory=True)

    # 5️⃣ Setup
    model     = MLP(X_tr.shape[1], y_tr.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # 6️⃣ Train loop
    best_val_f1 = 0.0
    for epoch in range(1, 51):
        print(f"\nEpoch {epoch:2d}/50")
        model.train()
        running_loss = 0.0
        for Xb, yb in tqdm(train_dl, desc='  Train', leave=False):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
        train_loss = running_loss / len(train_dl.dataset)

        # validation
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for Xb, yb in tqdm(val_dl, desc='  Val  ', leave=False):
                Xb = Xb.to(device)
                p  = model(Xb).cpu().numpy()
                all_p.append(p)
                all_t.append(yb.numpy())
        all_p = np.vstack(all_p)
        all_t = np.vstack(all_t)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_t, (all_p>0.5).astype(int), average='macro')

        print(f"  loss={train_loss:.4f}  val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(),
                os.path.join(config.MODELS_DIR, 'mlp_torch_best.pth')
            )

    print(f"\n✅ Done. Best val_macro_f1 = {best_val_f1:.4f}")

if __name__ == '__main__':
    main()
