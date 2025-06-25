import re
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from .config import STOPWORDS, SUFFIXES

# ── simple Bangla tokeniser ──────────────────────────────────────────────────
def _bangla_tokens(text: str):
    return re.findall(r'[অ-ঔক-হ]+', text)


def preprocess(text: str) -> list[str]:
    """
    Clean Bangla text:
      • Strip, drop HTML/URL
      • Tokenise → Bangla words
      • Lowercase, drop stop-words
      • Light stemming (strip common suffixes)
    """
    text = re.sub(r'<[^>]+>|http\S+|www\.\S+', ' ', text.strip())
    tokens = [t.lower() for t in _bangla_tokens(text)]
    tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS]

    stemmed = []
    for tok in tokens:
        for suf in SUFFIXES:
            if tok.endswith(suf) and len(tok) > len(suf) + 1:
                tok = tok[:-len(suf)]
                break
        stemmed.append(tok)
    return stemmed


def oversample_features(X, y, method: str = 'random', random_state: int = 42):
    """
    Oversample numeric features X and labels y.
    method: 'random' for RandomOverSampler, 'smote' for SMOTE.
    Returns: (X_resampled, y_resampled)
    """
    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif method == 'random':
        sampler = RandomOverSampler(random_state=random_state)
    else:
        raise ValueError("Unknown method for oversampling: choose 'random' or 'smote'")
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res


def random_oversample_texts(texts: list[str], labels: np.ndarray, random_state: int = 42):
    """
    Random duplication oversampling for multi-label text data.
    texts: list of raw text strings
    labels: numpy array of shape (n_samples, n_labels)
    Returns augmented_texts, augmented_labels
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_labels = labels.shape
    # count samples per label
    label_counts = labels.sum(axis=0)
    max_count = label_counts.max()
    oversample_idx = []
    for i in range(n_labels):
        idx = np.where(labels[:, i] == 1)[0]
        if idx.size == 0:
            continue
        n_to_sample = max_count - idx.size
        if n_to_sample <= 0:
            continue
        sampled = rng.choice(idx, size=n_to_sample, replace=True)
        oversample_idx.extend(sampled.tolist())
    if not oversample_idx:
        return texts, labels
    augmented_texts = texts + [texts[i] for i in oversample_idx]
    augmented_labels = np.vstack([labels, labels[oversample_idx]])
    return augmented_texts, augmented_labels
