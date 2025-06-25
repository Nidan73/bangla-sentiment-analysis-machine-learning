import os

# ── Project directories -----------------------------------------------------
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ── Dataset paths -----------------------------------------------------------
TRAIN_FILE      = os.path.join(DATA_DIR, 'Train.csv')
VAL_FILE        = os.path.join(DATA_DIR, 'Val.csv')
TEST_FILE       = os.path.join(DATA_DIR, 'Test.csv')

# ── Serialized artefacts ----------------------------------------------------
VECTORIZERS_PKL = os.path.join(MODELS_DIR, 'vectorizers.pkl')
MODELS_PKL      = os.path.join(MODELS_DIR, 'models.pkl')
HYPERPARAMS_PKL = os.path.join(MODELS_DIR, 'hyperparams.pkl')
THRESHOLDS_JSON = os.path.join(MODELS_DIR, 'thresholds.json')

# ── Label order (keep consistent everywhere) --------------------------------
LABELS = ['Love', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear']

# ── Pre-processing helpers --------------------------------------------------
STOPWORDS = {
    "এবং","কিন্তু","তবে","যে","এটা","সেটা","এই","ঐ","এটাই","সেই",
    "থেকে","পর্যন্ত","জন্য","আর","ও","এর","এদের","ছিল","ছিলাম","ছিলে",
    "করা","হয়ে","হচ্ছে","হতে","হয়","করছেন","করে","করেছিল","করবো",
    "ছিলো","ছিলেন","আমি","তুমি","সে","তারা","আমরা","আমাদের","তোমাদের",
    "হয়েছে","হবে","আছে","করেন","করেও","প্রয়োগ","দেখা","পেয়েছে"
}
SUFFIXES = ["র","রা","দের","গুলি","গুলো","টা","টি"]

# ── TF-IDF vectoriser parameters -------------------------------------------
TFIDF_VECT_PARAMS = {
    'ngram_range' : (1, 3),
    'min_df'      : 5,
    'max_df'      : 0.8,
    'token_pattern': None   # disable regex because we pass a tokenizer
}
CHAR_VECT_PARAMS = {
    'analyzer'    : 'char',
    'ngram_range' : (3, 5),
    'max_features': 5000    # no tokenizer here
}

# ── Sentiment lexicon (± polarity) -----------------------------------------
LEXICON = {
    "ভালো": 0.85, "আনন্দ": 0.90, "উল্লাস": 0.75,
    "দুঃখ": -0.80, "বিষন্ন": -0.70, "ব্যথা": -0.65,
    "রাগ": -0.90, "প্রতিবাদ": -0.75, "ভয়": -0.85,
    "আতঙ্ক": -0.80, "অবাক": 0.65, "বিস্মিত": 0.70,
    "ভালোবাসা": 0.95, "প্রেম": 0.90
}

# ── Hyper-parameter grids (5-fold CV) --------------------------------------
HYPERPARAM_GRIDS = {
    'SVM': {
        'C': [0.1, 1],
        'class_weight': ['balanced']
    },
    'LogisticRegression': {
        'C'           : [0.01, 0.1, 1, 10],
        'penalty'     : ['l2'],
        'solver'      : ['liblinear', 'saga'],
        'class_weight': ['balanced', None]
    },
    'RandomForest': {
        'n_estimators'     : [200, 350],
        'max_depth'        : [None, 80],
        'min_samples_leaf' : [1, 5],
        'class_weight'     : ['balanced', 'balanced_subsample']
    },
    'KNeighbors': {
        'n_neighbors': [3, 5, 7, 9],
        'weights'    : ['uniform', 'distance']
    },
    'MultinomialNB': {
        'alpha'     : [0.01, 0.05, 0.1, 0.3, 0.7, 1.0, 2.0, 5.0, 10.0],
        'fit_prior' : [True, False]
    },
    # ── New MLP grid ─────────────────────────────────────────────────────────
    'MLP': {
        'hidden_layer_sizes': [(100,), (200,), (100, 50)],
        'activation'        : ['relu', 'tanh'],
        'alpha'             : [1e-4, 1e-3, 1e-2],
        'learning_rate_init': [1e-3, 1e-4]
    }
}

# ── Default per-label thresholds (overwritten after calibration) ------------
DEFAULT_THRESHOLDS = {lbl: 0.5 for lbl in LABELS}

# ── Feature selection -------------------------------------------------------
FEATURE_SELECTION = {
    'method' : 'chi2',
    'k_best' : 10_000         # top-k after hstacking TF-IDF + dense extras
}
