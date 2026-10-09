"""Teammate 4's TF-IDF + Linear SVM model."""

import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# "squared_hinge" is the scikit-learn LinearSVC default; "hinge" is the classic
# soft-margin SVM loss. Both train one binary L2-regularized SVM per class
# (one-vs-rest) with liblinear's dual coordinate descent solver.
LOSSES = ("squared_hinge", "hinge")
DEFAULT_MAX_ITER = 10000
SEED = 42


def build_linear_svm(ngram_max=2, C=1.0, loss="squared_hinge", max_iter=DEFAULT_MAX_ITER, random_state=SEED):
    if not math.isfinite(C) or C <= 0:
        raise ValueError("C must be finite and positive")
    if ngram_max not in (1, 2):
        raise ValueError("ngram_max must be 1 or 2")
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {', '.join(LOSSES)}")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    return Pipeline([
        # Same feature settings as the shared Naive Bayes and Logistic Regression runs.
        ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=True)),
        # dual=True: fewer training messages than TF-IDF features, and "hinge" requires it.
        ("classifier", LinearSVC(C=C, loss=loss, dual=True, max_iter=max_iter, random_state=random_state)),
    ])


def optimizer_iterations(pipeline):
    """Largest liblinear iteration count across the 77 one-vs-rest classifiers."""
    return int(pipeline.named_steps["classifier"].n_iter_)
