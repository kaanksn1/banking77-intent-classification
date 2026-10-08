"""The repository owner's TF-IDF + Multinomial Naive Bayes model."""

import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def build_naive_bayes(ngram_max=2, alpha=1.0):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if ngram_max not in (1, 2):
        raise ValueError("ngram_max must be 1 or 2")
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=True)),
        ("classifier", MultinomialNB(alpha=alpha)),
    ])
