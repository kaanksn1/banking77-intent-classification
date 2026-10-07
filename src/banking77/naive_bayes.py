"""The repository owner's TF-IDF + Multinomial Naive Bayes model."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def build_naive_bayes(ngram_max=2, alpha=1.0):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=True)),
        ("classifier", MultinomialNB(alpha=alpha)),
    ])
