"""Initial classical baselines; each model owner can extend its experiments."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

MODEL_NAMES = ("naive_bayes", "logistic_regression", "svm")


def build_model(name, ngram_max=2, alpha=1.0, c=1.0, seed=42):
    classifiers = {
        "naive_bayes": lambda: MultinomialNB(alpha=alpha),
        "logistic_regression": lambda: LogisticRegression(C=c, max_iter=2000, random_state=seed),
        "svm": lambda: LinearSVC(C=c, random_state=seed),
    }
    if name not in classifiers:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=True)),
        ("classifier", classifiers[name]()),
    ])
