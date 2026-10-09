"""Teammate 3's TF-IDF + Logistic Regression model."""

import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

# "lbfgs" and "saga" minimize the multinomial (softmax) loss over all 77 classes.
# "liblinear-ovr" trains one binary L2 logistic regression per class (one-vs-rest),
# because liblinear has no multinomial mode.
SOLVERS = ("lbfgs", "saga", "liblinear-ovr")
DEFAULT_MAX_ITER = 2000
SEED = 42


def multiclass_strategy(solver):
    return "one-vs-rest" if solver == "liblinear-ovr" else "multinomial"


def build_logistic_regression(ngram_max=2, C=1.0, solver="lbfgs", max_iter=DEFAULT_MAX_ITER, random_state=SEED):
    if not math.isfinite(C) or C <= 0:
        raise ValueError("C must be finite and positive")
    if ngram_max not in (1, 2):
        raise ValueError("ngram_max must be 1 or 2")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {', '.join(SOLVERS)}")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if solver == "liblinear-ovr":
        classifier = OneVsRestClassifier(
            LogisticRegression(C=C, solver="liblinear", max_iter=max_iter, random_state=random_state)
        )
    else:
        classifier = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=random_state)
    return Pipeline([
        # Same feature settings as the shared Naive Bayes baseline.
        ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=True)),
        ("classifier", classifier),
    ])


def max_iterations(pipeline):
    """Largest optimizer iteration count across the fitted classifier(s)."""
    classifier = pipeline.named_steps["classifier"]
    estimators = getattr(classifier, "estimators_", [classifier])
    return int(max(int(max(estimator.n_iter_)) for estimator in estimators))
