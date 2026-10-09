"""Checks for teammate 3's Logistic Regression pipeline and benchmark rules."""

import unittest

from banking77.benchmark_logistic_regression import schedule, select_best_run
from banking77.logistic_regression import SOLVERS, build_logistic_regression

TRAINING = ["lost card stolen", "my stolen card", "card was stolen today",
            "transfer money account", "send money transfer", "money transfer failed"]
LABELS = ["card", "card", "card", "transfer", "transfer", "transfer"]


class LogisticRegressionPipelineTests(unittest.TestCase):
    def test_invalid_settings_are_rejected_before_training(self):
        for C in (0, -1, float("nan"), float("inf")):
            with self.subTest(C=C), self.assertRaises(ValueError):
                build_logistic_regression(C=C)
        with self.assertRaises(ValueError):
            build_logistic_regression(solver="liblinear")
        with self.assertRaises(ValueError):
            build_logistic_regression(ngram_max=3)

    def test_every_solver_trains_and_predicts(self):
        for solver in SOLVERS:
            with self.subTest(solver=solver):
                model = build_logistic_regression(C=10.0, solver=solver)
                model.fit(TRAINING, LABELS)
                self.assertEqual(model.predict(["stolen card"])[0], "card")
                self.assertEqual(model.predict(["money transfer"])[0], "transfer")

    def test_validation_only_word_does_not_enter_training_vocabulary(self):
        model = build_logistic_regression()
        model.fit(TRAINING, LABELS)
        model.predict(["unseenword card"])
        self.assertNotIn("unseenword", model.named_steps["tfidf"].vocabulary_)


class LogisticRegressionBenchmarkTests(unittest.TestCase):
    def test_default_solver_is_scheduled_first_for_tie_breaks(self):
        self.assertEqual(schedule()[0]["solver"], "lbfgs")

    def test_macro_f1_decides_selection_and_ties_keep_first(self):
        runs = [
            {"evaluation_split": "validation", "macro_f1": 0.8, "accuracy": 0.95},
            {"evaluation_split": "validation", "macro_f1": 0.8, "accuracy": 0.99},
            {"evaluation_split": "validation", "macro_f1": 0.7, "accuracy": 0.99},
        ]
        self.assertIs(select_best_run(runs), runs[0])

    def test_test_results_cannot_be_used_for_selection(self):
        with self.assertRaises(ValueError):
            select_best_run([{"evaluation_split": "test", "macro_f1": 1.0}])


if __name__ == "__main__":
    unittest.main()
