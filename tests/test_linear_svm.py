"""Checks for teammate 4's Linear SVM pipeline and benchmark rules."""

import unittest

from banking77.benchmark_linear_svm import feature_contributions, schedule, select_best_run
from banking77.linear_svm import LOSSES, build_linear_svm

TRAINING = ["lost card stolen", "my stolen card", "card was stolen today",
            "transfer money account", "send money transfer", "money transfer failed",
            "top up failed", "my top up did not work", "top up is not working"]
LABELS = ["card", "card", "card", "transfer", "transfer", "transfer", "top_up", "top_up", "top_up"]


class LinearSvmPipelineTests(unittest.TestCase):
    def test_invalid_settings_are_rejected_before_training(self):
        for C in (0, -1, float("nan"), float("inf")):
            with self.subTest(C=C), self.assertRaises(ValueError):
                build_linear_svm(C=C)
        with self.assertRaises(ValueError):
            build_linear_svm(loss="log_loss")
        with self.assertRaises(ValueError):
            build_linear_svm(ngram_max=3)

    def test_every_loss_trains_and_predicts(self):
        for loss in LOSSES:
            with self.subTest(loss=loss):
                model = build_linear_svm(C=1.0, loss=loss)
                model.fit(TRAINING, LABELS)
                self.assertEqual(model.predict(["stolen card"])[0], "card")
                self.assertEqual(model.predict(["money transfer"])[0], "transfer")

    def test_validation_only_word_does_not_enter_training_vocabulary(self):
        model = build_linear_svm()
        model.fit(TRAINING, LABELS)
        model.predict(["unseenword card"])
        self.assertNotIn("unseenword", model.named_steps["tfidf"].vocabulary_)

    def test_feature_contributions_explain_the_decision_gap(self):
        model = build_linear_svm()
        model.fit(TRAINING, LABELS)
        text = "stolen card transfer"
        classes = list(model.classes_)
        scores = model.decision_function([text])[0]
        classifier = model.named_steps["classifier"]
        gap = scores[classes.index("transfer")] - scores[classes.index("card")]
        intercepts = classifier.intercept_[classes.index("transfer")] - classifier.intercept_[classes.index("card")]
        pushes = feature_contributions(model, text, "card", "transfer", top=10)
        total = sum(item["value"] for item in pushes["toward_predicted"] + pushes["toward_true"])
        self.assertAlmostEqual(total + intercepts, gap, places=10)


class LinearSvmBenchmarkTests(unittest.TestCase):
    def test_default_loss_is_scheduled_first_for_tie_breaks(self):
        self.assertEqual(schedule()[0]["loss"], "squared_hinge")

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
