"""Checks for accidental data leakage and model input handling."""

import unittest

from banking77.data import clean_training_records, normalize_text, split_training_records
from banking77.naive_bayes import build_naive_bayes
from banking77.benchmark_naive_bayes import select_best_run


class DataProtocolTests(unittest.TestCase):
    def test_duplicates_and_blank_rows_are_removed(self):
        rows = [
            {"text": "My Card", "category": "card"},
            {"text": "  my   card ", "category": "card"},
            {"text": " ", "category": "card"},
        ]
        clean, stats = clean_training_records(rows, ["card"])
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean[0]["text"], "My Card")
        self.assertEqual(stats["duplicate_training_rows_removed"], 1)
        self.assertEqual(stats["blank_training_rows_removed"], 1)

    def test_conflicting_duplicate_labels_are_rejected(self):
        rows = [{"text": "My Card", "category": "a"}, {"text": "my card", "category": "b"}]
        with self.assertRaises(ValueError):
            clean_training_records(rows, ["a", "b"])

    def test_split_is_deterministic_disjoint_and_contains_all_classes(self):
        rows = [{"id": f"{label}-{i}", "text": f"{label} message {i}", "category": label} for label in ("a", "b", "c") for i in range(20)]
        train, validation = split_training_records(rows, 0.2, 42)
        self.assertEqual((train, validation), split_training_records(rows, 0.2, 42))
        self.assertFalse({normalize_text(row["text"]) for row in train} & {normalize_text(row["text"]) for row in validation})
        self.assertEqual({row["category"] for row in train}, {"a", "b", "c"})
        self.assertEqual({row["category"] for row in validation}, {"a", "b", "c"})


class PipelineTests(unittest.TestCase):
    def test_invalid_alpha_is_rejected_before_training(self):
        for alpha in (0, -1, float("nan"), float("inf")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                build_naive_bayes(alpha=alpha)

    def test_validation_only_word_does_not_enter_training_vocabulary(self):
        training = ["lost card stolen", "my stolen card", "transfer money account", "send money transfer"]
        labels = ["card", "card", "transfer", "transfer"]
        model = build_naive_bayes()
        model.fit(training, labels)
        model.predict(["unseenword card"])
        self.assertNotIn("unseenword", model.named_steps["tfidf"].vocabulary_)
        self.assertEqual(model.predict(["stolen card"])[0], "card")


class BenchmarkSelectionTests(unittest.TestCase):
    def test_macro_f1_decides_selection_even_when_accuracy_is_lower(self):
        runs = [
            {"evaluation_split": "validation", "macro_f1": 0.7, "accuracy": 0.95},
            {"evaluation_split": "validation", "macro_f1": 0.8, "accuracy": 0.90},
        ]
        self.assertIs(select_best_run(runs), runs[1])

    def test_test_results_cannot_be_used_for_selection(self):
        with self.assertRaises(ValueError):
            select_best_run([{"evaluation_split": "test", "macro_f1": 1.0}])


if __name__ == "__main__":
    unittest.main()
