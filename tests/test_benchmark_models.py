import unittest

import numpy as np
from sklearn.metrics import f1_score

from banking77 import benchmark_models as bm


def fake_run(accuracy=0.9, macro_f1=0.89, fit=1.0, sha="abc", files="f1"):
    return {
        "accuracy": accuracy, "macro_f1": macro_f1, "fit_seconds": fit, "predict_seconds": 0.01,
        "prediction_ms_per_message": 0.007, "dataset_summary_sha256": sha, "settings": {}, "run_id": "r",
        "training_rows": 10, "evaluation_rows": 5,
        "dataset_files_sha256": {"train.csv": files}, "converged": True,
    }


class BenchmarkModelsTest(unittest.TestCase):
    def test_command_names_the_split_it_was_run_with(self):
        self.assertEqual(bm.command_line("validation"), "python -m banking77.benchmark_models")
        self.assertEqual(bm.command_line("test"), "python -m banking77.benchmark_models --split test")

    def test_configs_match_agreed_settings(self):
        by_key = {c.key: c for c in bm.CONFIGS}
        self.assertEqual(by_key["nb_initial"].kwargs, {"alpha": 1.0})
        self.assertEqual(by_key["nb_selected"].kwargs, {"alpha": 0.05})
        self.assertEqual(by_key["lr_initial"].kwargs, {"solver": "lbfgs", "C": 1.0})
        self.assertEqual(by_key["lr_selected"].kwargs, {"solver": "liblinear-ovr", "C": 100.0})
        self.assertEqual(by_key["svm"].kwargs, {"loss": "squared_hinge", "C": 1.0})

    def test_every_run_uses_validation_only(self):
        splits = []

        def recorder(split, **kwargs):
            splits.append(split)
            return fake_run()

        originals = bm.CONFIGS
        try:
            bm.CONFIGS = tuple(
                bm.Config(c.key, c.model, c.stage, recorder, c.kwargs, c.label, c.reported_macro_f1)
                for c in originals
            )
            bm.run_all(repeats=2)
        finally:
            bm.CONFIGS = originals
        self.assertEqual(set(splits), {"validation"})
        self.assertEqual(len(splits), 2 * len(originals))

    def test_run_all_passes_requested_split(self):
        splits = []

        def recorder(split, **kwargs):
            splits.append(split)
            return fake_run()

        originals = bm.CONFIGS
        try:
            bm.CONFIGS = tuple(
                bm.Config(c.key, c.model, c.stage, recorder, c.kwargs, c.label, c.reported_macro_f1)
                for c in originals
            )
            bm.run_all(repeats=1, split="test")
        finally:
            bm.CONFIGS = originals
        self.assertEqual(set(splits), {"test"})

    def test_reported_value_check_applies_to_validation_only(self):
        keys = [c.key for c in bm.CONFIGS]
        rounds = [{k: fake_run(macro_f1=0.8935) for k in keys}]
        self.assertTrue(bm.aggregate(rounds, "validation")["svm"]["matches_reported_macro_f1"])
        self.assertIsNone(bm.aggregate(rounds, "test")["svm"]["matches_reported_macro_f1"])

    def test_aggregate_uses_median_time_and_rejects_changed_scores(self):
        keys = [c.key for c in bm.CONFIGS]
        rounds = [{k: fake_run(fit=t) for k in keys} for t in (3.0, 1.0, 2.0)]
        self.assertEqual(bm.aggregate(rounds)["svm"]["fit_seconds"], 2.0)
        rounds[1]["svm"] = fake_run(macro_f1=0.5)
        with self.assertRaises(ValueError):
            bm.aggregate(rounds)

    def test_aggregate_rejects_different_datasets(self):
        keys = [c.key for c in bm.CONFIGS]
        rounds = [{k: fake_run() for k in keys}]
        rounds[0]["svm"] = fake_run(sha="other")
        with self.assertRaises(ValueError):
            bm.aggregate(rounds)

    def test_aggregate_rejects_different_data_files(self):
        keys = [c.key for c in bm.CONFIGS]
        rounds = [{k: fake_run() for k in keys}]
        rounds[0]["svm"] = fake_run(files="other")
        with self.assertRaises(ValueError):
            bm.aggregate(rounds)

    def test_aggregate_exposes_convergence(self):
        keys = [c.key for c in bm.CONFIGS]
        rounds = [{k: fake_run() for k in keys}]
        rounds[0]["lr_selected"]["converged"] = False
        models = bm.aggregate(rounds)
        self.assertFalse(models["lr_selected"]["converged"])
        self.assertTrue(models["svm"]["converged"])

    def test_nearest_similarity_flags_reordered_copy(self):
        train = ["How is the exchange rate calculated?", "I lost my card yesterday"]
        evaluation = ["How is an exchange rate calculated?", "Where is the nearest ATM"]
        similarity = bm.nearest_train_similarity(train, evaluation)
        self.assertGreater(similarity[0], 0.8)
        self.assertLess(similarity[1], similarity[0])

    def test_subset_scores_use_only_present_categories(self):
        result = bm.subset_scores(["a", "a", "b"], ["a", "b", "b"])
        self.assertEqual(result["categories"], 2)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        expected = f1_score(["a", "a", "b"], ["a", "b", "b"], average="macro")
        self.assertAlmostEqual(result["macro_f1"], expected)

    def test_mcnemar_counts_and_symmetry(self):
        reference = [True, True, False, False, False, True]
        candidate = [True, False, True, True, True, True]
        result = bm.paired_comparison(reference, candidate)
        self.assertEqual(result["reference_only_correct"], 1)
        self.assertEqual(result["candidate_only_correct"], 3)
        swapped = bm.paired_comparison(candidate, reference)
        self.assertAlmostEqual(result["mcnemar_exact_p"], swapped["mcnemar_exact_p"])
        self.assertEqual(bm.paired_comparison([True], [True])["mcnemar_exact_p"], 1.0)

    def test_macro_f1_matches_sklearn_including_absent_classes(self):
        rng = np.random.default_rng(0)
        true = rng.integers(0, 4, 60)
        predicted = rng.integers(0, 4, 60)
        expected = f1_score(true, predicted, labels=list(range(6)), average="macro", zero_division=0)
        self.assertAlmostEqual(bm.macro_f1_codes(true, predicted, 6), expected)

    def test_bootstrap_is_seeded_and_brackets_observed(self):
        rng = np.random.default_rng(1)
        true = rng.integers(0, 5, 200)
        reference = np.where(rng.random(200) < 0.7, true, rng.integers(0, 5, 200))
        candidate = np.where(rng.random(200) < 0.8, true, rng.integers(0, 5, 200))
        first = bm.bootstrap_difference(true, reference, candidate, 5, resamples=200)
        second = bm.bootstrap_difference(true, reference, candidate, 5, resamples=200)
        self.assertEqual(first, second)
        self.assertLessEqual(first["ci95_low"], first["ci95_high"])

    def test_error_examples_only_use_real_rows(self):
        def row(i, true, pred):
            return {"id": f"v{i}", "text": f"msg {i}", "true_label": true, "predicted_label": pred,
                    "correct": str(true == pred)}

        predictions = {
            "nb_selected": [row(1, "a", "b"), row(2, "a", "a"), row(3, "c", "c")],
            "lr_selected": [row(1, "a", "b"), row(2, "a", "b"), row(3, "c", "c")],
            "svm": [row(1, "a", "b"), row(2, "a", "a"), row(3, "c", "c")],
        }
        result = bm.error_examples(predictions)
        self.assertEqual(result["all_models_wrong_count"], 1)
        self.assertEqual(result["all_models_wrong"][0]["id"], "v1")
        self.assertEqual(result["disagreement"][0]["id"], "v2")


if __name__ == "__main__":
    unittest.main()
