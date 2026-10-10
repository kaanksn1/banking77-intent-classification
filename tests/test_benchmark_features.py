import unittest

from banking77 import benchmark_features as bf


def run(macro_f1, split="validation", **settings):
    return {"macro_f1": macro_f1, "evaluation_split": split, "settings": settings}


class BenchmarkFeaturesTest(unittest.TestCase):
    def test_selection_prefers_highest_macro_f1_and_first_on_tie(self):
        runs = [run(0.80, alpha=1.0), run(0.85, alpha=0.5), run(0.85, alpha=0.1)]
        self.assertEqual(bf.select_best(runs)["settings"]["alpha"], 0.5)

    def test_selection_rejects_test_split_and_empty_runs(self):
        with self.assertRaises(ValueError):
            bf.select_best([run(0.9, split="test")])
        with self.assertRaises(ValueError):
            bf.select_best([])

    def test_same_setting_matches_only_requested_keys(self):
        item = run(0.9, solver="liblinear-ovr", C=100.0, max_iter=2000)
        self.assertTrue(bf.same_setting(item, {"solver": "liblinear-ovr", "C": 100.0}))
        self.assertFalse(bf.same_setting(item, {"solver": "lbfgs", "C": 100.0}))

    def test_bigram_only_ablation_covers_every_model(self):
        self.assertEqual(set(bf.BUILDERS), set(bf.MODELS))

    def test_grids_use_owners_schedules(self):
        sizes = {model: len(schedule) for model, (_, schedule, _) in bf.MODELS.items()}
        self.assertEqual(sizes, {"Naive Bayes": 5, "Logistic Regression": 15, "Linear SVM": 10})


if __name__ == "__main__":
    unittest.main()
