import copy
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.metrics import f1_score

from banking77 import compare_all_models as cam


class CompareAllModelsTest(unittest.TestCase):
    def test_sixteen_unique_methods_with_committed_outputs(self):
        self.assertEqual(len(cam.METHODS), 16)
        self.assertEqual(len({m.key for m in cam.METHODS}), 16)
        for method in cam.METHODS:
            self.assertTrue((cam.RESULTS / method.directory / "predictions.csv").exists(), method.key)

    def test_key_comparisons_use_known_methods(self):
        for reference, candidate, _ in cam.KEY_COMPARISONS:
            self.assertIn(reference, cam.BY_KEY)
            self.assertIn(candidate, cam.BY_KEY)
            self.assertNotEqual(reference, candidate)

    def test_holm_adjust_is_monotone_and_capped(self):
        adjusted = cam.holm_adjust({"a": 0.001, "b": 0.01, "c": 0.04, "d": 0.5})
        self.assertAlmostEqual(adjusted["a"], 0.004)
        self.assertAlmostEqual(adjusted["b"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.08)
        self.assertAlmostEqual(adjusted["d"], 0.5)
        self.assertTrue(all(0 <= value <= 1 for value in adjusted.values()))
        self.assertEqual(cam.holm_adjust({"a": 0.9, "b": 0.001})["a"], 0.9)

    def test_lookup_flips_direction_for_reversed_pair(self):
        pairs = {"a|b": {"reference_only_correct": 7, "candidate_only_correct": 3}}
        self.assertEqual(cam.lookup(pairs, "a", "b")[:2], (7, 3))
        self.assertEqual(cam.lookup(pairs, "b", "a")[:2], (3, 7))

    def test_pairwise_tests_count_all_pairs(self):
        true = np.array([0, 1, 2, 0, 1, 2])
        predicted = {"x": np.array([0, 1, 2, 0, 1, 1]), "y": np.array([0, 1, 0, 0, 0, 2]),
                     "z": true.copy()}
        result = cam.pairwise_tests(true, predicted)
        self.assertEqual(set(result), {"x|y", "x|z", "y|z"})
        self.assertEqual(result["x|z"]["reference_only_correct"], 0)
        self.assertEqual(result["x|z"]["candidate_only_correct"], 1)

    def test_bootstrap_scores_are_seeded_and_contain_point_estimate(self):
        rng = np.random.default_rng(0)
        true = rng.integers(0, 4, 200)
        noisy = np.where(rng.random(200) < 0.8, true, rng.integers(0, 4, 200))
        first = cam.bootstrap_scores(true, {"m": noisy}, list("abcd"), resamples=200)
        second = cam.bootstrap_scores(true, {"m": noisy}, list("abcd"), resamples=200)
        self.assertEqual(first, second)
        point = f1_score(true, noisy, average="macro")
        low, high = first["m"]
        self.assertLess(low, point)
        self.assertGreater(high, point)

    def test_mismatched_messages_are_rejected(self):
        original = cam.read_rows

        def shifted(directory):
            rows = original(directory)
            if directory == cam.METHODS[1].directory:
                rows = [dict(row) for row in rows]
                rows[0]["text"] = "changed"
            return rows

        with mock.patch.object(cam, "read_rows", shifted):
            with self.assertRaises(ValueError):
                cam.load_predictions(cam.load_sources())

    def test_changed_prediction_is_rejected(self):
        original = cam.read_rows

        def corrupted(directory):
            rows = original(directory)
            if directory == "neural_test/cnn":
                rows = [dict(row) for row in rows]
                rows[0]["predicted_label"] = "card_arrival"
                rows[0]["correct"] = "True"
            return rows

        with mock.patch.object(cam, "read_rows", corrupted):
            with self.assertRaises(ValueError):
                cam.load_predictions(cam.load_sources())

    def test_collectively_shifted_messages_are_rejected_against_official_test(self):
        original = cam.read_rows

        def shifted(directory):
            rows = [dict(row) for row in original(directory)]
            rows[0]["text"] += " changed in every method"
            return rows

        # Agreement between all prediction files cannot establish official-test identity.
        with mock.patch.object(cam, "read_rows", shifted):
            with self.assertRaises(ValueError):
                cam.load_predictions(cam.load_sources())

    def test_flipped_training_overlap_is_rejected_for_each_family(self):
        original = cam.read_rows
        for key in ("nb", "cnn", "bert"):
            directory = cam.BY_KEY[key].directory

            def flipped(name):
                rows = original(name)
                if name == directory:
                    rows = [dict(row) for row in rows]
                    flag = rows[0]["overlaps_training"]
                    rows[0]["overlaps_training"] = "False" if flag == "True" else "True"
                return rows

            with self.subTest(method=key), mock.patch.object(cam, "read_rows", flipped):
                with self.assertRaises(ValueError):
                    cam.load_predictions(cam.load_sources())

    def test_malformed_training_overlap_is_rejected(self):
        original = cam.read_rows
        for value in ("false", "0", ""):

            def malformed(directory):
                rows = original(directory)
                if directory == cam.BY_KEY["nb"].directory:
                    rows = [dict(row) for row in rows]
                    row = next(row for row in rows if row["overlaps_training"] == "False")
                    row["overlaps_training"] = value
                return rows

            with self.subTest(value=value), mock.patch.object(cam, "read_rows", malformed):
                with self.assertRaises(ValueError):
                    cam.load_predictions(cam.load_sources())

    def test_current_prepared_data_hash_must_match_recorded_provenance(self):
        original = cam.sha256
        for filename in ("train.csv", "test.csv", "summary.json", "categories.json"):
            target = cam.ROOT / "data/processed" / filename

            def changed(path):
                return "0" * 64 if Path(path) == target else original(path)

            with self.subTest(file=filename), mock.patch.object(cam, "sha256", changed):
                with self.assertRaises(ValueError):
                    cam.load_predictions(cam.load_sources())

    def test_source_dataset_provenance_must_match_current_data(self):
        for key in ("nb", "cnn"):
            for field in ("dataset_files_sha256", "dataset_summary_sha256"):
                sources = copy.deepcopy(cam.load_sources())
                method = cam.BY_KEY[key]
                record = cam.source_run(sources, method)
                metadata = record if method.source == "classical" else record["metrics"]
                if field == "dataset_files_sha256":
                    metadata[field]["test.csv"] = "0" * 64
                else:
                    metadata[field] = "0" * 64
                with self.subTest(method=key, field=field):
                    with self.assertRaises(ValueError):
                        cam.load_predictions(sources)

    def test_published_dataset_provenance_must_match_current_data(self):
        original = Path.read_text
        for key in ("nb", "cnn", "bert"):
            target = cam.RESULTS / cam.BY_KEY[key].directory / "metrics.json"
            for field in ("dataset_files_sha256", "dataset_summary_sha256"):

                def changed(path, *args, **kwargs):
                    text = original(path, *args, **kwargs)
                    if path == target:
                        metadata = json.loads(text)
                        if field == "dataset_files_sha256":
                            metadata[field]["test.csv"] = "0" * 64
                        else:
                            metadata[field] = "0" * 64
                        return json.dumps(metadata)
                    return text

                with self.subTest(method=key, field=field), mock.patch.object(Path, "read_text", changed):
                    with self.assertRaises(ValueError):
                        cam.load_predictions(cam.load_sources())

    def test_source_scores_and_settings_must_match_predictions(self):
        for field in ("accuracy", "settings"):
            sources = copy.deepcopy(cam.load_sources())
            metadata = cam.source_run(sources, cam.BY_KEY["cnn"])["metrics"]
            if field == "accuracy":
                metadata[field] += 0.01
            else:
                metadata[field]["seed"] += 1
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    cam.load_predictions(sources)

    def test_classical_source_must_describe_test_split(self):
        sources = copy.deepcopy(cam.load_sources())
        sources["classical"]["evaluation_split"] = "validation"
        with self.assertRaises(ValueError):
            cam.load_predictions(sources)

    def test_classical_prediction_hash_mismatch_is_rejected(self):
        original = cam.sha256
        for key in ("nb", "lr", "svm"):
            target = cam.RESULTS / cam.BY_KEY[key].directory / "predictions.csv"

            def changed(path):
                return "0" * 64 if Path(path) == target else original(path)

            with self.subTest(method=key), mock.patch.object(cam, "sha256", changed):
                with self.assertRaises(ValueError):
                    cam.load_predictions(cam.load_sources())

    def test_summary_matches_recorded_scores(self):
        summary = cam.build_summary()
        self.assertEqual(summary["methods_count"], 16)
        self.assertEqual(summary["evaluation_rows"], 3080)
        self.assertEqual(len(summary["pairwise_mcnemar"]), 120)
        self.assertEqual(summary["methods"][0]["key"], summary["best"])
        scores = [m["macro_f1"] for m in summary["methods"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for check in summary["integrity_checks"]:
            self.assertEqual(check["rows"], 3080)
        ablation = summary["ablation"]
        self.assertEqual(
            ablation["both_correct"] + ablation["only_nb_correct"]
            + ablation["only_cnn_correct"] + ablation["both_wrong"], 3080)


if __name__ == "__main__":
    unittest.main()
