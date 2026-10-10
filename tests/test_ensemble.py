"""Soft-voting selection and alignment must not use test outcomes."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import numpy as np
    from banking77 import train_ensemble as ensemble


@unittest.skipUnless(HAS_TORCH, "Install optional neural requirements")
class EnsembleTests(unittest.TestCase):
    def test_complementary_errors_can_improve_and_grid_contains_both_ablations(self):
        nb = np.array([[.7, .3], [.45, .55], [.1, .9], [.4, .6]])
        cnn = np.array([[.45, .55], [.7, .3], [.4, .6], [.55, .45]])
        selected, candidates = ensemble.select_weight(["a", "a", "b", "b"], ["a", "b"], nb, cnn)
        self.assertEqual(len(candidates), 11)
        self.assertEqual([candidates[0]["cnn_weight"], candidates[-1]["cnn_weight"]], [0, 1])
        self.assertEqual(selected["macro_f1"], 1)
        self.assertGreater(selected["macro_f1"], candidates[0]["macro_f1"])
        self.assertGreater(selected["macro_f1"], candidates[-1]["macro_f1"])

    def test_ties_keep_first_weight_and_accuracy_does_not_override_macro_f1(self):
        with patch.object(ensemble.neural, "score", side_effect=[
                {"accuracy": .99, "macro_f1": .5},
                {"accuracy": .8, "macro_f1": .7}] + [{"accuracy": .9, "macro_f1": .7}] * 9):
            selected, _ = ensemble.select_weight(["a"], ["a", "b"], np.array([[.6, .4]]), np.array([[.7, .3]]))
        self.assertEqual(selected["cnn_weight"], .1)

    def test_misaligned_or_invalid_probabilities_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "probabilities.npz"
            for ids, categories, probabilities in ((["wrong"], ["a", "b"], [[.5, .5]]),
                    (["id"], ["b", "a"], [[.5, .5]]), (["id"], ["a", "b"], [[.9, .9]]),
                    (["id"], ["a", "b"], [[float("nan"), .5]])):
                np.savez(path, ids=ids, categories=categories, probabilities=probabilities)
                with self.assertRaises(ValueError):
                    ensemble.aligned_probabilities(path, [{"id": "id"}], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
