"""Optional neural checks; the classical-only installation remains supported."""

from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from banking77.neural_models import WORD_MODELS, WordClassifier, build_vocabulary, encode
    from banking77 import train_neural as training


@unittest.skipUnless(HAS_TORCH, "Install optional neural requirements")
class NeuralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_heldout_words_are_unknown_and_empty_text_is_nonempty(self):
        vocabulary = build_vocabulary(["my card", "card payment"])
        self.assertNotIn("heldout", vocabulary)
        self.assertEqual(encode("heldout", vocabulary, 8), [1])
        self.assertEqual(encode("   ", vocabulary, 8), [1])

    def test_padding_and_batch_order_do_not_change_predictions(self):
        weights = torch.randn(10, 8)
        weights[0].zero_()
        for kind in WORD_MODELS:
            with self.subTest(model=kind):
                model = WordClassifier(kind, weights.clone(), 3, hidden_size=6, dropout=0).eval()
                ids = torch.tensor([[2, 3, 0, 0, 0], [4, 5, 6, 7, 8]])
                baseline = model(ids, ids.ne(0).long())
                padded = torch.nn.functional.pad(ids, (0, 7))
                torch.testing.assert_close(baseline, model(padded, padded.ne(0).long()))
                torch.testing.assert_close(baseline, model(ids.flip(0), ids.flip(0).ne(0).long()).flip(0))

    def test_all_word_architectures_learn_a_small_separable_problem(self):
        ids = torch.tensor([[2, 2, 0, 0, 0], [3, 3, 3, 0, 0]] * 4)
        labels = torch.tensor([0, 1] * 4)
        for kind in WORD_MODELS:
            with self.subTest(model=kind):
                weights = torch.randn(5, 8)
                weights[0].zero_()
                model = WordClassifier(kind, weights, 2, hidden_size=8, dropout=0)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
                for _ in range(25):
                    optimizer.zero_grad()
                    logits = model(ids, ids.ne(0).long())
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    optimizer.step()
                self.assertEqual(model(ids, ids.ne(0).long()).argmax(1).tolist(), labels.tolist())

    def test_embedding_file_dimension_and_nonfinite_values_are_rejected(self):
        from banking77.neural_models import embedding_weights
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.txt"
            for content in ("card 1 2\n", "card 1 nan 3\n"):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    embedding_weights({"<PAD>": 0, "<UNK>": 1, "card": 2}, 3, path)

    def test_freeze_rejects_test_runs_and_development_subsets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results/runs/example"
            output.mkdir(parents=True)
            for split, subset in (("test", False), ("validation", True)):
                (output / "metrics.json").write_text(json.dumps({"evaluation_split": split, "development_subset": subset}), encoding="utf-8")
                with patch.object(training, "ROOT", root), self.assertRaises(ValueError):
                    training.freeze("example", root / "results/protocol.json")

    def test_uncommitted_protocol_is_rejected_before_reading_test_data(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text("{}", encoding="utf-8")
            with patch.object(training, "ROOT", root), patch.object(training.subprocess, "run") as git, patch.object(training, "read_records") as reader:
                git.return_value.stdout = b'{"different":true}'
                with self.assertRaisesRegex(ValueError, "committed"):
                    training.test(protocol, "cpu")
                reader.assert_not_called()

    @unittest.skipUnless(find_spec("transformers") is not None, "Install transformers")
    def test_transformer_classification_head_has_77_outputs_and_gradients(self):
        from transformers import BertConfig, BertForSequenceClassification
        model = BertForSequenceClassification(BertConfig(vocab_size=20, hidden_size=16,
                    num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=77))
        logits = model(input_ids=torch.tensor([[2, 3, 4, 0]]), attention_mask=torch.tensor([[1, 1, 1, 0]])).logits
        self.assertEqual(tuple(logits.shape), (1, 77))
        torch.nn.functional.cross_entropy(logits, torch.tensor([76])).backward()
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
