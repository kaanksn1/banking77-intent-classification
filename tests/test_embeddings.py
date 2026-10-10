"""Check train-only embeddings, subword inference, and saved-model inference."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

HAS_EMBEDDINGS = all(find_spec(name) is not None for name in ("torch", "gensim"))
if HAS_EMBEDDINGS:
    import joblib
    import numpy as np
    from banking77.train_embeddings import MeanEmbeddingClassifier


@unittest.skipUnless(HAS_EMBEDDINGS, "Install optional neural requirements")
class EmbeddingTests(unittest.TestCase):
    texts = ["card payment card", "card payment", "cash withdrawal cash", "cash withdrawal"]
    labels = ["card", "card", "cash", "cash"]

    def test_heldout_vocabulary_is_not_learned_and_fasttext_handles_subwords(self):
        for kind in ("word2vec_cbow", "word2vec_skipgram", "fasttext"):
            with self.subTest(kind=kind):
                model = MeanEmbeddingClassifier(kind, epochs=2).fit(self.texts, self.labels)
                vocabulary = dict(model.embedding.wv.key_to_index)
                representation = model.transform(["payments", "   "])
                self.assertEqual(vocabulary, model.embedding.wv.key_to_index)
                self.assertNotIn("payments", vocabulary)
                self.assertTrue(np.isfinite(representation).all())
                np.testing.assert_array_equal(representation[1], np.zeros(100))
                if kind == "fasttext":
                    self.assertGreater(np.linalg.norm(representation[0]), 0)
                else:
                    np.testing.assert_array_equal(representation[0], np.zeros(100))

    def test_seeded_training_and_joblib_roundtrip_preserve_probabilities(self):
        for kind in ("word2vec_cbow", "word2vec_skipgram", "fasttext"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                first = MeanEmbeddingClassifier(kind, epochs=2).fit(self.texts, self.labels)
                second = MeanEmbeddingClassifier(kind, epochs=2).fit(self.texts, self.labels)
                expected = first.predict_proba(["card", "payments", "cash"])
                np.testing.assert_allclose(expected, second.predict_proba(["card", "payments", "cash"]))
                path = Path(directory) / "pipeline.joblib"
                joblib.dump(first, path)
                restored = joblib.load(path)
                np.testing.assert_array_equal(expected, restored.predict_proba(["card", "payments", "cash"]))
                self.assertEqual(first.classes_.tolist(), restored.classes_.tolist())


if __name__ == "__main__":
    unittest.main()
