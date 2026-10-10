"""Word-level course baselines; padding must never change a prediction."""

from collections import Counter
import hashlib
from pathlib import Path
import re
import unicodedata

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

WORD_MODELS = ("mean", "cnn", "rnn", "lstm", "bilstm")


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFKC", text).casefold())


def build_vocabulary(texts, max_words=20000):
    if max_words < 2:
        raise ValueError("max_words must include PAD and UNK")
    counts = Counter(word for text in texts for word in tokenize(text))
    words = sorted(counts, key=lambda word: (-counts[word], word))[:max_words - 2]
    return {"<PAD>": 0, "<UNK>": 1, **{word: i + 2 for i, word in enumerate(words)}}


def encode(text, vocabulary, max_length):
    if max_length < 1:
        raise ValueError("max_length must be positive")
    return [vocabulary.get(word, 1) for word in tokenize(text)[:max_length]] or [1]


def embedding_weights(vocabulary, dimension, vector_file=None):
    """Stream a local text embedding file, retaining only training vocabulary."""
    weights = torch.empty(len(vocabulary), dimension).normal_(mean=0, std=0.05)
    weights[0].zero_()
    metadata = {"initialization": "random", "dimension": dimension, "matched_words": 0}
    if vector_file is None:
        return weights, metadata
    path = Path(vector_file)
    digest = hashlib.sha256()
    matched = set()
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            fields = raw.decode("utf-8").rstrip().split()
            if not fields or fields[0] not in vocabulary or vocabulary[fields[0]] < 2:
                continue
            if len(fields) != dimension + 1:
                raise ValueError(f"Embedding dimension mismatch for {fields[0]}")
            values = torch.tensor([float(value) for value in fields[1:]])
            if not torch.isfinite(values).all():
                raise ValueError("Non-finite embedding vector")
            weights[vocabulary[fields[0]]] = values
            matched.add(fields[0])
    if not matched:
        raise ValueError("Embedding file has no matches to the training vocabulary")
    metadata.update(initialization="pretrained_text_vectors", filename=path.name,
                    sha256=digest.hexdigest(), matched_words=len(matched),
                    coverage=len(matched) / max(1, len(vocabulary) - 2))
    return weights, metadata


class WordClassifier(nn.Module):
    def __init__(self, model, weights, classes, hidden_size=128, dropout=0.3,
                 freeze_embeddings=False):
        super().__init__()
        if model not in WORD_MODELS:
            raise ValueError(f"Unknown word model: {model}")
        self.kind = model
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze_embeddings, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        dimension = weights.shape[1]
        if model == "cnn":
            self.kernels = (3, 4, 5)
            self.convolutions = nn.ModuleList(nn.Conv1d(dimension, hidden_size, k) for k in self.kernels)
            output_size = hidden_size * len(self.kernels)
        elif model in ("rnn", "lstm", "bilstm"):
            recurrent = nn.RNN if model == "rnn" else nn.LSTM
            self.recurrent = recurrent(dimension, hidden_size, batch_first=True,
                                       bidirectional=model == "bilstm")
            output_size = hidden_size * (2 if model == "bilstm" else 1)
        else:
            output_size = dimension
        self.classifier = nn.Linear(output_size, classes)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).clamp_min(1)
        if self.kind == "cnn":
            channels = embedded.transpose(1, 2)
            if channels.shape[-1] < max(self.kernels):
                channels = nn.functional.pad(channels, (0, max(self.kernels) - channels.shape[-1]))
            pooled = []
            for kernel, convolution in zip(self.kernels, self.convolutions):
                features = convolution(channels).relu()
                # Short texts get one zero-extended window; all-padding windows are excluded.
                valid = (lengths - kernel + 1).clamp_min(1)
                positions = torch.arange(features.shape[-1], device=features.device)
                features = features.masked_fill(positions[None, None, :] >= valid[:, None, None], -torch.inf)
                pooled.append(features.amax(dim=-1))
            representation = torch.cat(pooled, dim=1)
        elif self.kind in ("rnn", "lstm", "bilstm"):
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, state = self.recurrent(packed)
            hidden = state[0] if isinstance(state, tuple) else state
            representation = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.kind == "bilstm" else hidden[-1]
        else:
            representation = (embedded * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        return self.classifier(self.dropout(representation))
