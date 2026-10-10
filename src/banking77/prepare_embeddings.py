"""Download pinned Stanford GloVe and retain only words from the train split."""

import hashlib
import json
from urllib.request import Request, urlopen
from zipfile import ZipFile

from banking77.data import ROOT, read_records
from banking77.neural_models import build_vocabulary
from banking77.train_neural import sha256, write_json


def main():
    config = json.loads((ROOT / "configs/glove.json").read_text(encoding="utf-8"))
    cache = ROOT / ".cache/glove"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "glove.6B.zip"
    if not archive.exists():
        partial = cache / "glove.6B.part"
        request = Request(config["url"], headers={"User-Agent": "banking77-course-project"})
        with urlopen(request, timeout=60) as response, partial.open("wb") as handle:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(block)
        if sha256(partial) != config["archive_sha256"]:
            raise ValueError("Downloaded GloVe archive checksum mismatch")
        partial.rename(archive)
    if sha256(archive) != config["archive_sha256"]:
        raise ValueError("Cached GloVe archive checksum mismatch")
    vocabulary = build_vocabulary(row["text"] for row in read_records(ROOT / "data/processed/train.csv"))
    output = ROOT / "data/embeddings"
    output.mkdir(parents=True, exist_ok=True)
    subset = output / "glove.6B.100d.train.txt"
    member_hash = hashlib.sha256()
    matched = 0
    with ZipFile(archive) as zipped, zipped.open(config["member"]) as source, subset.open("wb") as target:
        for line in source:
            member_hash.update(line)
            word = line.split(b" ", 1)[0].decode("utf-8")
            if word in vocabulary:
                target.write(line)
                matched += 1
    if member_hash.hexdigest() != config["member_sha256"]:
        raise ValueError("GloVe member checksum mismatch")
    metadata = {**config, "subset_sha256": sha256(subset), "matched_words": matched,
                "vocabulary_size": len(vocabulary),
                "train_sha256": sha256(ROOT / "data/processed/train.csv")}
    write_json(output / "glove.6B.100d.train.metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
