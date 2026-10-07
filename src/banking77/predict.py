"""Predict a category with one of this project's saved pipelines."""

import argparse
from pathlib import Path

import joblib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text must not be empty")
    # Load only artifacts produced by this project: joblib uses pickle internally.
    pipeline = joblib.load(args.model_path)
    print(pipeline.predict([args.text])[0])


if __name__ == "__main__":
    main()
