"""Shared project config: paths, label schema, splits, seed."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
LOGS = ROOT / "logs"
for _p in (RAW, PROCESSED, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

IMDB_SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
TWEET_TEST_SIZE = 3000
MAX_WORDS = 128
