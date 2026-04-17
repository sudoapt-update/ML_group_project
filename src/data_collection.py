"""Task 1 — Data Collection (Imran Iskakov).

Downloads the source-domain IMDb Large Movie Review dataset and the
target-domain TweetEval Sentiment subset via HuggingFace Datasets, and
writes raw CSVs to data/raw/ with a unified schema (text, label, source).

IMDb note: the released dataset is binary (pos/neg); reviews with
ratings 5-6 were excluded by the original authors, so no "neutral" class
is available from this source. We keep labels binary here and map to the
shared 3-class schema in the cleaning step (positive=2, negative=0).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import pandas as pd
from datasets import load_dataset

from config import LOGS, RAW, SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOGS / "collect.log"), logging.StreamHandler()],
)
log = logging.getLogger("collect")


def collect_imdb() -> pd.DataFrame:
    log.info("Loading IMDb (stanfordnlp/imdb) ...")
    ds = load_dataset("stanfordnlp/imdb")
    frames = []
    for split in ("train", "test"):
        df = ds[split].to_pandas()
        df["orig_split"] = split
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    # HF label convention: 0=neg, 1=pos. Convert to string for clarity.
    full["label_str"] = full["label"].map({0: "negative", 1: "positive"})
    full["source"] = "imdb"
    out = RAW / "imdb_raw.csv"
    full[["text", "label_str", "orig_split", "source"]].to_csv(out, index=False)
    log.info("IMDb rows=%d -> %s", len(full), out)
    log.info("IMDb label distribution:\n%s", full["label_str"].value_counts().to_string())
    return full


def collect_tweeteval() -> pd.DataFrame:
    log.info("Loading TweetEval sentiment (cardiffnlp/tweet_eval) ...")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    frames = []
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas()
        df["orig_split"] = split
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    # TweetEval convention: 0=neg, 1=neu, 2=pos.
    full["label_str"] = full["label"].map({0: "negative", 1: "neutral", 2: "positive"})
    full["source"] = "tweeteval"
    out = RAW / "tweeteval_raw.csv"
    full[["text", "label_str", "orig_split", "source"]].to_csv(out, index=False)
    log.info("TweetEval rows=%d -> %s", len(full), out)
    log.info("TweetEval label distribution:\n%s", full["label_str"].value_counts().to_string())
    log.info("TweetEval split sizes:\n%s", full["orig_split"].value_counts().to_string())
    return full


def write_manifest(imdb: pd.DataFrame, tweets: pd.DataFrame) -> None:
    manifest = {
        "collected_at": datetime.utcnow().isoformat() + "Z",
        "seed": SEED,
        "imdb": {
            "source": "huggingface:stanfordnlp/imdb",
            "rows": int(len(imdb)),
            "label_counts": imdb["label_str"].value_counts().to_dict(),
            "note": "Binary (pos/neg) only — no neutral in released IMDb.",
        },
        "tweeteval_sentiment": {
            "source": "huggingface:cardiffnlp/tweet_eval (sentiment)",
            "rows": int(len(tweets)),
            "label_counts": tweets["label_str"].value_counts().to_dict(),
            "split_counts": tweets["orig_split"].value_counts().to_dict(),
        },
    }
    out = RAW / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest -> %s", out)


def main() -> None:
    imdb = collect_imdb()
    tweets = collect_tweeteval()
    write_manifest(imdb, tweets)
    log.info("Data collection complete.")


if __name__ == "__main__":
    main()
