"""Task 2 — Data Cleaning (Ivan Bychuk).

Reads data/raw/{imdb_raw,tweeteval_raw}.csv and produces cleaned,
split, and class-balanced datasets in data/processed/ with the shared
3-class label schema (negative=0, neutral=1, positive=2).

Preprocessing (per proposal):
  - lowercase
  - strip URLs, @mentions, whole #hashtag tokens
  - convert emojis to text via emoji.demojize (": smiling_face :" style)
  - collapse whitespace, drop HTML <br /> artefacts
  - truncate to MAX_WORDS whitespace tokens (tokenizer truncation to
    128 sub-word tokens happens at training time)
  - class balance by downsampling majority classes

Splits:
  IMDb: 70 / 15 / 15 train/val/test, stratified on label.
  TweetEval: 3000 stratified 'final_test'; remainder is the
  'unlabelled_pool' for adaptation (labels retained in a separate file
  for diagnostic use only — must NOT be read during training/adaptation).
"""
from __future__ import annotations

import logging
import re

import emoji
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ID2LABEL,
    LABEL2ID,
    LOGS,
    MAX_WORDS,
    PROCESSED,
    RAW,
    SEED,
    TWEET_TEST_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOGS / "clean.log"), logging.StreamHandler()],
)
log = logging.getLogger("clean")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
HTML_BR_RE = re.compile(r"<br\s*/?>")
WS_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    x = HTML_BR_RE.sub(" ", raw)
    x = URL_RE.sub(" ", x)
    x = MENTION_RE.sub(" ", x)
    x = HASHTAG_RE.sub(" ", x)
    x = emoji.demojize(x, delimiters=(" :", ": "))
    x = x.lower()
    x = WS_RE.sub(" ", x).strip()
    words = x.split(" ")
    if len(words) > MAX_WORDS:
        x = " ".join(words[:MAX_WORDS])
    return x


def _apply_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_clean"] = df["text"].map(clean_text)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    df["label"] = df["label_str"].map(LABEL2ID).astype(int)
    return df


def _balance(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Downsample every class to the smallest class size."""
    min_n = df[label_col].value_counts().min()
    parts = [
        g.sample(n=min_n, random_state=SEED)
        for _, g in df.groupby(label_col)
    ]
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def clean_imdb() -> None:
    src = RAW / "imdb_raw.csv"
    log.info("Reading %s", src)
    df = pd.read_csv(src)
    df = _apply_clean(df)
    log.info("IMDb after cleaning: %d rows. Label dist:\n%s",
             len(df), df["label_str"].value_counts().to_string())

    # Binary (no neutral in IMDb). Balance the two classes, then split.
    df = _balance(df)
    log.info("IMDb after class balance: %d rows.", len(df))

    # 70/15/15 stratified
    train, tmp = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["label"]
    )
    val, test = train_test_split(
        tmp, test_size=0.50, random_state=SEED, stratify=tmp["label"]
    )
    for name, part in (("train", train), ("val", val), ("test", test)):
        out = PROCESSED / f"imdb_{name}.csv"
        part[["text_clean", "label", "label_str", "source"]].to_csv(out, index=False)
        log.info("IMDb %s -> %s (%d rows)", name, out, len(part))


def clean_tweeteval() -> None:
    src = RAW / "tweeteval_raw.csv"
    log.info("Reading %s", src)
    df = pd.read_csv(src)

    # Per proposal, treat the TweetEval *test* split (~12k) as the target
    # domain pool: 3k stratified final_test, remainder is unlabelled pool.
    df = df[df["orig_split"] == "test"].reset_index(drop=True)
    df = _apply_clean(df)
    log.info("TweetEval test-split after cleaning: %d rows. Label dist:\n%s",
             len(df), df["label_str"].value_counts().to_string())

    final_test, pool = train_test_split(
        df,
        train_size=TWEET_TEST_SIZE,
        random_state=SEED,
        stratify=df["label"],
    )

    # Final test: keep labels. Balance classes for fair per-class F1.
    final_test_balanced = _balance(final_test)
    out = PROCESSED / "tweet_final_test.csv"
    final_test_balanced[["text_clean", "label", "label_str", "source"]].to_csv(out, index=False)
    log.info("Tweet final_test (balanced) -> %s (%d rows)", out, len(final_test_balanced))

    # Unlabelled adaptation pool: store text only.
    pool_out = PROCESSED / "tweet_unlabelled_pool.csv"
    pool[["text_clean", "source"]].to_csv(pool_out, index=False)
    log.info("Tweet unlabelled_pool -> %s (%d rows, labels stripped)", pool_out, len(pool))

    # Diagnostic-only label file for the pool (never used in training).
    diag = PROCESSED / "_tweet_pool_labels_DIAGNOSTIC_ONLY.csv"
    pool[["label", "label_str"]].to_csv(diag, index=False)
    log.info("Pool labels (diagnostic only) -> %s", diag)


def write_schema_note() -> None:
    note = f"""# Processed data schema

Shared label schema (both domains):
{LABEL2ID}

Files:
  imdb_train.csv / imdb_val.csv / imdb_test.csv
    columns: text_clean, label, label_str, source
    labels: positive/negative only (IMDb has no neutral).

  tweet_final_test.csv
    columns: text_clean, label, label_str, source
    3-class, class-balanced, used for zero-shot & post-adaptation eval.

  tweet_unlabelled_pool.csv
    columns: text_clean, source
    Used by CORAL and self-training. Labels deliberately stripped.

  _tweet_pool_labels_DIAGNOSTIC_ONLY.csv
    Held out for post-hoc diagnostics. Must NOT be read by training
    or adaptation code. Aligned row-for-row with the pool file.

Preprocessing applied:
  - HTML <br> stripped; URLs / @mentions / #hashtags removed
  - emojis demojized to ": name :"
  - lowercased, whitespace collapsed
  - truncated to {MAX_WORDS} whitespace tokens
  - classes balanced by downsampling to minority class size
"""
    (PROCESSED / "README.md").write_text(note)


def main() -> None:
    clean_imdb()
    clean_tweeteval()
    write_schema_note()
    log.info("Data cleaning complete. ID2LABEL=%s", ID2LABEL)


if __name__ == "__main__":
    main()
