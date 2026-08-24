"""Deterministic identifiers for downstream Factiva and LLM data lineage."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


ARTICLE_ID_SCHEMA = "title_primary_body_v1"
SENTENCE_ID_SCHEMA = "article_cleaned_sentence_v1"


def _require_text(value: Any, field: str) -> str:
    """Return an exact text value and reject missing lineage components."""
    if pd.isna(value):
        raise ValueError(f"Cannot create a stable identifier: {field} is missing")
    return str(value)


def _digest(payload: dict[str, str], prefix: str) -> str:
    """Hash a versioned canonical JSON payload with SHA-256."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}{hashlib.sha256(encoded).hexdigest()}"


def stable_article_id(title: Any, primary_body: Any) -> str:
    """Create an article ID that exactly mirrors the intentional deduplication key."""
    return _digest(
        {
            "schema": ARTICLE_ID_SCHEMA,
            "title": _require_text(title, "title"),
            "primary_body": _require_text(primary_body, "primary_body"),
        },
        "art_v1_",
    )


def stable_sentence_id(article_id: Any, cleaned_sentence: Any) -> str:
    """Create a stable sentence ID within a canonical retained article."""
    return _digest(
        {
            "schema": SENTENCE_ID_SCHEMA,
            "stable_article_id": _require_text(article_id, "stable_article_id"),
            "cleaned_sentence": _require_text(cleaned_sentence, "cleaned_sentence"),
        },
        "sent_v1_",
    )


def add_stable_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic article and sentence IDs without changing row order."""
    required = ["title", "primary_body", "cleaned_sentence"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing stable-ID source columns: {missing}")

    result = df.copy()
    article_keys = result[["title", "primary_body"]].drop_duplicates()
    article_keys["stable_article_id"] = article_keys.apply(
        lambda row: stable_article_id(row["title"], row["primary_body"]),
        axis=1,
    )
    result = result.merge(
        article_keys,
        on=["title", "primary_body"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result["stable_sentence_id"] = result.apply(
        lambda row: stable_sentence_id(
            row["stable_article_id"], row["cleaned_sentence"]
        ),
        axis=1,
    )

    if result["stable_article_id"].isna().any():
        raise ValueError("Stable article ID generation produced missing values")
    if result["stable_sentence_id"].duplicated().any():
        duplicate_count = int(result["stable_sentence_id"].duplicated(keep=False).sum())
        raise ValueError(
            f"Stable sentence IDs are not unique; {duplicate_count} rows are affected"
        )

    return result
