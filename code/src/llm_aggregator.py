"""Aggregate immutable LLM annotations into observed article and monthly points.

The aggregation stage creates stable article and sentence identifiers, article
party scores, monthly weighted stance means, raw party-month IQR, and monthly
polarization. It does not estimate uncertainty. The downstream statistical
stage adds the validated joint 5,000-replicate article-cluster bootstrap.
"""

from __future__ import annotations

import os
import hashlib
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DATA_PATH
from src.measurement_bootstrap import calculate_point_estimates
from src.stable_ids import ARTICLE_ID_SCHEMA, SENTENCE_ID_SCHEMA, add_stable_ids


class LLMAggregator:
    """Aggregate sentence scores to article-party and observed monthly grains."""

    INPUT_CSV = os.path.join(
        DATA_PATH,
        "pipeline_c",
        "llm_scorer",
        "Factiva_v1_llmrated.csv",
    )
    OUTPUT_DIR = os.path.join(DATA_PATH, "pipeline_c", "llm_aggregated")
    BATCH_LOG = os.path.join(
        DATA_PATH,
        "pipeline_c",
        "llm_scorer",
        "batch_log.json",
    )
    DIMENSIONS = ("q1", "q2", "q3", "q4")

    def __init__(
        self,
        input_csv: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.INPUT_CSV = input_csv or self.INPUT_CSV
        self.OUTPUT_DIR = output_dir or self.OUTPUT_DIR
        self.ARTICLE_SCORES_CSV = os.path.join(
            self.OUTPUT_DIR, "article_level_scores.csv"
        )
        self.MONTHLY_STANCE_CSV = os.path.join(
            self.OUTPUT_DIR, "monthly_stance.csv"
        )
        self.MONTHLY_POLARIZATION_CSV = os.path.join(
            self.OUTPUT_DIR, "monthly_polarization.csv"
        )
        self.SENTENCE_CROSSWALK_CSV = os.path.join(
            self.OUTPUT_DIR, "sentence_id_crosswalk.csv"
        )
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.sentence_crosswalk: Optional[pd.DataFrame] = None

    def run(self):
        """Run the complete observed-data aggregation."""

        print("=" * 70)
        print("LLM AGGREGATOR - Pipeline C")
        print("=" * 70)
        annotated = self._load_input()
        article = self._aggregate_to_article(annotated)
        print("\n[3/5] Calculating observed monthly stance and raw IQR...")
        print("\n[4/5] Calculating observed monthly polarization...")
        monthly_stance, monthly_polarization = calculate_point_estimates(article)
        self._save_outputs(article, monthly_stance, monthly_polarization)
        print("=" * 70)
        print("Aggregation complete")
        print(f"Output: {self.OUTPUT_DIR}")
        print("=" * 70)
        return article, monthly_stance, monthly_polarization

    def _load_input(self) -> pd.DataFrame:
        """Load immutable annotations, construct stable IDs, and retain successes."""

        print("\n[1/5] Loading immutable completed LLM annotations...")
        if not os.path.exists(self.INPUT_CSV):
            raise FileNotFoundError(f"Input file not found: {self.INPUT_CSV}")
        frame = pd.read_csv(self.INPUT_CSV)
        print(f"Loaded {len(frame):,} sentence rows")
        stored_identifier_columns = [
            column
            for column in ("stable_article_id", "stable_sentence_id")
            if column in frame.columns
        ]
        if stored_identifier_columns:
            frame = frame.drop(columns=stored_identifier_columns)
        required = {
            "title",
            "primary_body",
            "date",
            "party_mentioned",
            "cleaned_sentence",
            "article_weight",
            "llm_error_type",
            *self.DIMENSIONS,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        frame["_llm_row_number"] = np.arange(len(frame), dtype=np.int64)
        frame = add_stable_ids(frame)
        self.sentence_crosswalk = frame[
            [
                "_llm_row_number",
                "stable_sentence_id",
                "stable_article_id",
                "party_mentioned",
            ]
        ].rename(columns={"_llm_row_number": "llm_row_number"})
        self.sentence_crosswalk.insert(1, "article_id_schema", ARTICLE_ID_SCHEMA)
        self.sentence_crosswalk.insert(2, "sentence_id_schema", SENTENCE_ID_SCHEMA)

        original_count = len(frame)
        frame = frame.loc[frame["llm_error_type"].eq("success")].copy()
        print(
            f"Retained {len(frame):,} successful sentence rows and excluded "
            f"{original_count - len(frame):,} error rows"
        )
        frame["date"] = pd.to_datetime(frame["date"])
        frame["year_month"] = frame["date"].dt.to_period("M").astype(str)
        if "publisher" in frame.columns and "source" not in frame.columns:
            frame["source"] = frame["publisher"]
        if "source" not in frame.columns:
            raise ValueError("annotations contain neither source nor publisher")
        frame["party"] = frame["party_mentioned"].str.lower().str.strip()
        frame.loc[
            frame["party"].str.contains("rep", na=False), "party"
        ] = "republican"
        frame.loc[
            frame["party"].str.contains("dem", na=False), "party"
        ] = "democrat"
        unexpected = sorted(
            set(frame["party"].dropna()) - {"republican", "democrat"}
        )
        if unexpected:
            raise ValueError(f"Unexpected party values: {unexpected}")
        return frame

    def _aggregate_to_article(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentence scores to one stable article-party row."""

        print("\n[2/5] Aggregating sentences to stable article-party rows...")
        lineage_counts = frame.groupby("stable_article_id").agg(
            {"title": "nunique", "primary_body": "nunique"}
        )
        if (lineage_counts > 1).any(axis=None):
            raise ValueError(
                "A stable article ID maps to more than one title or primary_body"
            )
        base = (
            frame.groupby(["stable_article_id", "party"])
            .agg(
                {
                    "title": "first",
                    "date": "first",
                    "primary_body": "first",
                    "year_month": "first",
                    "source": "first",
                    "article_weight": "first",
                }
            )
            .reset_index()
        )
        for dimension in self.DIMENSIONS:
            scored = frame.loc[frame[dimension].notna()]
            if scored.empty:
                base[f"{dimension}_score"] = np.nan
                base[f"{dimension}_n"] = 0
                continue
            summary = (
                scored.groupby(["stable_article_id", "party"])
                .agg({dimension: ["mean", "count"]})
                .reset_index()
            )
            summary.columns = [
                "stable_article_id",
                "party",
                f"{dimension}_score",
                f"{dimension}_n",
            ]
            base = base.merge(
                summary,
                on=["stable_article_id", "party"],
                how="left",
                validate="one_to_one",
            )
        article = base.rename(columns={"article_weight": "weight"})
        columns = [
            "stable_article_id",
            "title",
            "date",
            "primary_body",
            "party",
            "year_month",
            "source",
            "weight",
        ]
        for dimension in self.DIMENSIONS:
            columns.extend([f"{dimension}_score", f"{dimension}_n"])
        article = (
            article[columns]
            .sort_values(["stable_article_id", "party"], kind="stable")
            .reset_index(drop=True)
        )
        print(f"Created {len(article):,} article-party rows")
        print(f"Unique stable articles: {article['stable_article_id'].nunique():,}")
        return article

    def _save_outputs(
        self,
        article: pd.DataFrame,
        monthly_stance: pd.DataFrame,
        monthly_polarization: pd.DataFrame,
    ) -> None:
        """Write observed aggregation outputs."""

        print("\n[5/5] Saving observed aggregation outputs...")
        self._write_csv_atomic(article, self.ARTICLE_SCORES_CSV)
        if self.sentence_crosswalk is None:
            raise RuntimeError("Sentence crosswalk was not constructed")
        self._write_csv_atomic(
            self.sentence_crosswalk,
            self.SENTENCE_CROSSWALK_CSV,
        )
        self._write_csv_atomic(monthly_stance, self.MONTHLY_STANCE_CSV)
        self._write_csv_atomic(
            monthly_polarization,
            self.MONTHLY_POLARIZATION_CSV,
        )
        print(f"article_level_scores.csv rows: {len(article):,}")
        print(f"sentence_id_crosswalk.csv rows: {len(self.sentence_crosswalk):,}")
        print(f"monthly_stance.csv rows: {len(monthly_stance):,}")
        print(f"monthly_polarization.csv rows: {len(monthly_polarization):,}")
        for dimension in self.DIMENSIONS:
            average = monthly_polarization.loc[
                monthly_polarization["dimension"].eq(dimension), "polarization"
            ].mean()
            print(f"Mean {dimension.upper()} polarization: {average:+.3f}")

    @staticmethod
    def _write_csv_atomic(frame: pd.DataFrame, destination: str) -> None:
        """Write one CSV atomically and retain byte-identical existing files."""

        output_dir = os.path.dirname(destination)
        os.makedirs(output_dir, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=output_dir,
            prefix=f".{os.path.basename(destination)}.",
            suffix=".tmp",
        )
        os.close(handle)
        try:
            frame.to_csv(temporary, index=False)
            if os.path.exists(destination):
                with open(temporary, "rb") as source:
                    proposed_hash = hashlib.sha256(source.read()).digest()
                with open(destination, "rb") as source:
                    existing_hash = hashlib.sha256(source.read()).digest()
                if proposed_hash == existing_hash:
                    os.unlink(temporary)
                    print(
                        f"Retained byte-identical {os.path.basename(destination)}"
                    )
                    return
            os.replace(temporary, destination)
            print(f"Replaced {os.path.basename(destination)} atomically")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def main() -> None:
    LLMAggregator().run()


if __name__ == "__main__":
    main()
