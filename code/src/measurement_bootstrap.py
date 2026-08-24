"""Joint article-cluster measurement bootstrap for Pipeline C.

The module treats a stable article as the resampling unit. Every selected article
carries all associated party and ideological-dimension rows so one replicate can
jointly estimate party stance, polarization, raw IQR, and downstream composite
uncertainty without breaking observed cross-party dependence.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


SEED_SCHEMA = "paper3-seed-v1"
BOOTSTRAP_SCHEMA = "paper3-joint-article-cluster-bootstrap-v1"
DIMENSIONS = ("q1", "q2", "q3", "q4")
PARTIES = ("democrat", "republican")


@dataclass(frozen=True)
class JointBootstrapConfig:
    """Configuration for one reproducible joint measurement bootstrap."""

    n_bootstrap: int = 500
    master_seed: int = 42
    ci_lower: float = 2.5
    ci_upper: float = 97.5
    min_valid_fraction: float = 0.95
    namespace: str = "joint_article_cluster_measurement"

    def validate(self) -> None:
        if self.n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2")
        if not 0 <= self.ci_lower < self.ci_upper <= 100:
            raise ValueError("bootstrap confidence percentiles are invalid")
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0, 1]")


@dataclass
class JointMeasurementResult:
    monthly_stance: pd.DataFrame
    monthly_polarization: pd.DataFrame
    replicates: pd.DataFrame
    metadata: Dict[str, object]


def derive_child_seed(master_seed: int, namespace: str) -> int:
    """Derive a stable 128-bit child seed without Python's built-in hash."""

    payload = f"{SEED_SCHEMA}|{int(master_seed)}|{namespace}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:16], byteorder="big")


def weighted_mean(scores: np.ndarray, weights: np.ndarray) -> float:
    """Return the finite positive-weighted mean used by the aggregation."""

    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    denominator = weights[valid].sum()
    if denominator <= 0:
        return np.nan
    return float(np.dot(scores[valid], weights[valid]) / denominator)


def raw_iqr(scores: np.ndarray, minimum_n: int = 4) -> float:
    """Return the observed article-score IQR using NumPy's linear quantiles."""

    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < minimum_n:
        return np.nan
    q25, q75 = np.percentile(values, [25.0, 75.0])
    return float(q75 - q25)


def _validate_article_frame(article_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "stable_article_id",
        "party",
        "year_month",
        "weight",
        *(f"{dimension}_score" for dimension in DIMENSIONS),
    }
    missing = sorted(required.difference(article_df.columns))
    if missing:
        raise ValueError(f"article data are missing required columns: {missing}")

    frame = article_df.copy()
    frame["stable_article_id"] = frame["stable_article_id"].astype("string")
    frame["party"] = frame["party"].astype("string").str.lower().str.strip()
    frame["year_month"] = frame["year_month"].astype("string")

    invalid_party = sorted(set(frame["party"].dropna()) - set(PARTIES))
    if invalid_party:
        raise ValueError(f"unexpected party values: {invalid_party}")
    if frame["stable_article_id"].isna().any():
        raise ValueError("stable_article_id contains missing values")
    if frame.duplicated(["stable_article_id", "party"]).any():
        raise ValueError("article data contain duplicate stable_article_id-party rows")
    if frame.groupby("stable_article_id")["year_month"].nunique().gt(1).any():
        raise ValueError("one stable article maps to multiple year_month values")
    return frame


def calculate_point_estimates(
    article_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate unchanged stance and polarization points plus observed raw IQR."""

    frame = _validate_article_frame(article_df)
    stance_rows = []
    for (year_month, party), group in frame.groupby(
        ["year_month", "party"], sort=True
    ):
        for dimension in DIMENSIONS:
            score_column = f"{dimension}_score"
            valid = group.loc[group[score_column].notna(), [score_column, "weight"]]
            scores = valid[score_column].to_numpy(dtype=float)
            weights = valid["weight"].to_numpy(dtype=float)
            stance_rows.append(
                {
                    "year_month": str(year_month),
                    "party": str(party),
                    "dimension": dimension,
                    "mean": weighted_mean(scores, weights),
                    "raw_iqr": raw_iqr(scores),
                    "n_articles": int(len(valid)),
                }
            )

    stance = pd.DataFrame(stance_rows).sort_values(
        ["year_month", "party", "dimension"], kind="stable"
    )
    stance = stance.reset_index(drop=True)

    polarization_rows = []
    for (year_month, dimension), group in stance.groupby(
        ["year_month", "dimension"], sort=True
    ):
        by_party = group.set_index("party")
        rep_mean = float(by_party.loc["republican", "mean"])
        dem_mean = float(by_party.loc["democrat", "mean"])
        rep_n = int(by_party.loc["republican", "n_articles"])
        dem_n = int(by_party.loc["democrat", "n_articles"])
        polarization_rows.append(
            {
                "year_month": str(year_month),
                "dimension": dimension,
                "rep_mean": rep_mean,
                "dem_mean": dem_mean,
                "polarization": rep_mean - dem_mean,
                "rep_n": rep_n,
                "dem_n": dem_n,
            }
        )

    polarization = pd.DataFrame(polarization_rows).sort_values(
        ["year_month", "dimension"], kind="stable"
    )
    return stance, polarization.reset_index(drop=True)


def _cell_arrays(
    month_frame: pd.DataFrame,
    article_ids: pd.Index,
    party: str,
    dimension: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align one party-dimension cell to the month's article-ID universe."""

    subset = month_frame.loc[
        month_frame["party"].eq(party),
        ["stable_article_id", f"{dimension}_score", "weight"],
    ].set_index("stable_article_id")
    subset = subset.reindex(article_ids)
    return (
        subset[f"{dimension}_score"].to_numpy(dtype=float),
        subset["weight"].to_numpy(dtype=float),
    )


def _bootstrap_cell(
    counts: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate weighted means and raw IQRs from shared article counts."""

    n_bootstrap = counts.shape[0]
    means = np.full(n_bootstrap, np.nan, dtype=float)
    iqrs = np.full(n_bootstrap, np.nan, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return means, iqrs

    cell_counts = counts[:, valid]
    cell_scores = scores[valid]
    cell_weights = weights[valid]
    denominators = cell_counts @ cell_weights
    numerators = cell_counts @ (cell_scores * cell_weights)
    np.divide(
        numerators,
        denominators,
        out=means,
        where=denominators > 0,
    )

    for replicate_index in range(n_bootstrap):
        replicate_counts = cell_counts[replicate_index]
        if int(replicate_counts.sum()) < 4:
            continue
        expanded_scores = np.repeat(cell_scores, replicate_counts)
        iqrs[replicate_index] = raw_iqr(expanded_scores)
    return means, iqrs


def generate_joint_replicates(
    article_df: pd.DataFrame,
    config: JointBootstrapConfig,
) -> Tuple[pd.DataFrame, int]:
    """Generate shared monthly article-cluster bootstrap replicates."""

    config.validate()
    frame = _validate_article_frame(article_df)
    child_seed = derive_child_seed(config.master_seed, config.namespace)
    rng = np.random.default_rng(child_seed)
    replicate_frames = []

    for year_month, month_frame in frame.groupby("year_month", sort=True):
        article_ids = pd.Index(
            sorted(month_frame["stable_article_id"].dropna().unique()),
            dtype="string",
        )
        n_articles = len(article_ids)
        if n_articles < 1:
            raise ValueError(f"month {year_month} has no stable articles")

        probabilities = np.full(n_articles, 1.0 / n_articles, dtype=float)
        counts = rng.multinomial(
            n_articles,
            probabilities,
            size=config.n_bootstrap,
        )
        unique_sampled = np.count_nonzero(counts, axis=1)

        for dimension in DIMENSIONS:
            values: Dict[str, np.ndarray] = {}
            for party in PARTIES:
                scores, weights = _cell_arrays(
                    month_frame,
                    article_ids,
                    party,
                    dimension,
                )
                means, iqrs = _bootstrap_cell(counts, scores, weights)
                values[f"{party}_mean"] = means
                values[f"{party}_raw_iqr"] = iqrs

            replicate_frames.append(
                pd.DataFrame(
                    {
                        "replicate": np.arange(1, config.n_bootstrap + 1),
                        "year_month": str(year_month),
                        "dimension": dimension,
                        "n_articles_drawn": n_articles,
                        "n_unique_articles_sampled": unique_sampled,
                        "rep_mean": values["republican_mean"],
                        "dem_mean": values["democrat_mean"],
                        "polarization": (
                            values["republican_mean"]
                            - values["democrat_mean"]
                        ),
                        "rep_raw_iqr": values["republican_raw_iqr"],
                        "dem_raw_iqr": values["democrat_raw_iqr"],
                    }
                )
            )

    replicates = pd.concat(replicate_frames, ignore_index=True)
    replicates = replicates.sort_values(
        ["replicate", "year_month", "dimension"], kind="stable"
    ).reset_index(drop=True)
    return replicates, child_seed


def _percentile_summary(
    values: Iterable[float],
    config: JointBootstrapConfig,
) -> Tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    minimum_valid = int(np.ceil(config.n_bootstrap * config.min_valid_fraction))
    if finite.size < minimum_valid:
        return np.nan, np.nan, int(finite.size)
    lower, upper = np.percentile(finite, [config.ci_lower, config.ci_upper])
    return float(lower), float(upper), int(finite.size)


def attach_joint_intervals(
    point_stance: pd.DataFrame,
    point_polarization: pd.DataFrame,
    replicates: pd.DataFrame,
    config: JointBootstrapConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Attach percentile intervals from the same shared replicate sequence."""

    stance_interval_rows = []
    for (year_month, dimension), group in replicates.groupby(
        ["year_month", "dimension"], sort=True
    ):
        for party, prefix in (("republican", "rep"), ("democrat", "dem")):
            mean_lower, mean_upper, mean_valid = _percentile_summary(
                group[f"{prefix}_mean"], config
            )
            iqr_lower, iqr_upper, iqr_valid = _percentile_summary(
                group[f"{prefix}_raw_iqr"], config
            )
            stance_interval_rows.append(
                {
                    "year_month": str(year_month),
                    "party": party,
                    "dimension": dimension,
                    "ci_lower": mean_lower,
                    "ci_upper": mean_upper,
                    "raw_iqr_ci_lower": iqr_lower,
                    "raw_iqr_ci_upper": iqr_upper,
                    "bootstrap_valid_mean": mean_valid,
                    "bootstrap_valid_raw_iqr": iqr_valid,
                }
            )

    stance_intervals = pd.DataFrame(stance_interval_rows)
    stance = point_stance.merge(
        stance_intervals,
        on=["year_month", "party", "dimension"],
        how="left",
        validate="one_to_one",
    )
    stance = stance[
        [
            "year_month",
            "party",
            "dimension",
            "mean",
            "ci_lower",
            "ci_upper",
            "raw_iqr",
            "raw_iqr_ci_lower",
            "raw_iqr_ci_upper",
            "n_articles",
            "bootstrap_valid_mean",
            "bootstrap_valid_raw_iqr",
        ]
    ].sort_values(["year_month", "party", "dimension"], kind="stable")

    polarization_interval_rows = []
    for (year_month, dimension), group in replicates.groupby(
        ["year_month", "dimension"], sort=True
    ):
        lower, upper, valid = _percentile_summary(group["polarization"], config)
        polarization_interval_rows.append(
            {
                "year_month": str(year_month),
                "dimension": dimension,
                "pol_ci_lower": lower,
                "pol_ci_upper": upper,
                "bootstrap_valid": valid,
            }
        )
    polarization_intervals = pd.DataFrame(polarization_interval_rows)
    polarization = point_polarization.merge(
        polarization_intervals,
        on=["year_month", "dimension"],
        how="left",
        validate="one_to_one",
    )
    polarization = polarization[
        [
            "year_month",
            "dimension",
            "rep_mean",
            "dem_mean",
            "polarization",
            "pol_ci_lower",
            "pol_ci_upper",
            "rep_n",
            "dem_n",
            "bootstrap_valid",
        ]
    ].sort_values(["year_month", "dimension"], kind="stable")
    return stance.reset_index(drop=True), polarization.reset_index(drop=True)


def run_joint_measurement_bootstrap(
    article_df: pd.DataFrame,
    config: JointBootstrapConfig | None = None,
) -> JointMeasurementResult:
    """Calculate points and one coherent bootstrap uncertainty layer."""

    selected_config = config or JointBootstrapConfig()
    selected_config.validate()
    point_stance, point_polarization = calculate_point_estimates(article_df)
    replicates, child_seed = generate_joint_replicates(article_df, selected_config)
    stance, polarization = attach_joint_intervals(
        point_stance,
        point_polarization,
        replicates,
        selected_config,
    )
    metadata: Dict[str, object] = {
        "schema": BOOTSTRAP_SCHEMA,
        "resampling_unit": "stable_article_id within year_month",
        "cluster_carry_rule": "all associated party and dimension rows",
        "point_estimate_change": False,
        "raw_iqr_minimum_articles": 4,
        "quantile_method": "numpy.percentile default linear interpolation",
        "n_bootstrap": selected_config.n_bootstrap,
        "ci_percentiles": [selected_config.ci_lower, selected_config.ci_upper],
        "minimum_valid_fraction": selected_config.min_valid_fraction,
        "master_seed": selected_config.master_seed,
        "seed_schema": SEED_SCHEMA,
        "child_namespace": selected_config.namespace,
        "child_seed": str(child_seed),
    }
    return JointMeasurementResult(
        monthly_stance=stance,
        monthly_polarization=polarization,
        replicates=replicates,
        metadata=metadata,
    )
