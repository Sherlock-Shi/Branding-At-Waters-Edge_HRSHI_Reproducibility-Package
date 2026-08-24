"""Observed-level general dominance with dependence-preserving resampling.

The analysis retains the monthly Q1 through Q4 polarization vector intact. It
uses a stationary bootstrap of month indices for temporal dependence and pairs
each temporal draw with one complete joint article-cluster measurement panel.
The resampling results are treated as dependence-aware sensitivity evidence
because the 34-month observed-level series contains structural nonstationarity.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.measurement_bootstrap import derive_child_seed


DOMINANCE_SCHEMA = "paper3-observed-level-dominance-v1"
DIMENSIONS = ("q1", "q2", "q3", "q4")
PREDICTORS = ("q2", "q3", "q4")
EXPECTED_MONTHS = pd.period_range("2022-03", "2024-12", freq="M").astype(str)
BOOTSTRAP_REPLICATES = 5000
MASTER_SEED = 42
PRIMARY_BLOCK_LENGTH = 4
SENSITIVITY_BLOCK_LENGTHS = (3, 4, 5)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _r_squared(response: np.ndarray, predictors: np.ndarray) -> float:
    response = np.asarray(response, dtype=float)
    predictors = np.asarray(predictors, dtype=float)
    total = float(np.sum((response - np.mean(response)) ** 2))
    if total <= np.finfo(float).eps:
        return 0.0
    design = np.column_stack([np.ones(len(response)), predictors])
    fitted = design @ np.linalg.lstsq(design, response, rcond=None)[0]
    residual = float(np.sum((response - fitted) ** 2))
    return float(np.clip(1.0 - residual / total, 0.0, 1.0))


def general_dominance(values: np.ndarray) -> tuple[float, np.ndarray]:
    """Return full R-squared and Budescu general dominance contributions."""

    values = np.asarray(values, dtype=float)
    if values.shape[1] != 4 or len(values) < 10 or not np.all(np.isfinite(values)):
        raise ValueError("dominance input must be a finite monthly Q1 through Q4 matrix")
    response = values[:, 0]
    predictors = values[:, 1:]
    subset_r2: dict[tuple[int, ...], float] = {(): 0.0}
    for size in range(1, len(PREDICTORS) + 1):
        for subset in combinations(range(len(PREDICTORS)), size):
            subset_r2[subset] = _r_squared(response, predictors[:, subset])
    contributions = np.zeros(len(PREDICTORS), dtype=float)
    for predictor in range(len(PREDICTORS)):
        conditional = []
        remaining = [index for index in range(len(PREDICTORS)) if index != predictor]
        for size in range(len(PREDICTORS)):
            increments = []
            for subset in combinations(remaining, size):
                expanded = tuple(sorted((*subset, predictor)))
                increments.append(subset_r2[expanded] - subset_r2[tuple(subset)])
            conditional.append(float(np.mean(increments)))
        contributions[predictor] = float(np.mean(conditional))
    full_r2 = subset_r2[tuple(range(len(PREDICTORS)))]
    if not np.isclose(np.sum(contributions), full_r2, atol=1.0e-10, rtol=0):
        raise RuntimeError("general dominance contributions do not sum to full R-squared")
    return float(full_r2), contributions


def stationary_bootstrap_indices(
    rng: np.random.Generator,
    n: int,
    expected_block_length: int,
) -> np.ndarray:
    if n < 2 or expected_block_length < 2:
        raise ValueError("stationary bootstrap dimensions are invalid")
    restart_probability = 1.0 / float(expected_block_length)
    indices = np.empty(n, dtype=np.int64)
    indices[0] = int(rng.integers(0, n))
    for position in range(1, n):
        if float(rng.random()) < restart_probability:
            indices[position] = int(rng.integers(0, n))
        else:
            indices[position] = (indices[position - 1] + 1) % n
    return indices


def _observed_matrix(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path).sort_values("year_month", kind="stable")
    required = {"year_month", *(f"{item}_polarization" for item in DIMENSIONS)}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"dominance design is missing columns: {missing}")
    if len(frame) != 34 or not np.array_equal(
        frame["year_month"].astype(str), EXPECTED_MONTHS
    ):
        raise ValueError("dominance design does not contain the complete calendar")
    values = frame[[f"{item}_polarization" for item in DIMENSIONS]].to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("dominance design contains non-finite values")
    return frame, values


def _measurement_array(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    frame = pd.read_csv(
        path,
        usecols=["replicate", "year_month", "dimension", "polarization"],
    )
    frame = frame.sort_values(
        ["replicate", "year_month", "dimension"], kind="stable"
    )
    pivot = frame.pivot(
        index=["replicate", "year_month"],
        columns="dimension",
        values="polarization",
    ).reset_index()
    pivot = pivot.sort_values(["replicate", "year_month"], kind="stable")
    if len(pivot) != BOOTSTRAP_REPLICATES * 34:
        raise ValueError("dominance measurement contract has the wrong row count")
    if not np.array_equal(
        pivot["replicate"].drop_duplicates().to_numpy(dtype=int),
        np.arange(1, BOOTSTRAP_REPLICATES + 1),
    ):
        raise ValueError("dominance measurement replicate identifiers are incomplete")
    values = pivot[list(DIMENSIONS)].to_numpy(dtype=float)
    values = values.reshape(BOOTSTRAP_REPLICATES, 34, len(DIMENSIONS))
    valid_panels = np.isfinite(values).all(axis=(1, 2))
    month_valid = (
        pivot.assign(complete=pivot[list(DIMENSIONS)].notna().all(axis=1))
        .groupby("year_month", sort=True)["complete"]
        .sum()
    )
    retained = values[valid_panels]
    if len(retained) < 2500:
        raise ValueError("too few complete four-dimensional measurement panels")
    return retained, {
        "requested_panels": BOOTSTRAP_REPLICATES,
        "complete_panels": int(valid_panels.sum()),
        "minimum_month_valid_replicates": int(month_valid.min()),
        "minimum_month": str(month_valid.idxmin()),
        "month_valid_replicates": {
            str(month): int(count) for month, count in month_valid.items()
        },
    }


def _statistics(values: np.ndarray) -> dict[str, float]:
    full_r2, contributions = general_dominance(values)
    correlations = np.corrcoef(values.T)[0, 1:]
    return {
        "full_r_squared": full_r2,
        "q2_contribution": contributions[0],
        "q3_contribution": contributions[1],
        "q4_contribution": contributions[2],
        "q4_minus_q2": contributions[2] - contributions[0],
        "q4_minus_q3": contributions[2] - contributions[1],
        "q1_q2_correlation": correlations[0],
        "q1_q3_correlation": correlations[1],
        "q1_q4_correlation": correlations[2],
    }


def _bootstrap_one_length(
    observed: np.ndarray,
    measurement: np.ndarray,
    block_length: int,
    master_seed: int,
) -> pd.DataFrame:
    namespace = f"dominance_stationary_month_vector_block_{block_length}_production"
    child_seed = derive_child_seed(master_seed, namespace)
    rng = np.random.default_rng(child_seed)
    rows = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = stationary_bootstrap_indices(
            rng, len(observed), expected_block_length=block_length
        )
        temporal = _statistics(observed[indices])
        measurement_panel = int(rng.integers(0, len(measurement)))
        combined = _statistics(measurement[measurement_panel, indices])
        for layer, statistics in (
            ("temporal_only", temporal),
            ("measurement_and_temporal", combined),
        ):
            rows.append(
                {
                    "replicate": replicate + 1,
                    "block_length": block_length,
                    "uncertainty_layer": layer,
                    **statistics,
                }
            )
    return pd.DataFrame(rows)


def _summary(
    replicates: pd.DataFrame,
    observed_statistics: dict[str, float],
) -> pd.DataFrame:
    targets = [
        "full_r_squared",
        "q2_contribution",
        "q3_contribution",
        "q4_contribution",
        "q4_minus_q2",
        "q4_minus_q3",
        "q1_q2_correlation",
        "q1_q3_correlation",
        "q1_q4_correlation",
    ]
    rows = []
    for (block_length, layer), group in replicates.groupby(
        ["block_length", "uncertainty_layer"], sort=True
    ):
        for target in targets:
            values = group[target].to_numpy(dtype=float)
            interval = np.percentile(values, [2.5, 97.5])
            rows.append(
                {
                    "block_length": int(block_length),
                    "uncertainty_layer": layer,
                    "statistic": target,
                    "observed_value": observed_statistics[target],
                    "bootstrap_mean": float(np.mean(values)),
                    "bootstrap_standard_error": float(np.std(values, ddof=1)),
                    "ci_lower_pointwise": float(interval[0]),
                    "ci_upper_pointwise": float(interval[1]),
                    "probability_positive": float(np.mean(values > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _contrast_contract(
    primary: pd.DataFrame,
    observed_statistics: dict[str, float],
) -> pd.DataFrame:
    primary = primary.loc[
        primary["block_length"].eq(PRIMARY_BLOCK_LENGTH)
        & primary["uncertainty_layer"].eq("measurement_and_temporal")
    ]
    contrasts = ("q4_minus_q2", "q4_minus_q3")
    deviations = np.column_stack(
        [
            primary[item].to_numpy(dtype=float) - observed_statistics[item]
            for item in contrasts
        ]
    )
    simultaneous_radius = float(np.percentile(np.max(np.abs(deviations), axis=1), 95.0))
    rows = []
    for contrast in contrasts:
        values = primary[contrast].to_numpy(dtype=float)
        pointwise = np.percentile(values, [2.5, 97.5])
        observed = observed_statistics[contrast]
        rows.append(
            {
                "contrast": contrast,
                "observed_difference": observed,
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_standard_error": float(np.std(values, ddof=1)),
                "pointwise_ci_lower": float(pointwise[0]),
                "pointwise_ci_upper": float(pointwise[1]),
                "simultaneous_ci_lower": observed - simultaneous_radius,
                "simultaneous_ci_upper": observed + simultaneous_radius,
                "probability_positive": float(np.mean(values > 0.0)),
                "simultaneous_family_size": 2,
                "interpretation_role": "dependence-aware sensitivity, not confirmatory coverage",
            }
        )
    return pd.DataFrame(rows)


def run_dominance_analysis(
    dominance_file: Path | str,
    measurement_replicate_file: Path | str,
    output_dir: Path | str,
    master_seed: int = MASTER_SEED,
) -> dict[str, pd.DataFrame]:
    """Run observed general dominance and the multivariate stationary bootstrap."""

    dominance_file = Path(dominance_file)
    measurement_replicate_file = Path(measurement_replicate_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, observed = _observed_matrix(dominance_file)
    measurement, measurement_availability = _measurement_array(
        measurement_replicate_file
    )
    observed_statistics = _statistics(observed)

    observed_rows = []
    total = observed_statistics["full_r_squared"]
    for predictor in PREDICTORS:
        contribution = observed_statistics[f"{predictor}_contribution"]
        observed_rows.append(
            {
                "predictor": predictor,
                "general_dominance_contribution": contribution,
                "share_of_model_r_squared": contribution / total,
                "correlation_with_q1": observed_statistics[
                    f"q1_{predictor}_correlation"
                ],
                "full_model_r_squared": total,
                "n_months": len(frame),
            }
        )
    observed_table = pd.DataFrame(observed_rows).sort_values(
        "general_dominance_contribution", ascending=False, kind="stable"
    )

    replicate_tables = [
        _bootstrap_one_length(observed, measurement, block_length, master_seed)
        for block_length in SENSITIVITY_BLOCK_LENGTHS
    ]
    all_replicates = pd.concat(replicate_tables, ignore_index=True)
    primary_replicates = all_replicates.loc[
        all_replicates["block_length"].eq(PRIMARY_BLOCK_LENGTH)
    ].reset_index(drop=True)
    sensitivity = _summary(all_replicates, observed_statistics)
    contrasts = _contrast_contract(all_replicates, observed_statistics)

    paths = {
        "observed": output_dir / "dominance_observed.csv",
        "replicates": output_dir / "dominance_bootstrap_replicates.csv",
        "sensitivity": output_dir / "dominance_block_sensitivity.csv",
        "contrasts": output_dir / "dominance_contrasts.csv",
    }
    observed_table.to_csv(paths["observed"], index=False)
    primary_replicates.to_csv(paths["replicates"], index=False)
    sensitivity.to_csv(paths["sensitivity"], index=False)
    contrasts.to_csv(paths["contrasts"], index=False)

    primary_combined = primary_replicates.loc[
        primary_replicates["uncertainty_layer"].eq("measurement_and_temporal")
    ]
    metadata = {
        "schema": DOMINANCE_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "generated_pending_independent_validation",
        "research_estimand": "contemporaneous observed-level general dominance of Q2, Q3, and Q4 for Q1 polarization",
        "n_months": 34,
        "period": ["2022-03", "2024-12"],
        "master_seed": master_seed,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_expected_block_length": PRIMARY_BLOCK_LENGTH,
        "block_length_basis": "ceiling of n^(1/3), with adjacent lengths 3 and 5 required as sensitivity checks",
        "sensitivity_expected_block_lengths": list(SENSITIVITY_BLOCK_LENGTHS),
        "resampling_unit": "complete four-dimensional monthly polarization vector",
        "measurement_pairing": "one complete defined joint article-cluster panel sampled for each stationary-bootstrap replicate",
        "measurement_panel_availability": measurement_availability,
        "production_child_seeds": {
            str(length): str(
                derive_child_seed(
                    master_seed,
                    f"dominance_stationary_month_vector_block_{length}_production",
                )
            )
            for length in SENSITIVITY_BLOCK_LENGTHS
        },
        "observed_ranking": list(observed_table["predictor"]),
        "primary_probability_q4_largest": float(
            np.mean(
                (primary_combined["q4_minus_q2"] > 0)
                & (primary_combined["q4_minus_q3"] > 0)
            )
        ),
        "reporting_status": "descriptive ranking with dependence-preserving sensitivity",
        "reason_not_confirmatory": "the short observed-level series contains structural nonstationarity and some Q3 article-cluster draws are undefined, so nominal coverage is not asserted",
        "interpretation_boundary": "contemporaneous co-movement, not temporal precedence or causation",
        "input_sha256": {
            str(dominance_file.resolve()): sha256_file(dominance_file),
            str(measurement_replicate_file.resolve()): sha256_file(
                measurement_replicate_file
            ),
        },
        "output_sha256": {
            path.name: sha256_file(path) for path in paths.values()
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "primary_sources": [
            {
                "citation": "Budescu, D. V. (1993). Dominance analysis: A new approach to the problem of relative importance of predictors in multiple regression. Psychological Bulletin, 114(3), 542-551.",
                "doi": "10.1037/0033-2909.114.3.542",
            },
            {
                "citation": "Politis, D. N., and Romano, J. P. (1994). The Stationary Bootstrap. Journal of the American Statistical Association, 89(428), 1303-1313.",
                "doi": "10.1080/01621459.1994.10476870",
            },
        ],
    }
    metadata_path = output_dir / "dominance_analysis_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "observed": observed_table,
        "replicates": primary_replicates,
        "sensitivity": sensitivity,
        "contrasts": contrasts,
    }
