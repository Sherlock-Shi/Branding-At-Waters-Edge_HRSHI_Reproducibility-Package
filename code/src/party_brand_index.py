"""Party Brand Index reconstruction from validated monthly measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


PBI_SCHEMA = "paper3-party-brand-index-v1"


@dataclass
class PartyBrandIndexResult:
    timeseries: pd.DataFrame
    monthly_coherence: pd.DataFrame
    replicates: pd.DataFrame
    metadata: Dict[str, object]


def _components(
    polarization: pd.Series,
    rep_raw_iqr: pd.Series,
    dem_raw_iqr: pd.Series,
    max_polarization: float,
    k: float,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if not np.isfinite(max_polarization) or max_polarization <= 0:
        raise ValueError("max_polarization must be positive and finite")
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be positive and finite")

    p = polarization.abs() / max_polarization
    c_r = k / (k + rep_raw_iqr)
    c_d = k / (k + dem_raw_iqr)
    pbi = np.cbrt(p * c_r * c_d) * 100.0
    return p, c_r, c_d, pbi


def _point_inputs(
    monthly_stance: pd.DataFrame,
    monthly_polarization: pd.DataFrame,
) -> pd.DataFrame:
    stance = monthly_stance.loc[
        monthly_stance["dimension"].eq("q1"),
        [
            "year_month",
            "party",
            "raw_iqr",
            "raw_iqr_ci_lower",
            "raw_iqr_ci_upper",
            "n_articles",
        ],
    ]
    pivot = stance.pivot(index="year_month", columns="party")
    pivot.columns = [f"{value}_{party}" for value, party in pivot.columns]
    pivot = pivot.reset_index().rename(
        columns={
            "raw_iqr_republican": "rep_raw_iqr",
            "raw_iqr_democrat": "dem_raw_iqr",
            "raw_iqr_ci_lower_republican": "rep_raw_iqr_ci_lower",
            "raw_iqr_ci_upper_republican": "rep_raw_iqr_ci_upper",
            "raw_iqr_ci_lower_democrat": "dem_raw_iqr_ci_lower",
            "raw_iqr_ci_upper_democrat": "dem_raw_iqr_ci_upper",
            "n_articles_republican": "rep_n_articles",
            "n_articles_democrat": "dem_n_articles",
        }
    )
    polarization = monthly_polarization.loc[
        monthly_polarization["dimension"].eq("q1"),
        ["year_month", "polarization", "pol_ci_lower", "pol_ci_upper"],
    ]
    return polarization.merge(
        pivot,
        on="year_month",
        how="inner",
        validate="one_to_one",
    ).sort_values("year_month", kind="stable")


def reconstruct_party_brand_index(
    monthly_stance: pd.DataFrame,
    monthly_polarization: pd.DataFrame,
    measurement_replicates: pd.DataFrame,
    ci_lower: float = 2.5,
    ci_upper: float = 97.5,
    min_valid_fraction: float = 0.95,
) -> PartyBrandIndexResult:
    """Reconstruct the point index and its joint bootstrap distribution.

    Each replicate reapplies the complete estimator, including the balanced
    pooled-median k rule and observed-maximum polarization normalization. This
    retains the established calculation while propagating calibration and
    component covariance through the article-sampling distribution.
    """

    point = _point_inputs(monthly_stance, monthly_polarization)
    max_polarization = float(point["polarization"].abs().max())
    median_rep_iqr = float(point["rep_raw_iqr"].median())
    median_dem_iqr = float(point["dem_raw_iqr"].median())
    k = 0.5 * median_rep_iqr + 0.5 * median_dem_iqr
    point["P"], point["C_R"], point["C_D"], point["party_brand_index"] = (
        _components(
            point["polarization"],
            point["rep_raw_iqr"],
            point["dem_raw_iqr"],
            max_polarization,
            k,
        )
    )
    point["max_pol_used"] = max_polarization
    point["k_used"] = k

    replicates = measurement_replicates.loc[
        measurement_replicates["dimension"].eq("q1"),
        [
            "replicate",
            "year_month",
            "polarization",
            "rep_raw_iqr",
            "dem_raw_iqr",
        ],
    ].copy()
    requested_bootstrap = int(measurement_replicates["replicate"].nunique())
    output_frames = []
    calibration_rows = []
    for replicate, group in replicates.groupby("replicate", sort=True):
        group = group.sort_values("year_month", kind="stable").copy()
        replicate_max = float(group["polarization"].abs().max())
        finite_rep_iqr = group.loc[
            np.isfinite(group["rep_raw_iqr"]), "rep_raw_iqr"
        ].to_numpy(dtype=float)
        finite_dem_iqr = group.loc[
            np.isfinite(group["dem_raw_iqr"]), "dem_raw_iqr"
        ].to_numpy(dtype=float)
        if not len(finite_rep_iqr) or not len(finite_dem_iqr):
            continue
        replicate_rep_median = float(np.median(finite_rep_iqr))
        replicate_dem_median = float(np.median(finite_dem_iqr))
        replicate_k = 0.5 * replicate_rep_median + 0.5 * replicate_dem_median
        if (
            not np.isfinite(replicate_max)
            or not np.isfinite(replicate_k)
            or replicate_max <= 0
            or replicate_k <= 0
        ):
            continue
        group["P"], group["C_R"], group["C_D"], group["party_brand_index"] = (
            _components(
                group["polarization"],
                group["rep_raw_iqr"],
                group["dem_raw_iqr"],
                replicate_max,
                replicate_k,
            )
        )
        group["max_pol_used"] = replicate_max
        group["k_used"] = replicate_k
        output_frames.append(group)
        calibration_rows.append(
            {
                "replicate": int(replicate),
                "max_pol_used": replicate_max,
                "k_used": replicate_k,
            }
        )

    if not output_frames:
        raise ValueError("no valid Party Brand Index bootstrap replicates")
    pbi_replicates = pd.concat(output_frames, ignore_index=True)
    pbi_replicates = pbi_replicates.sort_values(
        ["replicate", "year_month"], kind="stable"
    ).reset_index(drop=True)

    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    minimum_valid = int(
        np.ceil(requested_bootstrap * min_valid_fraction)
    )
    interval_columns = {
        "P": "P",
        "C_R": "C_R",
        "C_D": "C_D",
        "party_brand_index": "pbi",
    }
    interval_rows = []
    for year_month, group in pbi_replicates.groupby("year_month", sort=True):
        row: Dict[str, object] = {"year_month": str(year_month)}
        for source, prefix in interval_columns.items():
            finite = group.loc[
                np.isfinite(group[source]), source
            ].to_numpy(dtype=float)
            if finite.size >= minimum_valid:
                lower, upper = np.percentile(finite, [ci_lower, ci_upper])
            else:
                lower, upper = np.nan, np.nan
            row[f"{prefix}_ci_lower"] = float(lower)
            row[f"{prefix}_ci_upper"] = float(upper)
            row[f"{prefix}_bootstrap_valid"] = int(finite.size)
        interval_rows.append(row)
    intervals = pd.DataFrame(interval_rows)
    point = point.merge(intervals, on="year_month", how="left", validate="one_to_one")
    point = point[
        [
            "year_month",
            "polarization",
            "pol_ci_lower",
            "pol_ci_upper",
            "rep_raw_iqr",
            "rep_raw_iqr_ci_lower",
            "rep_raw_iqr_ci_upper",
            "dem_raw_iqr",
            "dem_raw_iqr_ci_lower",
            "dem_raw_iqr_ci_upper",
            "rep_n_articles",
            "dem_n_articles",
            "P",
            "P_ci_lower",
            "P_ci_upper",
            "C_R",
            "C_R_ci_lower",
            "C_R_ci_upper",
            "C_D",
            "C_D_ci_lower",
            "C_D_ci_upper",
            "party_brand_index",
            "pbi_ci_lower",
            "pbi_ci_upper",
            "max_pol_used",
            "k_used",
            "P_bootstrap_valid",
            "C_R_bootstrap_valid",
            "C_D_bootstrap_valid",
            "pbi_bootstrap_valid",
        ]
    ].reset_index(drop=True)

    coherence_frames = []
    for party, prefix, component in (
        ("republican", "rep", "C_R"),
        ("democrat", "dem", "C_D"),
    ):
        coherence_frames.append(
            pd.DataFrame(
                {
                    "year_month": point["year_month"],
                    "party": party,
                    "raw_iqr": point[f"{prefix}_raw_iqr"],
                    "raw_iqr_ci_lower": point[f"{prefix}_raw_iqr_ci_lower"],
                    "raw_iqr_ci_upper": point[f"{prefix}_raw_iqr_ci_upper"],
                    "message_coherence": point[component],
                    "coherence_ci_lower": point[f"{component}_ci_lower"],
                    "coherence_ci_upper": point[f"{component}_ci_upper"],
                    "n_articles": point[f"{prefix}_n_articles"].astype(int),
                    "bootstrap_valid_coherence": point[
                        f"{component}_bootstrap_valid"
                    ].astype(int),
                    "k_used": point["k_used"],
                }
            )
        )
    monthly_coherence = pd.concat(coherence_frames, ignore_index=True).sort_values(
        ["year_month", "party"], kind="stable"
    ).reset_index(drop=True)

    calibration = pd.DataFrame(calibration_rows)
    metadata: Dict[str, object] = {
        "schema": PBI_SCHEMA,
        "dimension": "q1",
        "formula": "100 * (P * C_R * C_D)^(1/3)",
        "polarization_component": "abs(polarization) / max(abs(polarization))",
        "coherence_component": "k / (k + raw_iqr)",
        "k_rule": "0.5 * median(rep_raw_iqr) + 0.5 * median(dem_raw_iqr)",
        "bootstrap_calibration": "recomputed within every joint replicate",
        "point_max_polarization": max_polarization,
        "point_median_rep_raw_iqr": median_rep_iqr,
        "point_median_dem_raw_iqr": median_dem_iqr,
        "point_k": k,
        "n_bootstrap": requested_bootstrap,
        "valid_calibration_replicates": int(
            pbi_replicates["replicate"].nunique()
        ),
        "ci_percentiles": [ci_lower, ci_upper],
        "minimum_valid_fraction": min_valid_fraction,
        "minimum_valid_replicates": minimum_valid,
        "replicate_k_min": float(calibration["k_used"].min()),
        "replicate_k_max": float(calibration["k_used"].max()),
        "replicate_max_pol_min": float(calibration["max_pol_used"].min()),
        "replicate_max_pol_max": float(calibration["max_pol_used"].max()),
    }
    return PartyBrandIndexResult(
        timeseries=point,
        monthly_coherence=monthly_coherence,
        replicates=pbi_replicates,
        metadata=metadata,
    )
