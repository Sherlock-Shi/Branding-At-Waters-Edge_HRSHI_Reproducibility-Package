"""Direct paired Sen-slope contrast for Paper 3 RQ2."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.trend_analysis import hamed_rao_sen_result, pairwise_slopes


CONTRAST_SCHEMA = "paper3-paired-party-sen-contrast-v1"
EXPECTED_MONTHS = pd.period_range("2022-03", "2024-12", freq="M").astype(str).tolist()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_pairwise_slopes(values: np.ndarray) -> np.ndarray:
    """Return slopes in a common lexicographic month-pair order."""

    series = np.asarray(values, dtype=float)
    return np.asarray(
        [
            (series[j] - series[i]) / (j - i)
            for i in range(len(series) - 1)
            for j in range(i + 1, len(series))
        ],
        dtype=float,
    )


def _q1_party_panel(stance: pd.DataFrame) -> pd.DataFrame:
    required = {"year_month", "party", "dimension", "mean"}
    if not required.issubset(stance.columns):
        raise ValueError(f"monthly stance input lacks columns: {sorted(required - set(stance.columns))}")
    q1 = stance.loc[stance["dimension"].eq("q1")].copy()
    panel = q1.pivot(index="year_month", columns="party", values="mean").sort_index()
    if panel.index.astype(str).tolist() != EXPECTED_MONTHS:
        raise ValueError("Q1 party stance panel does not contain the exact 34-month calendar")
    if list(panel.columns) != ["democrat", "republican"]:
        raise ValueError("Q1 party stance panel must contain Democrat and Republican series")
    if not np.isfinite(panel.to_numpy(float)).all():
        raise ValueError("Q1 party stance panel contains nonfinite values")
    return panel


def calculate_party_slope_contrast(
    stance: pd.DataFrame,
    polarization: pd.DataFrame,
    trends: pd.DataFrame,
) -> Dict[str, object]:
    """Calculate the direct paired RQ2 slope contrast and reconciliation fields."""

    panel = _q1_party_panel(stance)
    republican = panel["republican"].to_numpy(float)
    democrat = panel["democrat"].to_numpy(float)
    gap = republican - democrat

    q1_polarization = (
        polarization.loc[polarization["dimension"].eq("q1")]
        .sort_values("year_month")
        .reset_index(drop=True)
    )
    if q1_polarization["year_month"].astype(str).tolist() != EXPECTED_MONTHS:
        raise ValueError("Q1 polarization does not contain the exact 34-month calendar")
    if not np.allclose(
        gap,
        q1_polarization["polarization"].to_numpy(float),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("Republican minus Democratic stance does not equal Q1 polarization")

    republican_pair_slopes = ordered_pairwise_slopes(republican)
    democrat_pair_slopes = ordered_pairwise_slopes(democrat)
    paired_differences = np.sort(republican_pair_slopes - democrat_pair_slopes)
    gap_pair_slopes = pairwise_slopes(gap)
    if not np.allclose(paired_differences, gap_pair_slopes, atol=1e-12, rtol=1e-12):
        raise RuntimeError("paired month-pair slope identity failed")

    result = hamed_rao_sen_result(
        gap,
        series_name="RQ2 paired Republican minus Democratic stance slope contrast",
        lag=2,
        scale="Q1 stance points per month",
    ).values

    trend_by_series = trends.set_index("series")
    required_rows = {
        "Q1 polarization",
        "Q1 Republican stance",
        "Q1 Democratic stance",
    }
    if not required_rows.issubset(trend_by_series.index):
        raise ValueError("trend contract lacks one or more RQ2 rows")
    polarization_row = trend_by_series.loc["Q1 polarization"]
    for field, output_field in (
        ("S", "S"),
        ("variance_ordinary", "variance_ordinary"),
        ("variance_hamed_rao", "variance_hamed_rao"),
        ("hamed_rao_correction_factor", "hamed_rao_correction_factor"),
        ("Z", "Z"),
        ("tau", "tau"),
        ("p_value", "p_value"),
        ("sen_slope", "sen_slope"),
        ("sen_slope_ci_lower", "sen_slope_ci_lower"),
        ("sen_slope_ci_upper", "sen_slope_ci_upper"),
    ):
        if not np.isclose(
            float(result[output_field]),
            float(polarization_row[field]),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise RuntimeError(f"paired contrast does not reconcile with polarization field {field}")

    republican_slope = float(trend_by_series.loc["Q1 Republican stance", "sen_slope"])
    democrat_slope = float(trend_by_series.loc["Q1 Democratic stance", "sen_slope"])
    return {
        "schema": CONTRAST_SCHEMA,
        "contrast_id": "rq2_republican_minus_democratic_paired_sen_slope",
        "research_role": "direct inferential between-party temporal-change contrast",
        "estimand": "median paired difference in Republican and Democratic month-pair slopes",
        "algebraic_identity": "slope(R)-slope(D)=slope(R-D) for every common month pair",
        "paired_unit": "ordered pair of months",
        "n_observations": int(len(gap)),
        "n_month_pairs": int(len(gap_pair_slopes)),
        "period_start": EXPECTED_MONTHS[0],
        "period_end": EXPECTED_MONTHS[-1],
        "lag": int(result["lag"]),
        "S": float(result["S"]),
        "variance_ordinary": float(result["variance_ordinary"]),
        "variance_hamed_rao": float(result["variance_hamed_rao"]),
        "hamed_rao_correction_factor": float(result["hamed_rao_correction_factor"]),
        "Z": float(result["Z"]),
        "tau": float(result["tau"]),
        "p_value": float(result["p_value"]),
        "trend": str(result["trend"]),
        "significant_0_05": bool(result["significant_0_05"]),
        "paired_sen_slope_contrast": float(result["sen_slope"]),
        "paired_sen_slope_ci_lower": float(result["sen_slope_ci_lower"]),
        "paired_sen_slope_ci_upper": float(result["sen_slope_ci_upper"]),
        "confidence_level": float(result["sen_slope_confidence_level"]),
        "unit": "Q1 stance points per month",
        "republican_marginal_sen_slope": republican_slope,
        "democratic_marginal_sen_slope": democrat_slope,
        "descriptive_difference_of_marginal_sen_slopes": republican_slope - democrat_slope,
        "equals_polarization_sen_slope": True,
        "preferred_interpretation": (
            "Republican stance increased more than Democratic stance across the study period"
        ),
        "wording_boundary": (
            "Use increased more. Use increased faster only when defined as a larger typical "
            "monthly change under the paired Sen contrast."
        ),
        "validation_status": "requires matching independent receipt",
    }


def write_party_slope_contrast(
    stance_path: Path,
    polarization_path: Path,
    trends_path: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Write the paired RQ2 contrast and its source-bound metadata."""

    paths = [stance_path, polarization_path, trends_path]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing paired-contrast inputs: " + ", ".join(missing))
    output_directory.mkdir(parents=True, exist_ok=True)
    result = calculate_party_slope_contrast(
        pd.read_csv(stance_path),
        pd.read_csv(polarization_path),
        pd.read_csv(trends_path),
    )
    output_path = output_directory / "party_slope_contrast.csv"
    pd.DataFrame([result]).to_csv(output_path, index=False)

    repository_root = Path(__file__).resolve().parents[2]
    source_hashes = {str(path.resolve()): sha256_file(path) for path in paths}
    run_payload = "|".join(source_hashes.values()) + "|" + CONTRAST_SCHEMA
    run_id = "party-slope-contrast-" + hashlib.sha256(run_payload.encode("utf-8")).hexdigest()[:16]
    metadata = {
        "schema": "paper3-paired-party-sen-contrast-metadata-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "status": "generated_pending_independent_validation",
        "method": (
            "For every common month pair, subtract the Democratic slope from the "
            "Republican slope and take the median. The pairwise differences equal "
            "the slopes of the Republican minus Democratic gap exactly."
        ),
        "sources": [
            {
                "citation": (
                    "Sen, P. K. (1968). Estimates of the Regression Coefficient Based "
                    "on Kendall's Tau. Journal of the American Statistical Association, "
                    "63(324), 1379-1389."
                ),
                "doi": "10.1080/01621459.1968.10480934",
                "role": "paired median slope estimator and rank-order interval",
            },
            {
                "citation": (
                    "Hamed, K. H., and Rao, A. R. (1998). A modified Mann-Kendall trend "
                    "test for autocorrelated data. Journal of Hydrology, 204(1-4), 182-196."
                ),
                "doi": "10.1016/S0022-1694(97)00125-X",
                "role": "variance correction for serial dependence",
            },
        ],
        "input_sha256": source_hashes,
        "output_sha256": {output_path.name: sha256_file(output_path)},
        "code_sha256": {
            str(Path(__file__).resolve().relative_to(repository_root)): sha256_file(Path(__file__).resolve())
        },
    }
    metadata_path = output_directory / "party_slope_contrast_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, metadata_path
