"""Party-aware H3 model for polarization and message coherence.

The model retains Republican and Democratic raw IQR as separate outcomes. It
estimates party-specific polarization and time coefficients while allowing an
unrestricted contemporaneous covariance matrix and either independent monthly
errors or one common scalar AR(1) coefficient. The observed-data BIC comparison
selects the temporal error structure. The selected structure remains fixed
when the validated article-cluster measurement replicates are refitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf


H3_SCHEMA = "paper3-party-aware-h3-v1"
EXPECTED_MONTHS = pd.period_range("2022-03", "2024-12", freq="M").astype(str)
PARTIES = ("republican", "democrat")
TERMS = ("intercept", "polarization", "time")
AR_ORDERS = (0, 1)
PHI_BOUND = 0.95


@dataclass(frozen=True)
class CommonARFit:
    order: int
    phi: float
    coefficients: np.ndarray
    covariance: np.ndarray
    coefficient_covariance: np.ndarray
    residuals: np.ndarray
    innovations: np.ndarray
    log_likelihood: float
    bic: float
    n_months: int
    n_segments: int
    degrees_freedom: int


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_arrays(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("H3 outcome must contain two party columns")
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] != 3:
        raise ValueError("H3 design must contain intercept, polarization, and time")
    if len(y) < 12 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        raise ValueError("H3 arrays are incomplete or non-finite")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("H3 design matrix is rank deficient")
    return y, x


def _segment_starts(n: int, starts: np.ndarray | None) -> np.ndarray:
    if starts is None:
        result = np.zeros(n, dtype=bool)
        result[0] = True
        return result
    result = np.asarray(starts, dtype=bool)
    if result.shape != (n,) or not result[0]:
        raise ValueError("segment-start indicator is invalid")
    return result


def _transform_common_ar(
    y: np.ndarray,
    x: np.ndarray,
    phi: float,
    starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not -PHI_BOUND < phi < PHI_BOUND:
        raise ValueError("common AR coefficient is outside the stationary search bound")
    y_star = np.empty_like(y)
    x_star = np.empty_like(x)
    first_scale = float(np.sqrt(1.0 - phi * phi))
    for index in range(len(y)):
        if starts[index]:
            y_star[index] = first_scale * y[index]
            x_star[index] = first_scale * x[index]
        else:
            y_star[index] = y[index] - phi * y[index - 1]
            x_star[index] = x[index] - phi * x[index - 1]
    log_jacobian = 0.5 * y.shape[1] * int(starts.sum()) * np.log1p(-(phi * phi))
    return y_star, x_star, float(log_jacobian)


def _fit_at_phi(
    y: np.ndarray,
    x: np.ndarray,
    phi: float,
    starts: np.ndarray,
) -> dict[str, object]:
    y_star, x_star, log_jacobian = _transform_common_ar(y, x, phi, starts)
    coefficients, _, _, _ = np.linalg.lstsq(x_star, y_star, rcond=None)
    innovations = y_star - x_star @ coefficients
    covariance = innovations.T @ innovations / len(y)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(logdet):
        return {"log_likelihood": -np.inf}
    dimensions = y.shape[1]
    log_likelihood = (
        -0.5
        * len(y)
        * (dimensions * np.log(2.0 * np.pi) + logdet + dimensions)
        + log_jacobian
    )
    inverse_information = np.linalg.inv(x_star.T @ x_star)
    coefficient_covariance = np.kron(inverse_information, covariance)
    return {
        "coefficients": coefficients,
        "covariance": covariance,
        "coefficient_covariance": coefficient_covariance,
        "innovations": innovations,
        "log_likelihood": float(log_likelihood),
    }


def fit_common_ar(
    y: np.ndarray,
    x: np.ndarray,
    order: int,
    segment_starts: np.ndarray | None = None,
) -> CommonARFit:
    """Fit the exact stationary bivariate AR(0) or common AR(1) model."""

    y, x = _validate_arrays(y, x)
    if order not in AR_ORDERS:
        raise ValueError("H3 common AR order must be zero or one")
    starts = _segment_starts(len(y), segment_starts)
    if order == 0:
        phi = 0.0
    else:
        optimization = minimize_scalar(
            lambda candidate: -float(
                _fit_at_phi(y, x, float(candidate), starts)["log_likelihood"]
            ),
            bounds=(-PHI_BOUND + 1.0e-8, PHI_BOUND - 1.0e-8),
            method="bounded",
            options={"xatol": 1.0e-9, "maxiter": 300},
        )
        if not optimization.success or not np.isfinite(optimization.fun):
            raise RuntimeError("H3 common AR optimization did not converge")
        phi = float(optimization.x)
    fitted = _fit_at_phi(y, x, phi, starts)
    if not np.isfinite(float(fitted["log_likelihood"])):
        raise RuntimeError("H3 likelihood is not finite")
    coefficients = np.asarray(fitted["coefficients"], dtype=float)
    residuals = y - x @ coefficients
    n_parameters = coefficients.size + 3 + order
    bic = -2.0 * float(fitted["log_likelihood"]) + n_parameters * np.log(len(y))
    degrees_freedom = len(y) - x.shape[1] - order
    return CommonARFit(
        order=order,
        phi=phi,
        coefficients=coefficients,
        covariance=np.asarray(fitted["covariance"], dtype=float),
        coefficient_covariance=np.asarray(
            fitted["coefficient_covariance"], dtype=float
        ),
        residuals=residuals,
        innovations=np.asarray(fitted["innovations"], dtype=float),
        log_likelihood=float(fitted["log_likelihood"]),
        bic=float(bic),
        n_months=len(y),
        n_segments=int(starts.sum()),
        degrees_freedom=int(degrees_freedom),
    )


def build_design(
    polarization: Iterable[float],
    time_values: Iterable[float] | None = None,
) -> np.ndarray:
    polarization = np.asarray(list(polarization), dtype=float)
    time = (
        np.arange(len(polarization), dtype=float)
        if time_values is None
        else np.asarray(list(time_values), dtype=float)
    )
    if time.shape != polarization.shape:
        raise ValueError("time values do not align with polarization")
    return np.column_stack(
        (
            np.ones(len(polarization)),
            polarization - float(np.mean(polarization)),
            time - float(np.mean(time)),
        )
    )


def _coefficient_variance(fit: CommonARFit, term: int, party: int) -> float:
    index = term * 2 + party
    return float(fit.coefficient_covariance[index, index])


def _coefficient_covariance(
    fit: CommonARFit,
    term_a: int,
    party_a: int,
    term_b: int,
    party_b: int,
) -> float:
    index_a = term_a * 2 + party_a
    index_b = term_b * 2 + party_b
    return float(fit.coefficient_covariance[index_a, index_b])


def coefficient_table(fits: dict[int, CommonARFit]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for order, fit in fits.items():
        critical = float(student_t.ppf(0.975, fit.degrees_freedom))
        for term_index, term in enumerate(TERMS):
            for party_index, party in enumerate(PARTIES):
                estimate = float(fit.coefficients[term_index, party_index])
                standard_error = float(
                    np.sqrt(_coefficient_variance(fit, term_index, party_index))
                )
                statistic = estimate / standard_error
                rows.append(
                    {
                        "ar_order": order,
                        "party": party,
                        "term": term,
                        "estimate": estimate,
                        "standard_error_conditional": standard_error,
                        "t_statistic_conditional": statistic,
                        "degrees_freedom": fit.degrees_freedom,
                        "p_two_sided_conditional": float(
                            2.0 * student_t.sf(abs(statistic), fit.degrees_freedom)
                        ),
                        "p_one_sided_negative_conditional": float(
                            student_t.cdf(statistic, fit.degrees_freedom)
                        ),
                        "ci_lower_conditional": estimate - critical * standard_error,
                        "ci_upper_conditional": estimate + critical * standard_error,
                    }
                )
    return pd.DataFrame(rows)


def contrast_table(fits: dict[int, CommonARFit]) -> pd.DataFrame:
    rows = []
    term_index = TERMS.index("polarization")
    for order, fit in fits.items():
        estimate = float(
            fit.coefficients[term_index, 0] - fit.coefficients[term_index, 1]
        )
        variance = (
            _coefficient_variance(fit, term_index, 0)
            + _coefficient_variance(fit, term_index, 1)
            - 2.0
            * _coefficient_covariance(fit, term_index, 0, term_index, 1)
        )
        standard_error = float(np.sqrt(max(variance, 0.0)))
        statistic = estimate / standard_error
        critical = float(student_t.ppf(0.975, fit.degrees_freedom))
        rows.append(
            {
                "ar_order": order,
                "contrast": "republican_minus_democrat_polarization_coefficient",
                "estimate": estimate,
                "standard_error_conditional": standard_error,
                "t_statistic_conditional": statistic,
                "degrees_freedom": fit.degrees_freedom,
                "p_two_sided_conditional": float(
                    2.0 * student_t.sf(abs(statistic), fit.degrees_freedom)
                ),
                "ci_lower_conditional": estimate - critical * standard_error,
                "ci_upper_conditional": estimate + critical * standard_error,
            }
        )
    return pd.DataFrame(rows)


def _prepare_point_data(
    polarization_file: Path,
    coherence_file: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    polarization = pd.read_csv(polarization_file)
    polarization = polarization.loc[polarization["dimension"].eq("q1")].copy()
    polarization = polarization.sort_values("year_month", kind="stable")
    coherence = pd.read_csv(coherence_file)
    pivot = coherence.pivot(index="year_month", columns="party", values="raw_iqr")
    frame = polarization[["year_month", "polarization"]].merge(
        pivot.reset_index(), on="year_month", how="inner", validate="one_to_one"
    )
    frame = frame.sort_values("year_month", kind="stable").reset_index(drop=True)
    if not np.array_equal(frame["year_month"].astype(str), EXPECTED_MONTHS):
        raise ValueError("H3 point inputs do not contain the complete calendar")
    y = frame[["republican", "democrat"]].to_numpy(dtype=float)
    x = build_design(frame["polarization"].to_numpy(dtype=float))
    return frame, y, x


def _prepare_replicates(replicate_file: Path) -> pd.DataFrame:
    columns = [
        "replicate",
        "year_month",
        "dimension",
        "polarization",
        "rep_raw_iqr",
        "dem_raw_iqr",
    ]
    frame = pd.read_csv(replicate_file, usecols=columns)
    frame = frame.loc[frame["dimension"].eq("q1")].copy()
    frame = frame.sort_values(["replicate", "year_month"], kind="stable")
    counts = frame.groupby("replicate", sort=True).size()
    if len(counts) != 5000 or not counts.eq(34).all():
        raise ValueError("H3 measurement replicate contract must contain 5,000 by 34 rows")
    if not np.array_equal(
        frame["replicate"].drop_duplicates().to_numpy(dtype=int),
        np.arange(1, 5001),
    ):
        raise ValueError("H3 replicate identifiers are incomplete")
    valid_panels = frame.groupby("replicate", sort=True)[columns[3:]].apply(
        lambda group: bool(group.notna().all(axis=None))
    )
    if float(valid_panels.mean()) < 0.95:
        raise ValueError("fewer than 95 percent of H3 replicate panels are complete")
    return frame


def _fit_measurement_replicates(
    replicates: pd.DataFrame,
    selected_order: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for replicate, group in replicates.groupby("replicate", sort=True):
        if group[["polarization", "rep_raw_iqr", "dem_raw_iqr"]].isna().any(
            axis=None
        ):
            continue
        y = group[["rep_raw_iqr", "dem_raw_iqr"]].to_numpy(dtype=float)
        x = build_design(group["polarization"].to_numpy(dtype=float))
        by_order = {order: fit_common_ar(y, x, order) for order in AR_ORDERS}
        preferred_order = int(by_order[1].bic < by_order[0].bic)
        for order, fit in by_order.items():
            rows.append(
                {
                    "replicate": int(replicate),
                    "ar_order": order,
                    "primary_fixed_order": order == selected_order,
                    "phi": fit.phi,
                    "bic_ar0": by_order[0].bic,
                    "bic_ar1": by_order[1].bic,
                    "bic_preferred_ar_order": preferred_order,
                    "republican_intercept": fit.coefficients[0, 0],
                    "republican_polarization": fit.coefficients[1, 0],
                    "republican_time": fit.coefficients[2, 0],
                    "democrat_intercept": fit.coefficients[0, 1],
                    "democrat_polarization": fit.coefficients[1, 1],
                    "democrat_time": fit.coefficients[2, 1],
                    "polarization_effect_contrast": (
                        fit.coefficients[1, 0] - fit.coefficients[1, 1]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _measurement_summary(replicates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for order, group in replicates.groupby("ar_order", sort=True):
        for party in PARTIES:
            for term in TERMS:
                column = f"{party}_{term}"
                values = group[column].to_numpy(dtype=float)
                rows.append(
                    {
                        "ar_order": int(order),
                        "party": party,
                        "term": term,
                        "measurement_mean": float(np.mean(values)),
                        "measurement_standard_error": float(np.std(values, ddof=1)),
                        "measurement_ci_lower": float(np.percentile(values, 2.5)),
                        "measurement_ci_upper": float(np.percentile(values, 97.5)),
                        "measurement_probability_negative": float(
                            np.mean(values < 0.0)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _influence_table(frame: pd.DataFrame, selected_order: int) -> pd.DataFrame:
    rows = []
    full_index = np.arange(len(frame))
    for omitted in full_index:
        retained = full_index != omitted
        reduced = frame.loc[retained].reset_index(drop=True)
        original_positions = full_index[retained]
        starts = np.zeros(len(reduced), dtype=bool)
        starts[0] = True
        starts[1:] = np.diff(original_positions) > 1
        y = reduced[["republican", "democrat"]].to_numpy(dtype=float)
        x = build_design(
            reduced["polarization"].to_numpy(dtype=float),
            time_values=original_positions,
        )
        fit = fit_common_ar(y, x, selected_order, segment_starts=starts)
        rows.append(
            {
                "omitted_month": str(frame.loc[omitted, "year_month"]),
                "ar_order": selected_order,
                "phi": fit.phi,
                "republican_polarization": fit.coefficients[1, 0],
                "democrat_polarization": fit.coefficients[1, 1],
                "polarization_effect_contrast": (
                    fit.coefficients[1, 0] - fit.coefficients[1, 1]
                ),
            }
        )
    return pd.DataFrame(rows)


def _residual_table(fits: dict[int, CommonARFit]) -> pd.DataFrame:
    rows = []
    for order, fit in fits.items():
        for party_index, party in enumerate(PARTIES):
            values = fit.innovations[:, party_index]
            correlations = acf(values, nlags=8, fft=False)
            for lag in (4, 8):
                lb = acorr_ljungbox(values, lags=[lag], return_df=True).iloc[0]
                rows.append(
                    {
                        "ar_order": order,
                        "party": party,
                        "diagnostic_lag": lag,
                        "innovation_acf_at_lag": float(correlations[lag]),
                        "ljung_box_statistic": float(lb["lb_stat"]),
                        "ljung_box_p_value": float(lb["lb_pvalue"]),
                        "contemporaneous_innovation_correlation": float(
                            np.corrcoef(fit.innovations.T)[0, 1]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run_h3_analysis(
    polarization_file: Path | str,
    coherence_file: Path | str,
    measurement_replicate_file: Path | str,
    output_dir: Path | str,
) -> dict[str, pd.DataFrame]:
    """Fit, propagate measurement uncertainty, and write the H3 contract."""

    polarization_file = Path(polarization_file)
    coherence_file = Path(coherence_file)
    measurement_replicate_file = Path(measurement_replicate_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, y, x = _prepare_point_data(polarization_file, coherence_file)
    fits = {order: fit_common_ar(y, x, order) for order in AR_ORDERS}
    selected_order = min(fits, key=lambda order: fits[order].bic)
    model_rows = []
    for order, fit in fits.items():
        model_rows.append(
            {
                "ar_order": order,
                "phi": fit.phi,
                "log_likelihood": fit.log_likelihood,
                "bic": fit.bic,
                "delta_bic": fit.bic - fits[selected_order].bic,
                "selected": order == selected_order,
                "n_months": fit.n_months,
                "n_parameters": 6 + 3 + order,
                "residual_covariance_rr": fit.covariance[0, 0],
                "residual_covariance_rd": fit.covariance[0, 1],
                "residual_covariance_dd": fit.covariance[1, 1],
            }
        )
    models = pd.DataFrame(model_rows)
    coefficients = coefficient_table(fits)
    contrasts = contrast_table(fits)
    measurement_input = _prepare_replicates(measurement_replicate_file)
    measurement_coefficients = _fit_measurement_replicates(
        measurement_input, selected_order
    )
    measurement_summary = _measurement_summary(measurement_coefficients)
    coefficients = coefficients.merge(
        measurement_summary,
        on=["ar_order", "party", "term"],
        how="left",
        validate="one_to_one",
    )
    contrast_summaries = []
    for order, group in measurement_coefficients.groupby("ar_order", sort=True):
        values = group["polarization_effect_contrast"].to_numpy(dtype=float)
        contrast_summaries.append(
            {
                "ar_order": int(order),
                "measurement_mean": float(np.mean(values)),
                "measurement_standard_error": float(np.std(values, ddof=1)),
                "measurement_ci_lower": float(np.percentile(values, 2.5)),
                "measurement_ci_upper": float(np.percentile(values, 97.5)),
            }
        )
    contrasts = contrasts.merge(
        pd.DataFrame(contrast_summaries),
        on="ar_order",
        how="left",
        validate="one_to_one",
    )
    influence = _influence_table(frame, selected_order)
    residuals = _residual_table(fits)

    selected_effects = coefficients.loc[
        coefficients["ar_order"].eq(selected_order)
        & coefficients["term"].eq("polarization")
    ].set_index("party")
    symmetric_support = bool(
        all(
            selected_effects.loc[party, "estimate"] < 0
            and selected_effects.loc[party, "ci_upper_conditional"] < 0
            and selected_effects.loc[party, "measurement_ci_upper"] < 0
            for party in PARTIES
        )
    )
    sign_pattern = {
        party: (
            "negative"
            if selected_effects.loc[party, "estimate"] < 0
            else "positive"
            if selected_effects.loc[party, "estimate"] > 0
            else "zero"
        )
        for party in PARTIES
    }
    ar_sign_invariant = all(
        np.sign(fits[0].coefficients[1, party_index])
        == np.sign(fits[1].coefficients[1, party_index])
        for party_index in range(2)
    )

    paths = {
        "models": output_dir / "h3_model_comparison.csv",
        "coefficients": output_dir / "h3_coefficients.csv",
        "contrasts": output_dir / "h3_party_contrast.csv",
        "replicates": output_dir / "h3_measurement_replicate_coefficients.csv",
        "influence": output_dir / "h3_influence_diagnostics.csv",
        "residuals": output_dir / "h3_residual_diagnostics.csv",
    }
    for key, data in (
        ("models", models),
        ("coefficients", coefficients),
        ("contrasts", contrasts),
        ("replicates", measurement_coefficients),
        ("influence", influence),
        ("residuals", residuals),
    ):
        data.to_csv(paths[key], index=False)

    metadata = {
        "schema": H3_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "generated_pending_independent_validation",
        "period": ["2022-03", "2024-12"],
        "n_months": 34,
        "outcomes": ["republican_raw_iqr", "democrat_raw_iqr"],
        "formula": "raw_iqr ~ polarization * party + time * party",
        "error_structure": "unrestricted contemporaneous covariance and common scalar AR order",
        "ar_orders_compared": list(AR_ORDERS),
        "selected_ar_order": int(selected_order),
        "selection_criterion": "observed-data BIC over 34 monthly vector observations",
        "measurement_rule": "selected observed-data AR order fixed across 5,000 joint article-cluster replicates",
        "measurement_replicates_requested": 5000,
        "measurement_replicates_valid": int(
            measurement_coefficients["replicate"].nunique()
        ),
        "symmetric_h3_rule": "both party polarization coefficients negative with conditional and measurement intervals below zero",
        "symmetric_h3_supported": symmetric_support,
        "selected_sign_pattern": sign_pattern,
        "coefficient_signs_invariant_across_ar_orders": ar_sign_invariant,
        "important_limitations": [
            "association rather than causation",
            "conditional model intervals do not include temporal model-selection uncertainty",
            "measurement percentile intervals quantify article-sampling uncertainty",
            "the sample contains 34 monthly observations",
        ],
        "input_sha256": {
            str(polarization_file.resolve()): sha256_file(polarization_file),
            str(coherence_file.resolve()): sha256_file(coherence_file),
            str(measurement_replicate_file.resolve()): sha256_file(
                measurement_replicate_file
            ),
        },
        "output_sha256": {
            path.name: sha256_file(path) for path in paths.values()
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    metadata_path = output_dir / "h3_analysis_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "models": models,
        "coefficients": coefficients,
        "contrasts": contrasts,
        "replicates": measurement_coefficients,
        "influence": influence,
        "residuals": residuals,
    }
