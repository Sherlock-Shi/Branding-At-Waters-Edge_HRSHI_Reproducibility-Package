#!/usr/bin/env python3
"""Compare gradual trend, persistent shifts, and their joint specification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from scripts import joint_ar_breakpoint_analysis as joint  # noqa: E402
from src.measurement_bootstrap import derive_child_seed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rpy2.robjects as ro  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
STATISTICS_DIR = (
    REPO_ROOT
    / "UKRAID-Politicization"
    / "Factiva_Data"
    / "pipeline_c"
    / "llm_statistics"
)
POLARIZATION_FILE = (
    REPO_ROOT
    / "UKRAID-Politicization"
    / "Factiva_Data"
    / "pipeline_c"
    / "llm_aggregated"
    / "monthly_polarization.csv"
)
OUTPUT_SUBDIR = "trend_break_comparison"
MODEL_CLASSES = ("trend_only", "break_only", "break_plus_trend")
MASTER_SEED = 42
RUNS_PER_MODE = 5
PRIMARY_MINIMUM_DURATION = 3


R_CODE = r"""
paper3_trend_shift_bic <- function(chromosome, plen=1, XMat, Xt, minseg) {
  N <- length(Xt)
  m <- as.integer(chromosome[1])
  p <- as.integer(chromosome[2])
  if (m < 1L) return(1e12)
  tau <- as.integer(chromosome[3:(2 + m)])
  seglen <- diff(c(1L, tau, N + 1L))
  if (length(seglen) != m + 1L || any(seglen < minseg)) return(1e12)
  CpMat <- sapply(tau, function(cp) as.numeric(seq_len(N) >= cp))
  if (m == 1L) CpMat <- matrix(CpMat, ncol=1)
  DesignX <- cbind(XMat, CpMat)
  fit <- try(
    stats::arima(
      Xt, order=c(p, 0, 0), xreg=DesignX, include.mean=FALSE,
      method="ML", optim.control=list(maxit=1000)
    ),
    silent=TRUE
  )
  if (inherits(fit, "try-error") || !is.finite(stats::BIC(fit))) return(1e12)
  as.numeric(stats::BIC(fit) + m * log(N))
}

paper3_parse_trend_suggestions <- function(text) {
  if (is.null(text) || !nzchar(text)) return(NULL)
  items <- strsplit(text, ";", fixed=TRUE)[[1]]
  lapply(items, function(item) {
    if (!nzchar(item)) return(NULL)
    as.integer(strsplit(item, "|", fixed=TRUE)[[1]])
  })
}

paper3_run_trend_shift_ga <- function(y, minseg, seed, suggestion_text="") {
  N <- length(y)
  centered_time <- seq_len(N) - mean(seq_len(N))
  XMat <- cbind(intercept=rep(1, N), time=centered_time)
  mmax <- floor(N / minseg) - 1L
  suggestions <- paper3_parse_trend_suggestions(suggestion_text)
  set.seed(seed)
  initial_population <- changepointGA::random_population(
    popSize=%d,
    prange=list(ar=c(0, 2)),
    N=N,
    minDist=minseg,
    pchangepoint=%s,
    mmax=mmax,
    lmax=3L + mmax
  )
  if (!is.null(suggestions)) {
    seeded <- list()
    for (tau in suggestions) {
      if (length(tau) < 1L) next
      for (p in 0:2) seeded[[length(seeded) + 1L]] <- list(tau=tau, p=p)
    }
    for (i in seq_len(min(length(seeded), ncol(initial_population)))) {
      tau <- seeded[[i]]$tau
      p <- seeded[[i]]$p
      chrom <- rep(0, nrow(initial_population))
      chrom[1] <- length(tau)
      chrom[2] <- p
      chrom[3:(2 + length(tau))] <- tau
      chrom[3 + length(tau)] <- N + 1L
      initial_population[, i] <- chrom
    }
  }
  population_initializer <- function(...) initial_population
  result <- changepointGA::cptgaisl(
    ObjFunc=paper3_trend_shift_bic,
    N=N,
    prange=list(ar=c(0, 2)),
    popSize=%d,
    numIslands=%d,
    pchangepoint=%s,
    minDist=minseg,
    mmax=mmax,
    lmax=3L + mmax,
    maxMig=%d,
    maxgen=%d,
    maxconv=%d,
    option="both",
    monitoring=FALSE,
    parallel=FALSE,
    seed=seed,
    popInitialize=population_initializer,
    suggestions=NULL,
    XMat=XMat,
    Xt=y,
    minseg=minseg
  )
  list(
    chromosome=as.numeric(result@overbestchrom),
    objective=as.numeric(result@overbestfit),
    convergence_counter=as.numeric(result@convg)
  )
}

paper3_fit_trend_shift_config <- function(y, p, tau, include_trend=TRUE) {
  N <- length(y)
  m <- length(tau)
  centered_time <- seq_len(N) - mean(seq_len(N))
  DesignX <- matrix(1, nrow=N, ncol=1)
  colnames(DesignX) <- "intercept"
  if (include_trend) DesignX <- cbind(DesignX, time=centered_time)
  if (m > 0) {
    CpMat <- sapply(tau, function(cp) as.numeric(seq_len(N) >= cp))
    if (m == 1) CpMat <- matrix(CpMat, ncol=1)
    colnames(CpMat) <- paste0("shift_at_", tau)
    DesignX <- cbind(DesignX, CpMat)
  }
  fit <- stats::arima(
    y, order=c(p, 0, 0), xreg=DesignX, include.mean=FALSE,
    method="ML", optim.control=list(maxit=2000)
  )
  all_coef <- stats::coef(fit)
  beta <- tail(all_coef, ncol(DesignX))
  all_vcov <- stats::vcov(fit)
  xreg_indices <- (length(all_coef) - ncol(DesignX) + 1L):length(all_coef)
  beta_vcov <- all_vcov[xreg_indices, xreg_indices, drop=FALSE]
  structural <- as.numeric(DesignX %%*%% beta)
  innovations <- as.numeric(stats::residuals(fit))
  raw_bic <- as.numeric(stats::BIC(fit))
  list(
    log_likelihood=as.numeric(stats::logLik(fit)),
    raw_bic=raw_bic,
    location_penalty=m * log(N),
    objective=raw_bic + m * log(N),
    sigma2=as.numeric(fit$sigma2),
    ar_coefficients=if (p > 0) as.numeric(all_coef[seq_len(p)]) else numeric(0),
    beta=as.numeric(beta),
    beta_names=names(beta),
    beta_vcov=beta_vcov,
    structural_mean=structural,
    innovations=innovations,
    one_step_fitted=as.numeric(y - innovations)
  )
}
""" % (
    joint.POP_SIZE,
    joint.INITIAL_CHANGE_PROBABILITY,
    joint.POP_SIZE,
    joint.NUM_ISLANDS,
    joint.INITIAL_CHANGE_PROBABILITY,
    joint.MAX_MIGRATIONS,
    joint.GENERATIONS_PER_MIGRATION,
    joint.CONVERGENCE_MIGRATIONS,
)

ro.r(R_CODE)
RUN_TREND_SHIFT_GA = ro.globalenv["paper3_run_trend_shift_ga"]
FIT_TREND_SHIFT = ro.globalenv["paper3_fit_trend_shift_config"]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit(
    values: np.ndarray,
    order: int,
    tau: tuple[int, ...],
    include_trend: bool,
) -> dict[str, object]:
    result = FIT_TREND_SHIFT(
        ro.FloatVector(values.tolist()),
        int(order),
        ro.IntVector(list(tau)),
        bool(include_trend),
    )
    beta_vcov = np.asarray(result.rx2("beta_vcov"), dtype=float).copy()
    return {
        "log_likelihood": float(result.rx2("log_likelihood")[0]),
        "raw_bic": float(result.rx2("raw_bic")[0]),
        "location_penalty": float(result.rx2("location_penalty")[0]),
        "objective": float(result.rx2("objective")[0]),
        "sigma2": float(result.rx2("sigma2")[0]),
        "ar_coefficients": np.asarray(
            result.rx2("ar_coefficients"), dtype=float
        ).copy(),
        "beta": np.asarray(result.rx2("beta"), dtype=float).copy(),
        "beta_names": [str(item) for item in result.rx2("beta_names")],
        "beta_vcov": beta_vcov,
        "structural_mean": np.asarray(
            result.rx2("structural_mean"), dtype=float
        ).copy(),
        "innovations": np.asarray(result.rx2("innovations"), dtype=float).copy(),
        "one_step_fitted": np.asarray(
            result.rx2("one_step_fitted"), dtype=float
        ).copy(),
    }


def _load_polarization(path: Path) -> tuple[pd.Series, np.ndarray]:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["dimension"].eq("q1")].sort_values(
        "year_month", kind="stable"
    )
    expected = pd.period_range("2022-03", "2024-12", freq="M").astype(str)
    if len(frame) != 34 or not np.array_equal(frame["year_month"].astype(str), expected):
        raise ValueError("trend-break input does not contain the complete calendar")
    values = frame["polarization"].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("trend-break input contains non-finite values")
    dates = pd.to_datetime(frame["year_month"], format="%Y-%m")
    return dates.reset_index(drop=True), values


def _seed(minimum_duration: int, mode: str, replicate: int) -> int:
    namespace = (
        f"trend_break_joint_search_minimum_{minimum_duration}_{mode}_{replicate}"
    )
    return int(derive_child_seed(MASTER_SEED, namespace) % (2**31 - 2) + 1)


def _structured_suggestions(minimum_duration: int, n: int) -> tuple[str, int]:
    text, _ = joint._suggestions("Polarization", minimum_duration, n)
    candidates = [item for item in text.split(";") if item != "none"]
    return ";".join(candidates), len(candidates)


def _search_break_plus_trend(
    dates: pd.Series,
    values: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    total = len(joint.MIN_SEGLENS) * 2 * RUNS_PER_MODE
    completed = 0
    for minimum_duration in joint.MIN_SEGLENS:
        suggestion_text, suggestion_count = _structured_suggestions(
            minimum_duration, len(values)
        )
        for mode in ("structured", "random"):
            for replicate in range(1, RUNS_PER_MODE + 1):
                seed = _seed(minimum_duration, mode, replicate)
                supplied = suggestion_text if mode == "structured" else ""
                result = RUN_TREND_SHIFT_GA(
                    ro.FloatVector(values.tolist()),
                    int(minimum_duration),
                    int(seed),
                    supplied,
                )
                chromosome, objective, counter = joint._extract_ga_result(result)
                count, order, tau = joint._decode_chromosome(chromosome)
                if count < 1 or order not in joint.AR_ORDERS or not joint._valid_tau(
                    tau, len(values), minimum_duration
                ):
                    raise RuntimeError("trend-plus-shift optimizer returned an invalid model")
                refit = _fit(values, order, tau, include_trend=True)
                if not np.isclose(
                    objective,
                    refit["objective"],
                    atol=joint.OBJECTIVE_TOLERANCE,
                    rtol=0,
                ):
                    raise RuntimeError("trend-plus-shift optimizer and refit disagree")
                records.append(
                    {
                        "minimum_regime_duration": minimum_duration,
                        "initialization": mode,
                        "replicate": replicate,
                        "seed": seed,
                        "suggestion_count": suggestion_count if mode == "structured" else 0,
                        "n_breakpoints": count,
                        "ar_order": order,
                        "break_onset_indices_1based": joint._tau_signature(tau),
                        "break_onset_dates": joint._date_signature(dates, tau),
                        "joint_bic": objective,
                        "log_likelihood": refit["log_likelihood"],
                        "convergence_counter": counter,
                    }
                )
                completed += 1
                print(
                    f"[{completed}/{total}] duration={minimum_duration} {mode} "
                    f"K={count}, AR({order}), BIC={objective:.6f}",
                    flush=True,
                )
    runs = pd.DataFrame(records)
    runs["configuration"] = (
        runs["n_breakpoints"].astype(str)
        + "|"
        + runs["ar_order"].astype(str)
        + "|"
        + runs["break_onset_indices_1based"].fillna("")
    )
    winners = []
    for minimum_duration, group in runs.groupby("minimum_regime_duration", sort=True):
        best = float(group["joint_bic"].min())
        rows = group.loc[np.isclose(group["joint_bic"], best, atol=1.0e-5, rtol=0)]
        winner = rows.sort_values(["initialization", "seed"], kind="stable").iloc[0].copy()
        winner["runs_reaching_best"] = len(rows)
        winner["total_runs"] = len(group)
        winner["distinct_configurations"] = group["configuration"].nunique()
        winners.append(winner)
    return runs, pd.DataFrame(winners)


def _parse_tau(value: object) -> tuple[int, ...]:
    if pd.isna(value) or str(value).strip() == "":
        return ()
    return tuple(int(item) for item in str(value).split("|") if item)


def _coefficient_rows(
    model_class: str,
    minimum_duration: int,
    order: int,
    tau: tuple[int, ...],
    fit: dict[str, object],
) -> list[dict[str, object]]:
    beta = np.asarray(fit["beta"], dtype=float)
    covariance = np.asarray(fit["beta_vcov"], dtype=float)
    degrees_freedom = 34 - len(beta) - order
    critical = float(student_t.ppf(0.975, degrees_freedom))
    rows = []
    for index, name in enumerate(fit["beta_names"]):
        standard_error = float(np.sqrt(covariance[index, index]))
        statistic = float(beta[index] / standard_error)
        rows.append(
            {
                "model_class": model_class,
                "minimum_regime_duration": minimum_duration,
                "ar_order": order,
                "break_onset_indices_1based": joint._tau_signature(tau),
                "term": name,
                "estimate": beta[index],
                "standard_error_conditional": standard_error,
                "t_statistic_conditional": statistic,
                "degrees_freedom": degrees_freedom,
                "p_two_sided_conditional": float(
                    2.0 * student_t.sf(abs(statistic), degrees_freedom)
                ),
                "ci_lower_conditional": beta[index] - critical * standard_error,
                "ci_upper_conditional": beta[index] + critical * standard_error,
            }
        )
    return rows


def _endpoint_row(
    role: str,
    model_class: str,
    minimum_duration: int,
    order: int,
    tau: tuple[int, ...],
    fit: dict[str, object],
) -> dict[str, object]:
    names = list(fit["beta_names"])
    contrast = np.zeros(len(names), dtype=float)
    if "time" in names:
        contrast[names.index("time")] = 33.0
    for index, name in enumerate(names):
        if name.startswith("shift_at_"):
            contrast[index] = 1.0
    beta = np.asarray(fit["beta"], dtype=float)
    covariance = np.asarray(fit["beta_vcov"], dtype=float)
    estimate = float(contrast @ beta)
    standard_error = float(np.sqrt(contrast @ covariance @ contrast))
    degrees_freedom = 34 - len(beta) - order
    critical = float(student_t.ppf(0.975, degrees_freedom))
    statistic = estimate / standard_error
    return {
        "model_role": role,
        "model_class": model_class,
        "minimum_regime_duration": minimum_duration,
        "ar_order": order,
        "break_onset_indices_1based": joint._tau_signature(tau),
        "estimate_end_minus_start": estimate,
        "standard_error_conditional": standard_error,
        "t_statistic_conditional": statistic,
        "degrees_freedom": degrees_freedom,
        "p_two_sided_conditional": float(
            2.0 * student_t.sf(abs(statistic), degrees_freedom)
        ),
        "ci_lower_conditional": estimate - critical * standard_error,
        "ci_upper_conditional": estimate + critical * standard_error,
        "selection_uncertainty_included": False,
    }


def _s_decomposition(values: np.ndarray, tau: tuple[int, ...]) -> pd.DataFrame:
    regime = np.zeros(len(values), dtype=int)
    for breakpoint in tau:
        regime[np.arange(len(values)) >= breakpoint - 1] += 1
    counts = {
        "within_regime": {"positive": 0, "negative": 0, "ties": 0},
        "between_regime": {"positive": 0, "negative": 0, "ties": 0},
    }
    for first in range(len(values) - 1):
        for second in range(first + 1, len(values)):
            category = (
                "within_regime"
                if regime[first] == regime[second]
                else "between_regime"
            )
            difference = float(values[second] - values[first])
            if difference > 0:
                counts[category]["positive"] += 1
            elif difference < 0:
                counts[category]["negative"] += 1
            else:
                counts[category]["ties"] += 1
    rows = []
    for category, values_by_sign in counts.items():
        rows.append(
            {
                "pair_category": category,
                **values_by_sign,
                "pairs": int(sum(values_by_sign.values())),
                "mann_kendall_s_contribution": int(
                    values_by_sign["positive"] - values_by_sign["negative"]
                ),
            }
        )
    total_positive = sum(item["positive"] for item in counts.values())
    total_negative = sum(item["negative"] for item in counts.values())
    total_ties = sum(item["ties"] for item in counts.values())
    rows.append(
        {
            "pair_category": "total",
            "positive": total_positive,
            "negative": total_negative,
            "ties": total_ties,
            "pairs": total_positive + total_negative + total_ties,
            "mann_kendall_s_contribution": total_positive - total_negative,
        }
    )
    return pd.DataFrame(rows)


def run_trend_break_analysis(
    polarization_file: Path | str,
    statistics_dir: Path | str,
) -> dict[str, pd.DataFrame]:
    polarization_file = Path(polarization_file)
    statistics_dir = Path(statistics_dir)
    output_dir = statistics_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    dates, values = _load_polarization(polarization_file)

    search_runs, trend_winners = _search_break_plus_trend(dates, values)
    existing_winners = pd.read_csv(
        statistics_dir
        / "joint_ar_breakpoints"
        / "joint_ar_bic_winners.csv"
    )
    existing_winners = existing_winners.loc[
        existing_winners["series"].eq("Polarization")
    ]

    trend_only_fits = {
        order: _fit(values, order, (), include_trend=True)
        for order in joint.AR_ORDERS
    }
    trend_only_order = min(
        trend_only_fits, key=lambda order: trend_only_fits[order]["objective"]
    )
    trend_only_fit = trend_only_fits[trend_only_order]

    comparison_rows = []
    coefficient_rows = []
    fit_registry: dict[tuple[int, str], tuple[int, tuple[int, ...], dict[str, object]]] = {}
    for minimum_duration in joint.MIN_SEGLENS:
        break_row = existing_winners.loc[
            existing_winners["minseglen"].eq(minimum_duration)
        ].iloc[0]
        break_tau = _parse_tau(break_row["break_onset_indices_1based"])
        break_order = int(break_row["ar_order"])
        break_fit = _fit(values, break_order, break_tau, include_trend=False)
        if not np.isclose(
            break_fit["objective"],
            float(break_row["ga_objective_bic"]),
            atol=joint.OBJECTIVE_TOLERANCE,
            rtol=0,
        ):
            raise RuntimeError("break-only refit does not match the selected search")

        trend_row = trend_winners.loc[
            trend_winners["minimum_regime_duration"].eq(minimum_duration)
        ].iloc[0]
        trend_tau = _parse_tau(trend_row["break_onset_indices_1based"])
        trend_order = int(trend_row["ar_order"])
        trend_fit = _fit(values, trend_order, trend_tau, include_trend=True)
        fits = {
            "trend_only": (trend_only_order, (), trend_only_fit),
            "break_only": (break_order, break_tau, break_fit),
            "break_plus_trend": (trend_order, trend_tau, trend_fit),
        }
        minimum_bic = min(item[2]["objective"] for item in fits.values())
        for model_class, (order, tau, fit) in fits.items():
            comparison_rows.append(
                {
                    "minimum_regime_duration": minimum_duration,
                    "model_class": model_class,
                    "ar_order": order,
                    "n_breakpoints": len(tau),
                    "break_onset_indices_1based": joint._tau_signature(tau),
                    "break_onset_dates": joint._date_signature(dates, tau),
                    "log_likelihood": fit["log_likelihood"],
                    "joint_bic": fit["objective"],
                    "delta_bic_within_duration": fit["objective"] - minimum_bic,
                    "selected_within_duration": np.isclose(
                        fit["objective"], minimum_bic, atol=1.0e-8
                    ),
                }
            )
            coefficient_rows.extend(
                _coefficient_rows(
                    model_class, minimum_duration, order, tau, fit
                )
            )
            fit_registry[(minimum_duration, model_class)] = (order, tau, fit)

    comparison = pd.DataFrame(comparison_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    primary = comparison.loc[
        comparison["minimum_regime_duration"].eq(PRIMARY_MINIMUM_DURATION)
    ]
    selected_class = str(
        primary.loc[primary["selected_within_duration"], "model_class"].iloc[0]
    )
    selected_order, selected_tau, selected_fit = fit_registry[
        (PRIMARY_MINIMUM_DURATION, selected_class)
    ]
    joint_order, joint_tau, joint_fit = fit_registry[
        (PRIMARY_MINIMUM_DURATION, "break_plus_trend")
    ]

    endpoints = pd.DataFrame(
        [
            _endpoint_row(
                "selected_h1b_class",
                selected_class,
                PRIMARY_MINIMUM_DURATION,
                selected_order,
                selected_tau,
                selected_fit,
            ),
            _endpoint_row(
                "general_break_plus_trend",
                "break_plus_trend",
                PRIMARY_MINIMUM_DURATION,
                joint_order,
                joint_tau,
                joint_fit,
            ),
        ]
    ).drop_duplicates(
        ["model_class", "minimum_regime_duration", "ar_order", "break_onset_indices_1based"]
    )

    decomposition = _s_decomposition(values, selected_tau)
    if int(decomposition.loc[decomposition["pair_category"].eq("total"), "mann_kendall_s_contribution"].iloc[0]) != 185:
        raise RuntimeError("Mann-Kendall S decomposition does not reproduce S=185")

    fitted_rows = []
    residual_rows = []
    for model_class in MODEL_CLASSES:
        order, tau, fit = fit_registry[(PRIMARY_MINIMUM_DURATION, model_class)]
        fitted_rows.append(
            pd.DataFrame(
                {
                    "model_class": model_class,
                    "year_month": dates.dt.strftime("%Y-%m"),
                    "observed": values,
                    "structural_mean": fit["structural_mean"],
                    "one_step_fitted": fit["one_step_fitted"],
                    "innovation": fit["innovations"],
                }
            )
        )
        for lag in (4, 8):
            diagnostic = acorr_ljungbox(
                fit["innovations"], lags=[lag], return_df=True
            ).iloc[0]
            residual_rows.append(
                {
                    "model_class": model_class,
                    "minimum_regime_duration": PRIMARY_MINIMUM_DURATION,
                    "ar_order": order,
                    "diagnostic_lag": lag,
                    "ljung_box_statistic": float(diagnostic["lb_stat"]),
                    "ljung_box_p_value": float(diagnostic["lb_pvalue"]),
                }
            )
    fitted = pd.concat(fitted_rows, ignore_index=True)
    residuals = pd.DataFrame(residual_rows)

    paths = {
        "search_runs": output_dir / "trend_break_search_runs.csv",
        "search_winners": output_dir / "trend_break_search_winners.csv",
        "comparison": statistics_dir / "trend_break_model_comparison.csv",
        "coefficients": statistics_dir / "trend_break_coefficients.csv",
        "endpoints": statistics_dir / "fitted_endpoint_contrast.csv",
        "decomposition": statistics_dir / "mann_kendall_s_decomposition.csv",
        "fitted": statistics_dir / "trend_break_fitted_series.csv",
        "residuals": statistics_dir / "trend_break_residual_diagnostics.csv",
    }
    for key, data in (
        ("search_runs", search_runs),
        ("search_winners", trend_winners),
        ("comparison", comparison),
        ("coefficients", coefficients),
        ("endpoints", endpoints),
        ("decomposition", decomposition),
        ("fitted", fitted),
        ("residuals", residuals),
    ):
        data.to_csv(paths[key], index=False)

    selected_joint_time = coefficients.loc[
        coefficients["minimum_regime_duration"].eq(PRIMARY_MINIMUM_DURATION)
        & coefficients["model_class"].eq("break_plus_trend")
        & coefficients["term"].eq("time")
    ].iloc[0]
    metadata = {
        "schema": "paper3-trend-break-comparison-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "generated_pending_independent_validation",
        "model_classes": list(MODEL_CLASSES),
        "criterion": "common maximum-likelihood AR model with location-adjusted BIC",
        "minimum_regime_durations": list(joint.MIN_SEGLENS),
        "primary_minimum_regime_duration": PRIMARY_MINIMUM_DURATION,
        "selected_model_class": selected_class,
        "selected_break_onset_dates": joint._date_signature(dates, selected_tau),
        "selected_ar_order": selected_order,
        "break_plus_trend_common_slope": float(selected_joint_time["estimate"]),
        "break_plus_trend_common_slope_ci": [
            float(selected_joint_time["ci_lower_conditional"]),
            float(selected_joint_time["ci_upper_conditional"]),
        ],
        "endpoint_intervals_are_conditional_on_selected_structure": True,
        "mann_kendall_decomposition_is_descriptive": True,
        "search_runs_per_mode_and_duration": RUNS_PER_MODE,
        "master_seed": MASTER_SEED,
        "input_sha256": {
            str(polarization_file.resolve()): sha256_file(polarization_file),
            str(
                (
                    statistics_dir
                    / "joint_ar_breakpoints"
                    / "joint_ar_bic_winners.csv"
                ).resolve()
            ): sha256_file(
                statistics_dir
                / "joint_ar_breakpoints"
                / "joint_ar_bic_winners.csv"
            ),
        },
        "output_sha256": {
            path.name: sha256_file(path) for path in paths.values()
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    metadata_path = statistics_dir / "trend_break_analysis_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "search_runs": search_runs,
        "comparison": comparison,
        "coefficients": coefficients,
        "endpoints": endpoints,
        "decomposition": decomposition,
        "fitted": fitted,
        "residuals": residuals,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--polarization-file", type=Path, default=POLARIZATION_FILE
    )
    parser.add_argument(
        "--statistics-dir", type=Path, default=STATISTICS_DIR
    )
    args = parser.parse_args(argv)
    run_trend_break_analysis(args.polarization_file, args.statistics_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
