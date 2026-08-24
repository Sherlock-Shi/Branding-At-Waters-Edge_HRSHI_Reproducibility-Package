#!/usr/bin/env python3
"""Jointly select mean shifts and a common AR order by BIC.

It fits

    y_t = mu_t + e_t,    phi(B)e_t = u_t,

where ``mu_t`` is piecewise constant and one stationary AR(p) process is
shared across every regime.  The shift count, onsets, and p in
{0, 1, 2} are optimized together with the island genetic algorithm in R
``changepointGA``.  The objective is the mean-shift BIC described by Lund et
al. (2023): the ordinary fitted-model BIC plus ``m * log(n)`` to count the
estimated onsets of the m shifts.  AIC and AICc are not used.

The analysis repeats outcome-independent structured and random initializations for minimum regime
durations of 3, 4, 5, and 6 months.  Outputs are derivative research records.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _register_windows_runtime() -> list[object]:
    handles: list[object] = []
    if os.name != "nt":
        return handles
    directories = [
        sys.prefix,
        os.path.join(sys.prefix, "Library", "mingw-w64", "bin"),
        os.path.join(sys.prefix, "Library", "usr", "bin"),
        os.path.join(sys.prefix, "Library", "bin"),
        os.path.join(sys.prefix, "Scripts"),
        os.path.join(sys.prefix, "bin"),
    ]
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            handles.append(os.add_dll_directory(directory))
    return handles


_RUNTIME_DLL_HANDLES = _register_windows_runtime()

# On Windows, make bundled R and its libraries discoverable when running under
# a Conda environment. The guarded registration is a no-op on macOS and Linux.
import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import numpy2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

CHANGEPOINT_GA = importr("changepointGA")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.stattools import acf  # noqa: E402
from src.measurement_bootstrap import derive_child_seed  # noqa: E402
from src.plot_style import (  # noqa: E402
    COLORS,
    CONSTRUCT_LABELS,
    REFERENCE_STYLES,
    SERIES_STYLES,
    add_breakpoint_lines,
    apply_apa_style,
    format_figure_note,
    format_month_axis,
    plot_structural_mean,
    style_axes,
)

apply_apa_style()


def _display_series(series: str) -> str:
    """Return the reader-facing construct name without changing data keys."""

    if series == "Polarization":
        return CONSTRUCT_LABELS["q1_polarization"]
    if series == "Party Brand Index":
        return CONSTRUCT_LABELS["party_brand_index"]
    raise KeyError(f"unknown reporting series: {series}")


REPO_ROOT = Path(__file__).resolve().parents[2]
POLARIZATION_FILE = (
    REPO_ROOT
    / "UKRAID-Politicization"
    / "Factiva_Data"
    / "pipeline_c"
    / "llm_aggregated"
    / "monthly_polarization.csv"
)
PBI_FILE = (
    REPO_ROOT
    / "UKRAID-Politicization"
    / "Factiva_Data"
    / "pipeline_c"
    / "llm_statistics"
    / "party_brand_index_timeseries.csv"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "UKRAID-Politicization"
    / "Factiva_Data"
    / "pipeline_c"
    / "llm_statistics"
    / "joint_ar_breakpoints"
)
MIN_SEGLENS = (3, 4, 5, 6)
AR_ORDERS = (0, 1, 2)
EXPECTED_MONTHS = 34
POP_SIZE = 100
NUM_ISLANDS = 5
INITIAL_CHANGE_PROBABILITY = 0.12
MAX_MIGRATIONS = 150
GENERATIONS_PER_MIGRATION = 30
CONVERGENCE_MIGRATIONS = 25
OBJECTIVE_TOLERANCE = 1.0e-5
PLOT_COLORS = {0: COLORS["q4"], 1: COLORS["democrat"], 2: COLORS["republican"]}
MASTER_SEED = 42
PLOT_JITTER_NAMESPACE = "joint_ar_breakpoint_diagnostic_plot_jitter"
PLOT_JITTER_SEED = derive_child_seed(MASTER_SEED, PLOT_JITTER_NAMESPACE)


def _random_stream_contract() -> dict[str, object]:
    """Describe analytical and diagnostic random streams without changing results."""
    return {
        "master_seed": MASTER_SEED,
        "named_child_streams": {
            "diagnostic_plot_jitter": {
                "namespace": PLOT_JITTER_NAMESPACE,
                "seed": int(PLOT_JITTER_SEED),
                "role": "deterministic display jitter only; no analytical effect",
            }
        },
        "approved_historical_exception": {
            "stream": "joint_genetic_search_seed_grid",
            "rule": (
                "structured base 3100 or random base 4100, plus minimum regime "
                "duration times 100, replicate number, and a 50-unit Party Brand "
                "Index offset"
            ),
            "reason": (
                "retain the deterministic production search grid used by the "
                "repeated search"
            ),
        },
    }


def _synchronize_plot_stream_metadata(paths: tuple[Path, ...]) -> None:
    """Record the diagnostic child stream in existing derivative metadata."""
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["random_stream_contract"] = _random_stream_contract()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

# These configurations implement the user's approved reporting decision. The
# full joint search is rerun first; these exact configurations are then refit
# under the same likelihood so the selected and disclosed alternative models
# share one machine-readable specification.
APPROVED_REPORTING_MODELS = {
    "Polarization": (
        {
            "reporting_role": "selected_primary",
            "candidate_label": "Selected three-break AR(2)",
            "ar_order": 2,
            "tau": (7, 14, 19),
            "selection_basis": "minimum joint BIC among the focused polarization candidates",
        },
        {
            "reporting_role": "required_alternative",
            "candidate_label": "Competitive four-break AR(2)",
            "ar_order": 2,
            "tau": (7, 11, 15, 19),
            "selection_basis": "required disclosure because the model remains competitive",
        },
    ),
    "Party Brand Index": (
        {
            "reporting_role": "selected_primary",
            "candidate_label": "Selected one-break AR(0)",
            "ar_order": 0,
            "tau": (19,),
            "selection_basis": "minimum joint BIC under every examined duration",
        },
        {
            "reporting_role": "required_alternative",
            "candidate_label": "No-break AR(1) comparator",
            "ar_order": 1,
            "tau": (),
            "selection_basis": "structural-change comparator",
        },
    ),
}


R_CODE = r"""
paper3_joint_ar_bic <- function(chromosome, plen=1, XMat, Xt, minseg) {
  N <- length(Xt)
  m <- as.integer(chromosome[1])
  p <- as.integer(chromosome[2])
  tau <- if (m > 0) as.integer(chromosome[3:(2 + m)]) else integer(0)
  seglen <- diff(c(1L, tau, N + 1L))
  if (length(seglen) != m + 1L || any(seglen < minseg)) return(1e12)
  DesignX <- XMat
  if (m > 0) {
    CpMat <- sapply(tau, function(cp) as.numeric(seq_len(N) >= cp))
    if (m == 1) CpMat <- matrix(CpMat, ncol=1)
    DesignX <- cbind(XMat, CpMat)
  }
  fit <- try(
    stats::arima(
      Xt, order=c(p, 0, 0), xreg=DesignX, include.mean=FALSE,
      method="ML", optim.control=list(maxit=1000)
    ),
    silent=TRUE
  )
  if (inherits(fit, "try-error") || !is.finite(stats::BIC(fit))) return(1e12)
  # stats::BIC counts the regime-level coefficients, AR coefficients, and
  # innovation variance.  Add m*log(N) to count the m selected locations.
  as.numeric(stats::BIC(fit) + m * log(N))
}

paper3_parse_suggestions <- function(text) {
  if (is.null(text) || !nzchar(text)) return(NULL)
  items <- strsplit(text, ";", fixed=TRUE)[[1]]
  lapply(items, function(item) {
    if (identical(item, "none") || !nzchar(item)) return(NULL)
    as.integer(strsplit(item, "|", fixed=TRUE)[[1]])
  })
}

paper3_run_joint_ga <- function(y, minseg, seed, suggestion_text="") {
  N <- length(y)
  XMat <- matrix(1, nrow=N, ncol=1)
  mmax <- floor(N / minseg) - 1L
  suggestions <- paper3_parse_suggestions(suggestion_text)
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
      for (p in 0:2) seeded[[length(seeded) + 1L]] <- list(tau=tau, p=p)
    }
    for (i in seq_len(min(length(seeded), ncol(initial_population)))) {
      tau <- seeded[[i]]$tau
      p <- seeded[[i]]$p
      chrom <- rep(0, nrow(initial_population))
      chrom[1] <- length(tau)
      chrom[2] <- p
      if (length(tau) > 0) chrom[3:(2 + length(tau))] <- tau
      chrom[3 + length(tau)] <- N + 1L
      initial_population[, i] <- chrom
    }
  }
  population_initializer <- function(...) initial_population
  result <- changepointGA::cptgaisl(
    ObjFunc=paper3_joint_ar_bic,
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

paper3_fit_joint_config <- function(y, p, tau) {
  N <- length(y)
  m <- length(tau)
  DesignX <- matrix(1, nrow=N, ncol=1)
  colnames(DesignX) <- "regime_1_level"
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
  beta <- tail(stats::coef(fit), ncol(DesignX))
  structural <- as.numeric(DesignX %%*%% beta)
  innovations <- as.numeric(stats::residuals(fit))
  raw_bic <- as.numeric(stats::BIC(fit))
  list(
    log_likelihood=as.numeric(stats::logLik(fit)),
    raw_bic=raw_bic,
    location_penalty=m * log(N),
    objective=raw_bic + m * log(N),
    sigma2=as.numeric(fit$sigma2),
    coefficients=as.numeric(stats::coef(fit)),
    coefficient_names=names(stats::coef(fit)),
    structural_mean=structural,
    innovations=innovations,
    one_step_fitted=as.numeric(y - innovations)
  )
}
""" % (
    POP_SIZE,
    INITIAL_CHANGE_PROBABILITY,
    POP_SIZE,
    NUM_ISLANDS,
    INITIAL_CHANGE_PROBABILITY,
    MAX_MIGRATIONS,
    GENERATIONS_PER_MIGRATION,
    CONVERGENCE_MIGRATIONS,
)

ro.r(R_CODE)
RUN_JOINT_GA = ro.globalenv["paper3_run_joint_ga"]
FIT_JOINT_CONFIG = ro.globalenv["paper3_fit_joint_config"]


def _r_version() -> str:
    return str(ro.r("paste(R.version$major, R.version$minor, sep='.')")[0])


def _r_package_version(name: str) -> str:
    return str(ro.r(f"as.character(packageVersion('{name}'))")[0])


def _series_specs(
    polarization: pd.DataFrame,
    pbi: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    polarization = (
        polarization.loc[polarization["dimension"].eq("q1")]
        .sort_values("year_month")
        .reset_index(drop=True)
    )
    pbi = pbi.sort_values("year_month").reset_index(drop=True)
    specs = {
        "Polarization": {
            "dates": pd.to_datetime(polarization["year_month"]),
            "values": polarization["polarization"].to_numpy(dtype=float),
            "ylabel": "General Ukraine Aid Stance Polarization (Points)",
        },
        "Party Brand Index": {
            "dates": pd.to_datetime(pbi["year_month"]),
            "values": pbi["party_brand_index"].to_numpy(dtype=float),
            "ylabel": "Party Brand Index (points)",
        },
    }
    expected = pd.date_range("2022-03-01", "2024-12-01", freq="MS")
    for name, spec in specs.items():
        dates = pd.DatetimeIndex(spec["dates"])
        values = np.asarray(spec["values"], dtype=float)
        if len(values) != EXPECTED_MONTHS:
            raise ValueError(f"{name} has {len(values)} months, expected 34")
        if not dates.equals(expected):
            raise ValueError(f"{name} does not contain the complete monthly calendar")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")
    return specs


def _load_series(
    polarization_file: Path = POLARIZATION_FILE,
    pbi_file: Path = PBI_FILE,
) -> dict[str, dict[str, object]]:
    return _series_specs(
        pd.read_csv(polarization_file),
        pd.read_csv(pbi_file),
    )


def _valid_tau(tau: tuple[int, ...], n: int, minseg: int) -> bool:
    return bool(
        len(set(tau)) == len(tau)
        and tuple(sorted(tau)) == tau
        and all(length >= minseg for length in np.diff((1, *tau, n + 1)))
    )


def _suggestions(series: str, minseg: int, n: int) -> tuple[str, int]:
    del series
    candidates: set[tuple[int, ...]] = {()}
    maximum = min(4, n // minseg - 1)
    for count in range(1, maximum + 1):
        raw = [
            int(round(1 + index * n / (count + 1)))
            for index in range(1, count + 1)
        ]
        tau = tuple(sorted(set(raw)))
        if len(tau) == count and _valid_tau(tau, n, minseg):
            candidates.add(tau)
    ordered = sorted(candidates, key=lambda item: (len(item), item))
    text = ";".join("none" if not item else "|".join(map(str, item)) for item in ordered)
    return text, len(ordered)


def _extract_ga_result(result: ro.vectors.ListVector) -> tuple[np.ndarray, float, float]:
    chromosome = np.asarray(result.rx2("chromosome"), dtype=float).astype(int)
    objective = float(result.rx2("objective")[0])
    counter = float(result.rx2("convergence_counter")[0])
    return chromosome, objective, counter


def _decode_chromosome(chromosome: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    m = int(chromosome[0])
    p = int(chromosome[1])
    tau = tuple(int(value) for value in chromosome[2 : 2 + m])
    return m, p, tau


def _fit_r(values: np.ndarray, p: int, tau: tuple[int, ...]) -> dict[str, object]:
    fit = FIT_JOINT_CONFIG(
        ro.FloatVector(values.tolist()), int(p), ro.IntVector(list(tau))
    )
    return {
        "log_likelihood": float(fit.rx2("log_likelihood")[0]),
        "raw_bic": float(fit.rx2("raw_bic")[0]),
        "location_penalty": float(fit.rx2("location_penalty")[0]),
        "objective": float(fit.rx2("objective")[0]),
        "sigma2": float(fit.rx2("sigma2")[0]),
        "coefficients": np.asarray(fit.rx2("coefficients"), dtype=float),
        "coefficient_names": [str(value) for value in fit.rx2("coefficient_names")],
        "structural_mean": np.asarray(fit.rx2("structural_mean"), dtype=float),
        "innovations": np.asarray(fit.rx2("innovations"), dtype=float),
        "one_step_fitted": np.asarray(fit.rx2("one_step_fitted"), dtype=float),
    }


def _date_signature(dates: pd.Series, tau: tuple[int, ...]) -> str:
    return "none" if not tau else "|".join(dates.iloc[value - 1].strftime("%Y-%m") for value in tau)


def _tau_signature(tau: tuple[int, ...]) -> str:
    return "" if not tau else "|".join(map(str, tau))


def _run_searches(
    specs: dict[str, dict[str, object]], runs_per_mode: int
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    modes = (("structured", 3100), ("random", 4100))
    total = len(specs) * len(MIN_SEGLENS) * len(modes) * runs_per_mode
    completed = 0
    for series, spec in specs.items():
        values = np.asarray(spec["values"], dtype=float)
        dates = pd.Series(spec["dates"])
        for minseg in MIN_SEGLENS:
            suggestion_text, suggestion_count = _suggestions(series, minseg, len(values))
            for mode, seed_base in modes:
                for replicate in range(1, runs_per_mode + 1):
                    seed = seed_base + minseg * 100 + replicate + (0 if series == "Polarization" else 50)
                    supplied = suggestion_text if mode == "structured" else ""
                    result = RUN_JOINT_GA(
                        ro.FloatVector(values.tolist()),
                        int(minseg),
                        int(seed),
                        supplied,
                    )
                    chromosome, ga_objective, counter = _extract_ga_result(result)
                    m, p, tau = _decode_chromosome(chromosome)
                    if p not in AR_ORDERS or not _valid_tau(tau, len(values), minseg):
                        raise RuntimeError(
                            f"Invalid GA result for {series}, minseg={minseg}: {chromosome}"
                        )
                    refit = _fit_r(values, p, tau)
                    if not np.isclose(
                        ga_objective, refit["objective"], atol=OBJECTIVE_TOLERANCE, rtol=0
                    ):
                        raise RuntimeError(
                            f"GA/refit objective mismatch: {ga_objective} vs {refit['objective']}"
                        )
                    records.append(
                        {
                            "series": series,
                            "minseglen": minseg,
                            "initialization": mode,
                            "replicate": replicate,
                            "seed": seed,
                            "suggestion_count": suggestion_count if mode == "structured" else 0,
                            "n_breakpoints": m,
                            "ar_order": p,
                            "break_onset_indices_1based": _tau_signature(tau),
                            "break_onset_dates": _date_signature(dates, tau),
                            "ga_objective_bic": ga_objective,
                            "refit_raw_bic": refit["raw_bic"],
                            "break_location_penalty": refit["location_penalty"],
                            "log_likelihood": refit["log_likelihood"],
                            "ga_convergence_counter": counter,
                        }
                    )
                    completed += 1
                    print(
                        f"[{completed}/{total}] {series} minseg={minseg} {mode} "
                        f"seed={seed}: K={m}, AR({p}), BIC={ga_objective:.6f}",
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
    return runs


def _summarize_searches(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    winners: list[pd.Series] = []
    for (series, minseg), group in runs.groupby(["series", "minseglen"], sort=False):
        best_value = float(group["ga_objective_bic"].min())
        best_rows = group.loc[
            np.isclose(group["ga_objective_bic"], best_value, atol=OBJECTIVE_TOLERANCE, rtol=0)
        ]
        signature_counts = group["configuration"].value_counts()
        winner = best_rows.sort_values(["initialization", "seed"]).iloc[0].copy()
        winner["runs_reaching_best"] = len(best_rows)
        winner["total_runs"] = len(group)
        winner["best_recovery_rate"] = len(best_rows) / len(group)
        winner["distinct_configurations"] = int(group["configuration"].nunique())
        winners.append(winner)
        for signature, count in signature_counts.items():
            current = group.loc[group["configuration"].eq(signature)].iloc[0]
            summaries.append(
                {
                    "series": series,
                    "minseglen": minseg,
                    "configuration": signature,
                    "n_breakpoints": int(current["n_breakpoints"]),
                    "ar_order": int(current["ar_order"]),
                    "break_onset_indices_1based": current["break_onset_indices_1based"],
                    "break_onset_dates": current["break_onset_dates"],
                    "run_frequency": int(count),
                    "run_proportion": float(count / len(group)),
                    "best_objective_for_configuration": float(
                        group.loc[group["configuration"].eq(signature), "ga_objective_bic"].min()
                    ),
                    "delta_bic_from_cell_best": float(
                        group.loc[group["configuration"].eq(signature), "ga_objective_bic"].min()
                        - best_value
                    ),
                    "is_cell_winner": bool(signature in set(best_rows["configuration"])),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(winners).reset_index(drop=True)


def _winner_details(
    winners: pd.DataFrame, specs: dict[str, dict[str, object]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, object]] = []
    fitted_rows: list[pd.DataFrame] = []
    regime_rows: list[dict[str, object]] = []
    for _, row in winners.iterrows():
        series = str(row["series"])
        minseg = int(row["minseglen"])
        p = int(row["ar_order"])
        tau = tuple(
            int(item)
            for item in str(row["break_onset_indices_1based"]).split("|")
            if item and item != "nan"
        )
        values = np.asarray(specs[series]["values"], dtype=float)
        dates = pd.Series(specs[series]["dates"])
        fit = _fit_r(values, p, tau)
        standardized = fit["innovations"] / np.sqrt(float(fit["sigma2"]))
        finite = standardized[np.isfinite(standardized)]
        residual_acf = acf(finite, nlags=min(8, len(finite) - 1), fft=False)
        ljung = acorr_ljungbox(finite, lags=[4, 8], model_df=p, return_df=True)
        model_id = f"{series}_minseg{minseg}_K{len(tau)}_AR{p}"
        model_rows.append(
            {
                "model_id": model_id,
                "series": series,
                "minseglen": minseg,
                "n_breakpoints": len(tau),
                "ar_order": p,
                "break_onset_indices_1based": _tau_signature(tau),
                "break_onset_dates": _date_signature(dates, tau),
                "log_likelihood": fit["log_likelihood"],
                "raw_fitted_model_bic": fit["raw_bic"],
                "break_location_penalty": fit["location_penalty"],
                "joint_selection_bic": fit["objective"],
                "innovation_variance": fit["sigma2"],
                "standardized_innovation_acf_lag1": float(residual_acf[1]),
                "standardized_innovation_acf_lag2": float(residual_acf[2]),
                "ljung_box_stat_lag4": float(ljung.loc[4, "lb_stat"]),
                "ljung_box_p_lag4": float(ljung.loc[4, "lb_pvalue"]),
                "ljung_box_stat_lag8": float(ljung.loc[8, "lb_stat"]),
                "ljung_box_p_lag8": float(ljung.loc[8, "lb_pvalue"]),
                "runs_reaching_best": int(row["runs_reaching_best"]),
                "total_runs": int(row["total_runs"]),
                "best_recovery_rate": float(row["best_recovery_rate"]),
                "coefficient_names": "|".join(fit["coefficient_names"]),
                "coefficient_values": "|".join(f"{value:.12g}" for value in fit["coefficients"]),
            }
        )
        fitted_rows.append(
            pd.DataFrame(
                {
                    "model_id": model_id,
                    "series": series,
                    "minseglen": minseg,
                    "year_month": dates.dt.strftime("%Y-%m"),
                    "observed": values,
                    "structural_regime_mean": fit["structural_mean"],
                    "one_step_ar_fitted": fit["one_step_fitted"],
                    "innovation": fit["innovations"],
                    "standardized_innovation": standardized,
                }
            )
        )
        starts = (1, *tau)
        ends = tuple(value - 1 for value in tau) + (len(values),)
        for regime, (start, end) in enumerate(zip(starts, ends), start=1):
            mask = slice(start - 1, end)
            structural_level = float(np.mean(fit["structural_mean"][mask]))
            previous = regime_rows[-1]["structural_regime_mean"] if regime > 1 else np.nan
            regime_rows.append(
                {
                    "model_id": model_id,
                    "series": series,
                    "minseglen": minseg,
                    "regime": regime,
                    "start_index_1based": start,
                    "end_index_1based": end,
                    "start_date": dates.iloc[start - 1].strftime("%Y-%m"),
                    "end_date": dates.iloc[end - 1].strftime("%Y-%m"),
                    "duration_months": end - start + 1,
                    "sample_mean": float(np.mean(values[mask])),
                    "structural_regime_mean": structural_level,
                    "adjacent_structural_shift": (
                        np.nan if regime == 1 else structural_level - float(previous)
                    ),
                }
            )
    return (
        pd.DataFrame(model_rows),
        pd.concat(fitted_rows, ignore_index=True),
        pd.DataFrame(regime_rows),
    )


def _plot_stability(
    runs: pd.DataFrame, winners: pd.DataFrame, regime: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    rng = np.random.default_rng(PLOT_JITTER_SEED)
    for row_index, series in enumerate(("Polarization", "Party Brand Index")):
        display_series = _display_series(series)
        current = runs.loc[runs["series"].eq(series)].copy()
        current["cell_best"] = current.groupby("minseglen")["ga_objective_bic"].transform("min")
        current["delta_bic"] = current["ga_objective_bic"] - current["cell_best"]
        ax = axes[row_index, 0]
        for mode, marker in (("structured", "o"), ("random", "x")):
            subset = current.loc[current["initialization"].eq(mode)]
            jitter = rng.normal(0, 0.035, len(subset))
            ax.scatter(
                subset["minseglen"] + jitter,
                subset["delta_bic"],
                marker=marker,
                s=34,
                alpha=0.8,
                label=mode,
            )
        ax.axhline(0, **REFERENCE_STYLES["zero"])
        ax.set_title(f"{display_series}: Repeated-Search Convergence")
        ax.set_xlabel("Minimum regime duration (months)")
        ax.set_ylabel("Delta BIC from best run")
        ax.set_xticks(MIN_SEGLENS)
        if row_index == 0:
            ax.legend(frameon=False)
        style_axes(ax, show_y_grid=True)

        ax = axes[row_index, 1]
        for p, color in PLOT_COLORS.items():
            subset = current.loc[current["ar_order"].eq(p)]
            jitter_x = rng.normal(0, 0.035, len(subset))
            jitter_y = rng.normal(0, 0.035, len(subset))
            ax.scatter(
                subset["minseglen"] + jitter_x,
                subset["n_breakpoints"] + jitter_y,
                color=color,
                s=34,
                alpha=0.75,
                label=f"AR({p})",
            )
        ax.set_title(f"{display_series}: Breakpoint-Count Distribution")
        ax.set_xlabel("Minimum regime duration (months)")
        ax.set_ylabel("Breakpoints returned by each run")
        ax.set_xticks(MIN_SEGLENS)
        observed_counts = sorted(current["n_breakpoints"].astype(int).unique())
        ax.set_yticks(observed_counts)
        if row_index == 0:
            ax.legend(frameon=False, ncol=3)
        style_axes(ax, show_y_grid=True)

        ax = axes[row_index, 2]
        win = winners.loc[winners["series"].eq(series)]
        reg = regime.loc[regime["series"].eq(series)]
        for _, wrow in win.iterrows():
            minseg = int(wrow["minseglen"])
            model_id = f"{series}_minseg{minseg}_K{int(wrow['n_breakpoints'])}_AR{int(wrow['ar_order'])}"
            changes = reg.loc[reg["model_id"].eq(model_id) & reg["regime"].gt(1)]
            for _, change in changes.iterrows():
                shift = float(change["adjacent_structural_shift"])
                ax.scatter(
                    pd.Timestamp(change["start_date"] + "-01"),
                    minseg,
                    s=45 + 5 * min(abs(shift), 8),
                    color=COLORS["selected"] if shift >= 0 else COLORS["alternative"],
                    marker="^" if shift >= 0 else "v",
                    edgecolor=COLORS["ink"],
                    linewidth=0.4,
                )
        ax.set_title(f"{display_series}: Winning Breakpoint Months")
        ax.set_xlabel("New-regime onset")
        ax.set_ylabel("Minimum regime duration (months)")
        ax.set_yticks(MIN_SEGLENS)
        ax.set_xlim(pd.Timestamp("2022-03-01"), pd.Timestamp("2024-12-01"))
        format_month_axis(
            ax,
            interval=2,
            rotation=45,
            start=pd.Timestamp("2022-03-01"),
            end=pd.Timestamp("2024-12-01"),
        )
        style_axes(ax, show_y_grid=True)
    fig.suptitle(
        "Joint Mean-Shift and Common AR Selection by Location-Adjusted BIC",
        fontsize=15,
        y=0.98,
    )
    direction_handles = [
        Line2D(
            [0], [0], marker="^", linestyle="none", color=COLORS["selected"],
            markeredgecolor=COLORS["ink"], label="Upward structural shift",
        ),
        Line2D(
            [0], [0], marker="v", linestyle="none", color=COLORS["alternative"],
            markeredgecolor=COLORS["ink"], label="Downward structural shift",
        ),
    ]
    fig.legend(
        handles=direction_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.text(
        0.5,
        0.005,
        format_figure_note("Marker size represents the absolute structural-shift magnitude."),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.06, right=0.99, bottom=0.12, top=0.90, wspace=0.18, hspace=0.35
    )
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_diagnostics(
    series: str,
    specs: dict[str, dict[str, object]],
    models: pd.DataFrame,
    fitted: pd.DataFrame,
    output: Path,
) -> None:
    dates = pd.Series(specs[series]["dates"])
    values = np.asarray(specs[series]["values"], dtype=float)
    fig, axes = plt.subplots(3, 4, figsize=(20, 11.5))
    y_min, y_max = float(values.min()), float(values.max())
    pad = max((y_max - y_min) * 0.08, 0.5)
    for column, minseg in enumerate(MIN_SEGLENS):
        model = models.loc[
            models["series"].eq(series) & models["minseglen"].eq(minseg)
        ].iloc[0]
        data = fitted.loc[fitted["model_id"].eq(model["model_id"])]
        tau = tuple(
            int(item)
            for item in str(model["break_onset_indices_1based"]).split("|")
            if item and item != "nan"
        )
        ax = axes[0, column]
        observed_style = SERIES_STYLES["observed"]
        ax.plot(
            dates,
            values,
            color=observed_style["color"],
            linestyle=observed_style["linestyle"],
            marker=observed_style["marker"],
            markerfacecolor=observed_style["markerfacecolor"],
            markeredgecolor=observed_style["markeredgecolor"],
            markersize=3.2,
            linewidth=1,
            label="Observed",
        )
        plot_structural_mean(
            ax,
            dates,
            data["structural_regime_mean"],
            role="selected",
            label="Structural Regime Mean",
            annotate=True,
        )
        ax.plot(
            dates,
            data["one_step_ar_fitted"],
            color=SERIES_STYLES["comparison"]["color"],
            linewidth=SERIES_STYLES["comparison"]["linewidth"],
            linestyle=SERIES_STYLES["comparison"]["linestyle"],
            label="One-step AR fit",
        )
        add_breakpoint_lines(
            ax,
            [dates.iloc[cp - 1] for cp in tau],
            label="_nolegend_",
            show_arrows=False,
        )
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_title(
            f"Minimum Duration: {minseg} Months | {int(model['n_breakpoints'])} Breakpoints | "
            f"AR({int(model['ar_order'])}) | BIC = {float(model['joint_selection_bic']):.2f}"
        )
        format_month_axis(
            ax,
            interval=4,
            rotation=45,
            start=dates.iloc[0],
            end=dates.iloc[-1],
        )
        ax.tick_params(axis="x", labelsize=8)
        if column == 0:
            ax.set_ylabel(str(specs[series]["ylabel"]))
        style_axes(ax, show_y_grid=True)

        ax = axes[1, column]
        standardized = data["standardized_innovation"].to_numpy(dtype=float)
        ax.axhline(0, **REFERENCE_STYLES["zero"])
        ax.axhline(2, color=COLORS["neutral"], linewidth=0.8, linestyle=":")
        ax.axhline(-2, color=COLORS["neutral"], linewidth=0.8, linestyle=":")
        ax.plot(
            dates,
            standardized,
            color=SERIES_STYLES["observed"]["color"],
            marker=SERIES_STYLES["observed"]["marker"],
            markerfacecolor=SERIES_STYLES["observed"]["markerfacecolor"],
            markeredgecolor=SERIES_STYLES["observed"]["markeredgecolor"],
            markersize=3,
            linewidth=1,
        )
        ax.set_title(
            f"Standardized Innovations, Ljung-Box Lag 8 p = {float(model['ljung_box_p_lag8']):.3f}"
        )
        ax.tick_params(axis="x", labelbottom=False)
        if column == 0:
            ax.set_ylabel("Standardized innovation")
        style_axes(ax, show_y_grid=True)

        ax = axes[2, column]
        residual_acf = acf(standardized[np.isfinite(standardized)], nlags=8, fft=False)
        lags = np.arange(len(residual_acf))
        ax.axhline(0, **REFERENCE_STYLES["zero"])
        bound = 1.96 / np.sqrt(len(standardized))
        ax.axhline(bound, **REFERENCE_STYLES["diagnostic_threshold"])
        ax.axhline(-bound, **REFERENCE_STYLES["diagnostic_threshold"])
        ax.vlines(lags[1:], 0, residual_acf[1:], color=COLORS["ink"], linewidth=1.0)
        ax.scatter(lags[1:], residual_acf[1:], color=COLORS["ink"], s=20)
        ax.set_xlim(0.5, 8.5)
        ax.set_ylim(-1, 1)
        ax.set_xticks(range(1, 9))
        ax.set_title("Innovation ACF with Approximate 95% Bounds")
        ax.set_xlabel("Lag (months)")
        if column == 0:
            ax.set_ylabel("ACF")
        style_axes(ax, show_y_grid=True)
    fig.suptitle(
        f"{_display_series(series)}: Joint Breakpoint and Common AR Visual Diagnostics",
        fontsize=15,
        y=0.985,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.96))
    fig.subplots_adjust(
        left=0.055, right=0.99, bottom=0.06, top=0.89, wspace=0.14, hspace=0.50
    )
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _focused_candidates(
    specs: dict[str, dict[str, object]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = {
        "Polarization": [
            ("Joint K=3 winner", (7, 14, 19)),
            ("Prior K=4 candidate", (7, 11, 15, 19)),
            ("Prior K=7 candidate", (4, 7, 11, 15, 19, 28, 31)),
        ],
        "Party Brand Index": [
            ("Joint K=1 winner", (19,)),
            ("No-break comparator", ()),
        ],
    }
    model_rows: list[dict[str, object]] = []
    fitted_rows: list[pd.DataFrame] = []
    for series, candidates in definitions.items():
        values = np.asarray(specs[series]["values"], dtype=float)
        dates = pd.Series(specs[series]["dates"])
        for order, (label, tau) in enumerate(candidates, start=1):
            fits = [(p, _fit_r(values, p, tau)) for p in AR_ORDERS]
            p, fit = min(fits, key=lambda item: float(item[1]["objective"]))
            standardized = fit["innovations"] / np.sqrt(float(fit["sigma2"]))
            finite = standardized[np.isfinite(standardized)]
            ljung = acorr_ljungbox(finite, lags=[4, 8], model_df=p, return_df=True)
            model_id = f"{series}_focused_{order}_K{len(tau)}_AR{p}"
            model_rows.append(
                {
                    "model_id": model_id,
                    "series": series,
                    "display_order": order,
                    "candidate_label": label,
                    "n_breakpoints": len(tau),
                    "ar_order": p,
                    "break_onset_indices_1based": _tau_signature(tau),
                    "break_onset_dates": _date_signature(dates, tau),
                    "raw_fitted_model_bic": fit["raw_bic"],
                    "break_location_penalty": fit["location_penalty"],
                    "joint_selection_bic": fit["objective"],
                    "ljung_box_p_lag4": float(ljung.loc[4, "lb_pvalue"]),
                    "ljung_box_p_lag8": float(ljung.loc[8, "lb_pvalue"]),
                    "comparison_role": "visual and diagnostic comparison only; candidate dates are not selected by visual inspection",
                }
            )
            fitted_rows.append(
                pd.DataFrame(
                    {
                        "model_id": model_id,
                        "series": series,
                        "year_month": dates.dt.strftime("%Y-%m"),
                        "observed": values,
                        "structural_regime_mean": fit["structural_mean"],
                        "one_step_ar_fitted": fit["one_step_fitted"],
                        "standardized_innovation": standardized,
                    }
                )
            )
    return pd.DataFrame(model_rows), pd.concat(fitted_rows, ignore_index=True)


def _reporting_contract(
    specs: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit the approved primary and required alternative configurations."""
    model_rows: list[dict[str, object]] = []
    fitted_rows: list[pd.DataFrame] = []
    regime_rows: list[dict[str, object]] = []

    for series, definitions in APPROVED_REPORTING_MODELS.items():
        values = np.asarray(specs[series]["values"], dtype=float)
        dates = pd.Series(specs[series]["dates"])
        series_models: list[dict[str, object]] = []

        for definition in definitions:
            tau = tuple(int(value) for value in definition["tau"])
            p = int(definition["ar_order"])
            fit = _fit_r(values, p, tau)
            standardized = fit["innovations"] / np.sqrt(float(fit["sigma2"]))
            finite = standardized[np.isfinite(standardized)]
            residual_acf = acf(finite, nlags=min(8, len(finite) - 1), fft=False)
            ljung = acorr_ljungbox(
                finite,
                lags=[4, 8],
                model_df=p,
                return_df=True,
            )
            role = str(definition["reporting_role"])
            model_id = f"{series}_{role}_K{len(tau)}_AR{p}"
            row = {
                "model_id": model_id,
                "series": series,
                "reporting_role": role,
                "selected_for_primary_reporting": role == "selected_primary",
                "candidate_label": definition["candidate_label"],
                "selection_basis": definition["selection_basis"],
                "selection_timing": "approved after focused candidate comparison; not preregistered",
                "n_observations": len(values),
                "n_breakpoints": len(tau),
                "ar_order": p,
                "break_onset_indices_1based": _tau_signature(tau),
                "break_onset_dates": _date_signature(dates, tau),
                "log_likelihood": fit["log_likelihood"],
                "raw_fitted_model_bic": fit["raw_bic"],
                "break_location_penalty": fit["location_penalty"],
                "joint_selection_bic": fit["objective"],
                "innovation_variance": fit["sigma2"],
                "standardized_innovation_acf_lag1": float(residual_acf[1]),
                "standardized_innovation_acf_lag2": float(residual_acf[2]),
                "ljung_box_stat_lag4": float(ljung.loc[4, "lb_stat"]),
                "ljung_box_p_lag4": float(ljung.loc[4, "lb_pvalue"]),
                "ljung_box_stat_lag8": float(ljung.loc[8, "lb_stat"]),
                "ljung_box_p_lag8": float(ljung.loc[8, "lb_pvalue"]),
                "coefficient_names": "|".join(fit["coefficient_names"]),
                "coefficient_values": "|".join(
                    f"{value:.12g}" for value in fit["coefficients"]
                ),
                "validation_status": "pending integrated-pipeline validation",
            }
            series_models.append(row)

            fitted_rows.append(
                pd.DataFrame(
                    {
                        "model_id": model_id,
                        "series": series,
                        "reporting_role": role,
                        "year_month": dates.dt.strftime("%Y-%m"),
                        "observed": values,
                        "structural_regime_mean": fit["structural_mean"],
                        "one_step_ar_fitted": fit["one_step_fitted"],
                        "innovation": fit["innovations"],
                        "standardized_innovation": standardized,
                    }
                )
            )

            starts = (1, *tau)
            ends = tuple(value - 1 for value in tau) + (len(values),)
            previous_level = np.nan
            for regime, (start, end) in enumerate(zip(starts, ends), start=1):
                mask = slice(start - 1, end)
                structural_level = float(np.mean(fit["structural_mean"][mask]))
                regime_rows.append(
                    {
                        "model_id": model_id,
                        "series": series,
                        "reporting_role": role,
                        "regime": regime,
                        "start_index_1based": start,
                        "end_index_1based": end,
                        "start_date": dates.iloc[start - 1].strftime("%Y-%m"),
                        "end_date": dates.iloc[end - 1].strftime("%Y-%m"),
                        "duration_months": end - start + 1,
                        "sample_mean": float(np.mean(values[mask])),
                        "structural_regime_mean": structural_level,
                        "adjacent_structural_shift": (
                            np.nan
                            if regime == 1
                            else structural_level - float(previous_level)
                        ),
                    }
                )
                previous_level = structural_level

        baseline = min(float(row["joint_selection_bic"]) for row in series_models)
        for row in series_models:
            row["delta_bic_within_reported_pair"] = (
                float(row["joint_selection_bic"]) - baseline
            )
            row["minimum_bic_within_reported_pair"] = np.isclose(
                float(row["joint_selection_bic"]),
                baseline,
                atol=OBJECTIVE_TOLERANCE,
                rtol=0,
            )
            model_rows.append(row)

    return (
        pd.DataFrame(model_rows),
        pd.concat(fitted_rows, ignore_index=True),
        pd.DataFrame(regime_rows),
    )


def _plot_focused(
    series: str,
    specs: dict[str, dict[str, object]],
    models: pd.DataFrame,
    fitted: pd.DataFrame,
    output: Path,
) -> None:
    dates = pd.Series(specs[series]["dates"])
    values = np.asarray(specs[series]["values"], dtype=float)
    selected = models.loc[models["series"].eq(series)].sort_values("display_order")
    baseline = float(selected["joint_selection_bic"].min())
    n_columns = len(selected)
    fig, axes = plt.subplots(
        3,
        n_columns,
        figsize=(5.7 * n_columns, 11.5),
        squeeze=False,
    )
    y_min, y_max = float(values.min()), float(values.max())
    pad = max((y_max - y_min) * 0.08, 0.5)
    for column, (_, model) in enumerate(selected.iterrows()):
        data = fitted.loc[fitted["model_id"].eq(model["model_id"])]
        tau = tuple(int(item) for item in str(model["break_onset_indices_1based"]).split("|") if item)
        ax = axes[0, column]
        observed_style = SERIES_STYLES["observed"]
        ax.plot(
            dates,
            values,
            color=observed_style["color"],
            linestyle=observed_style["linestyle"],
            marker=observed_style["marker"],
            markerfacecolor=observed_style["markerfacecolor"],
            markeredgecolor=observed_style["markeredgecolor"],
            markersize=3.2,
            linewidth=1,
            label="Observed",
        )
        plot_structural_mean(
            ax,
            dates,
            data["structural_regime_mean"],
            role="selected",
            label="Structural Regime Mean",
            annotate=True,
        )
        ax.plot(
            dates,
            data["one_step_ar_fitted"],
            color=SERIES_STYLES["comparison"]["color"],
            linewidth=SERIES_STYLES["comparison"]["linewidth"],
            linestyle=SERIES_STYLES["comparison"]["linestyle"],
            label="One-step AR fit",
        )
        add_breakpoint_lines(
            ax,
            [dates.iloc[cp - 1] for cp in tau],
            label="_nolegend_",
            show_arrows=False,
        )
        ax.set_ylim(y_min - pad, y_max + pad)
        delta = float(model["joint_selection_bic"]) - baseline
        ax.set_title(
            f"{model['candidate_label']}\n{int(model['n_breakpoints'])} Breakpoints, AR({int(model['ar_order'])}), "
            f"BIC = {float(model['joint_selection_bic']):.2f}, Delta BIC = {delta:.2f}",
            fontsize=10,
        )
        format_month_axis(
            ax,
            interval=4,
            rotation=45,
            start=dates.iloc[0],
            end=dates.iloc[-1],
        )
        ax.tick_params(axis="x", labelsize=8)
        if column == 0:
            ax.set_ylabel(str(specs[series]["ylabel"]))
        style_axes(ax, show_y_grid=True)

        standardized = data["standardized_innovation"].to_numpy(dtype=float)
        ax = axes[1, column]
        ax.axhline(0, **REFERENCE_STYLES["zero"])
        ax.axhline(2, color=COLORS["neutral"], linewidth=0.8, linestyle=":")
        ax.axhline(-2, color=COLORS["neutral"], linewidth=0.8, linestyle=":")
        ax.plot(
            dates,
            standardized,
            color=SERIES_STYLES["observed"]["color"],
            marker=SERIES_STYLES["observed"]["marker"],
            markerfacecolor=SERIES_STYLES["observed"]["markerfacecolor"],
            markeredgecolor=SERIES_STYLES["observed"]["markeredgecolor"],
            markersize=3,
            linewidth=1,
        )
        ax.set_title(
            f"Standardized Innovations, Ljung-Box Lag 8 p = {float(model['ljung_box_p_lag8']):.3f}"
        )
        ax.tick_params(axis="x", labelbottom=False)
        if column == 0:
            ax.set_ylabel("Standardized innovation")
        style_axes(ax, show_y_grid=True)

        ax = axes[2, column]
        residual_acf = acf(standardized[np.isfinite(standardized)], nlags=8, fft=False)
        lags = np.arange(len(residual_acf))
        bound = 1.96 / np.sqrt(len(standardized))
        ax.axhline(0, **REFERENCE_STYLES["zero"])
        ax.axhline(bound, **REFERENCE_STYLES["diagnostic_threshold"])
        ax.axhline(-bound, **REFERENCE_STYLES["diagnostic_threshold"])
        ax.vlines(lags[1:], 0, residual_acf[1:], color=COLORS["ink"], linewidth=1.0)
        ax.scatter(lags[1:], residual_acf[1:], color=COLORS["ink"], s=20)
        ax.set_xlim(0.5, 8.5)
        ax.set_ylim(-1, 1)
        ax.set_xticks(range(1, 9))
        ax.set_title("Innovation ACF with Approximate 95% Bounds")
        ax.set_xlabel("Lag (months)")
        if column == 0:
            ax.set_ylabel("ACF")
        style_axes(ax, show_y_grid=True)
    fig.suptitle(
        f"{_display_series(series)}: Focused Common AR Candidate Comparison",
        fontsize=15,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.06, right=0.99, bottom=0.06, top=0.90, wspace=0.10, hspace=0.52
    )
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _metadata(runs_per_mode: int, execution_context: str) -> dict[str, object]:
    return {
        "analysis_status": "integrated output requires independent validator",
        "analysis_date": pd.Timestamp.now(tz="Europe/London").isoformat(),
        "execution_context": execution_context,
        "model": "piecewise constant structural mean with one common stationary AR(p) error process",
        "ar_order_search": list(AR_ORDERS),
        "minimum_regime_durations": list(MIN_SEGLENS),
        "criterion": "BIC only; stats::BIC(fit) + m*log(N) to count selected breakpoint locations",
        "criterion_formula": "-2 log L + (2m + p + 2) log N",
        "aic_used": False,
        "aicc_used": False,
        "search_algorithm": "changepointGA cptgaisl island genetic algorithm",
        "runs_per_initialization_mode": runs_per_mode,
        "initialization_modes": [
            "outcome-independent evenly spaced structured starts",
            "fully random",
        ],
        "ga_parameters": {
            "population_size": POP_SIZE,
            "number_of_islands": NUM_ISLANDS,
            "initial_change_probability": INITIAL_CHANGE_PROBABILITY,
            "maximum_migrations": MAX_MIGRATIONS,
            "generations_per_migration": GENERATIONS_PER_MIGRATION,
            "convergence_migrations": CONVERGENCE_MIGRATIONS,
        },
        "random_stream_contract": _random_stream_contract(),
        "n_months": EXPECTED_MONTHS,
        "period": "2022-03 through 2024-12",
        "r_version": _r_version(),
        "r_changepointGA_version": _r_package_version("changepointGA"),
        "rpy2_version": importlib.metadata.version("rpy2"),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "statsmodels_version": statsmodels.__version__,
        "approved_reporting_models": APPROVED_REPORTING_MODELS,
        "primary_sources": [
            {
                "citation": "Lund, R., Beaulieu, C., Killick, R., Lu, Q., & Shi, X. (2023). Good Practices and Common Pitfalls in Climate Time Series Changepoint Techniques: A Review. Journal of Climate, 36(23), 8041-8057.",
                "doi": "10.1175/JCLI-D-22-0954.1",
                "role": "common-AR mean-shift estimand, joint penalized likelihood, and BIC location penalty",
            },
            {
                "citation": "Li, M., & Lu, Q. (2024). changepointGA: An R package for Fast Changepoint Detection via Genetic Algorithms.",
                "url": "https://arxiv.org/abs/2410.15571",
                "role": "joint stochastic optimization of changepoints and AR order",
            },
            {
                "citation": "Beaulieu, C., & Killick, R. (2018). Distinguishing Trends and Shifts from Memory in Climate Data. Journal of Climate, 31(23), 9519-9543.",
                "doi": "10.1175/JCLI-D-17-0863.1",
                "role": "need to distinguish persistent structural shifts from autoregressive memory",
            },
        ],
        "important_limitation": "The genetic algorithm is stochastic; repeated outcome-independent structured and random starts quantify search convergence.",
    }


def run_joint_ar_breakpoint_analysis(
    monthly_polarization: pd.DataFrame,
    party_brand_index: pd.DataFrame,
    statistics_output_dir: Path | str,
    runs_per_mode: int = 5,
    generate_plots: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the joint search and write the production output specification."""
    if runs_per_mode < 1:
        raise ValueError("runs_per_mode must be positive")

    specs = _series_specs(monthly_polarization, party_brand_index)
    statistics_dir = Path(statistics_output_dir)
    output_dir = statistics_dir / "joint_ar_breakpoints"
    statistics_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _run_searches(specs, runs_per_mode)
    configurations, winners = _summarize_searches(runs)
    models, fitted, regimes = _winner_details(winners, specs)
    focused_models, focused_fitted = _focused_candidates(specs)
    reporting_models, reporting_fitted, reporting_regimes = _reporting_contract(specs)

    runs.to_csv(output_dir / "joint_ar_bic_search_runs.csv", index=False)
    configurations.to_csv(output_dir / "joint_ar_bic_search_convergence.csv", index=False)
    winners.to_csv(output_dir / "joint_ar_bic_winners.csv", index=False)
    models.to_csv(output_dir / "joint_ar_bic_model_diagnostics.csv", index=False)
    fitted.to_csv(output_dir / "joint_ar_bic_fitted_series.csv", index=False)
    regimes.to_csv(output_dir / "joint_ar_bic_regime_statistics.csv", index=False)
    focused_models.to_csv(output_dir / "joint_ar_bic_focused_candidates.csv", index=False)
    focused_fitted.to_csv(output_dir / "joint_ar_bic_focused_fitted_series.csv", index=False)

    reporting_models.to_csv(statistics_dir / "breakpoint_models.csv", index=False)
    reporting_fitted.to_csv(statistics_dir / "breakpoint_fitted_series.csv", index=False)
    reporting_regimes.to_csv(statistics_dir / "breakpoint_regimes.csv", index=False)

    if generate_plots:
        _plot_stability(runs, winners, regimes, output_dir / "joint_ar_bic_stability.png")
        for series, filename in (
            ("Polarization", "joint_ar_bic_polarization_diagnostics.png"),
            ("Party Brand Index", "joint_ar_bic_party_brand_index_diagnostics.png"),
        ):
            _plot_diagnostics(series, specs, models, fitted, output_dir / filename)
        for series, filename in (
            ("Polarization", "joint_ar_bic_polarization_focused.png"),
            ("Party Brand Index", "joint_ar_bic_party_brand_index_focused.png"),
        ):
            _plot_focused(
                series,
                specs,
                focused_models,
                focused_fitted,
                output_dir / filename,
            )

    metadata = _metadata(runs_per_mode, "centralized Pipeline C statistics stage")
    (output_dir / "joint_ar_bic_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (statistics_dir / "breakpoint_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "search_runs": runs,
        "search_convergence": configurations,
        "search_winners": winners,
        "model_diagnostics": models,
        "fitted_series": fitted,
        "regime_statistics": regimes,
        "reporting_models": reporting_models,
        "reporting_fitted_series": reporting_fitted,
        "reporting_regimes": reporting_regimes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-per-mode",
        type=int,
        default=5,
        help="Independent structured and random GA runs per series/minimum duration",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate figures and focused candidate tables from existing CSV outputs",
    )
    args = parser.parse_args()
    if args.runs_per_mode < 1:
        raise ValueError("--runs-per-mode must be positive")

    specs = _load_series()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        runs = pd.read_csv(OUTPUT_DIR / "joint_ar_bic_search_runs.csv")
        winners = pd.read_csv(OUTPUT_DIR / "joint_ar_bic_winners.csv")
        models = pd.read_csv(OUTPUT_DIR / "joint_ar_bic_model_diagnostics.csv")
        fitted = pd.read_csv(OUTPUT_DIR / "joint_ar_bic_fitted_series.csv")
        regimes = pd.read_csv(OUTPUT_DIR / "joint_ar_bic_regime_statistics.csv")
        focused_models, focused_fitted = _focused_candidates(specs)
        focused_models.to_csv(OUTPUT_DIR / "joint_ar_bic_focused_candidates.csv", index=False)
        focused_fitted.to_csv(OUTPUT_DIR / "joint_ar_bic_focused_fitted_series.csv", index=False)
        _plot_stability(runs, winners, regimes, OUTPUT_DIR / "joint_ar_bic_stability.png")
        for series, filename in (
            ("Polarization", "joint_ar_bic_polarization_diagnostics.png"),
            ("Party Brand Index", "joint_ar_bic_party_brand_index_diagnostics.png"),
        ):
            _plot_diagnostics(series, specs, models, fitted, OUTPUT_DIR / filename)
        for series, filename in (
            ("Polarization", "joint_ar_bic_polarization_focused.png"),
            ("Party Brand Index", "joint_ar_bic_party_brand_index_focused.png"),
        ):
            _plot_focused(series, specs, focused_models, focused_fitted, OUTPUT_DIR / filename)
        _synchronize_plot_stream_metadata(
            (
                OUTPUT_DIR / "joint_ar_bic_metadata.json",
                OUTPUT_DIR.parent / "breakpoint_analysis_metadata.json",
            )
        )
        print(f"Regenerated plots in {OUTPUT_DIR}", flush=True)
        return

    runs = _run_searches(specs, args.runs_per_mode)
    configurations, winners = _summarize_searches(runs)
    models, fitted, regimes = _winner_details(winners, specs)

    runs.to_csv(OUTPUT_DIR / "joint_ar_bic_search_runs.csv", index=False)
    configurations.to_csv(OUTPUT_DIR / "joint_ar_bic_search_convergence.csv", index=False)
    winners.to_csv(OUTPUT_DIR / "joint_ar_bic_winners.csv", index=False)
    models.to_csv(OUTPUT_DIR / "joint_ar_bic_model_diagnostics.csv", index=False)
    fitted.to_csv(OUTPUT_DIR / "joint_ar_bic_fitted_series.csv", index=False)
    regimes.to_csv(OUTPUT_DIR / "joint_ar_bic_regime_statistics.csv", index=False)
    focused_models, focused_fitted = _focused_candidates(specs)
    focused_models.to_csv(OUTPUT_DIR / "joint_ar_bic_focused_candidates.csv", index=False)
    focused_fitted.to_csv(OUTPUT_DIR / "joint_ar_bic_focused_fitted_series.csv", index=False)

    _plot_stability(
        runs,
        winners,
        regimes,
        OUTPUT_DIR / "joint_ar_bic_stability.png",
    )
    for series, filename in (
        ("Polarization", "joint_ar_bic_polarization_diagnostics.png"),
        ("Party Brand Index", "joint_ar_bic_party_brand_index_diagnostics.png"),
    ):
        _plot_diagnostics(series, specs, models, fitted, OUTPUT_DIR / filename)
    for series, filename in (
        ("Polarization", "joint_ar_bic_polarization_focused.png"),
        ("Party Brand Index", "joint_ar_bic_party_brand_index_focused.png"),
    ):
        _plot_focused(series, specs, focused_models, focused_fitted, OUTPUT_DIR / filename)

    metadata = {
        "analysis_status": "provisional until validator and user review complete",
        "analysis_date": pd.Timestamp.now(tz="Europe/London").isoformat(),
        "model": "piecewise constant structural mean with one common stationary AR(p) error process",
        "ar_order_search": list(AR_ORDERS),
        "minimum_regime_durations": list(MIN_SEGLENS),
        "criterion": "BIC only; stats::BIC(fit) + m*log(N) to count selected breakpoint locations",
        "criterion_formula": "-2 log L + (2m + p + 2) log N",
        "aic_used": False,
        "aicc_used": False,
        "search_algorithm": "changepointGA cptgaisl island genetic algorithm",
        "runs_per_initialization_mode": args.runs_per_mode,
        "initialization_modes": [
            "outcome-independent evenly spaced structured starts",
            "fully random starts",
        ],
        "ga_parameters": {
            "population_size": POP_SIZE,
            "number_of_islands": NUM_ISLANDS,
            "initial_change_probability": INITIAL_CHANGE_PROBABILITY,
            "maximum_migrations": MAX_MIGRATIONS,
            "generations_per_migration": GENERATIONS_PER_MIGRATION,
            "convergence_migrations": CONVERGENCE_MIGRATIONS,
        },
        "random_stream_contract": _random_stream_contract(),
        "n_months": EXPECTED_MONTHS,
        "period": "2022-03 through 2024-12",
        "r_version": _r_version(),
        "r_changepointGA_version": _r_package_version("changepointGA"),
        "rpy2_version": importlib.metadata.version("rpy2"),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "statsmodels_version": statsmodels.__version__,
        "primary_sources": [
            {
                "citation": "Lund, R., Beaulieu, C., Killick, R., Lu, Q., & Shi, X. (2023). Good Practices and Common Pitfalls in Climate Time Series Changepoint Techniques: A Review. Journal of Climate, 36(23), 8041-8057.",
                "doi": "10.1175/JCLI-D-22-0954.1",
                "role": "common-AR mean-shift estimand, joint penalized likelihood, and BIC location penalty",
            },
            {
                "citation": "Li, M., & Lu, Q. (2024). changepointGA: An R package for Fast Changepoint Detection via Genetic Algorithms.",
                "url": "https://arxiv.org/abs/2410.15571",
                "role": "joint stochastic optimization of changepoints and AR order",
            },
            {
                "citation": "Beaulieu, C., & Killick, R. (2018). Distinguishing Trends and Shifts from Memory in Climate Data. Journal of Climate, 31(23), 9519-9543.",
                "doi": "10.1175/JCLI-D-17-0863.1",
                "role": "need to distinguish persistent structural shifts from autoregressive memory",
            },
        ],
        "important_limitation": "The genetic algorithm is stochastic; repeated outcome-independent structured and random starts quantify search convergence.",
    }
    (OUTPUT_DIR / "joint_ar_bic_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote joint analysis to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
