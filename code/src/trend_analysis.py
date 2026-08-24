"""Validated Hamed-Rao Mann-Kendall and Sen-slope reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pymannkendall as mk
from scipy.stats import norm


TREND_SCHEMA = "paper3-hamed-rao-sen-trend-v1"


@dataclass(frozen=True)
class TrendResult:
    """Complete machine-readable trend result."""

    values: Dict[str, object]


def pairwise_slopes(values: np.ndarray) -> np.ndarray:
    """Return sorted pairwise slopes on the one-month time scale."""

    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or len(series) < 2 or not np.all(np.isfinite(series)):
        raise ValueError("values must be a finite one-dimensional series")
    slopes = [
        (series[j] - series[i]) / (j - i)
        for i in range(len(series) - 1)
        for j in range(i + 1, len(series))
    ]
    return np.sort(np.asarray(slopes, dtype=float))


def sen_slope_interval(
    values: np.ndarray,
    variance_s: float,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Calculate Sen's slope and its rank-order confidence interval.

    The interval follows equation 2.6 in Sen (1968), replacing the ordinary
    Mann-Kendall variance with the reported Hamed-Rao variance. When the
    correction factor equals one, the result is identical to the ordinary
    Sen rank-order interval.
    """

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not np.isfinite(variance_s) or variance_s <= 0:
        raise ValueError("variance_s must be positive and finite")
    slopes = pairwise_slopes(values)
    estimate = float(np.median(slopes))
    alpha = 1.0 - confidence_level
    z = float(norm.ppf(alpha / 2.0))
    sigma = float(np.sqrt(variance_s))
    count = len(slopes)
    upper_index = min(int(np.round((count - z * sigma) / 2.0)), count - 1)
    lower_index = max(int(np.round((count + z * sigma) / 2.0)) - 1, 0)
    return estimate, float(slopes[lower_index]), float(slopes[upper_index])


def hamed_rao_sen_result(
    values: np.ndarray,
    series_name: str,
    lag: int,
    scale: str,
    confidence_level: float = 0.95,
) -> TrendResult:
    """Run the full Hamed-Rao and Sen-slope reporting contract."""

    series = np.asarray(values, dtype=float)
    if lag < 0:
        raise ValueError("lag must be nonnegative")
    alpha = 1.0 - confidence_level
    adjusted = mk.hamed_rao_modification_test(series, alpha=alpha, lag=lag)
    ordinary = mk.original_test(series, alpha=alpha)
    slope, slope_lower, slope_upper = sen_slope_interval(
        series,
        float(adjusted.var_s),
        confidence_level,
    )
    if not np.isclose(slope, float(adjusted.slope), atol=1e-12, rtol=1e-12):
        raise RuntimeError("independent Sen slope does not match pyMannKendall")
    correction_factor = float(adjusted.var_s) / float(ordinary.var_s)
    values_out: Dict[str, object] = {
        "schema": TREND_SCHEMA,
        "test_name": "hamed_rao_mann_kendall_with_sen_slope",
        "series": series_name,
        "procedure": "pymannkendall.hamed_rao_modification_test",
        "lag": int(lag),
        "n_observations": int(len(series)),
        "S": float(adjusted.s),
        "variance_ordinary": float(ordinary.var_s),
        "variance_hamed_rao": float(adjusted.var_s),
        "hamed_rao_correction_factor": correction_factor,
        "Z": float(adjusted.z),
        "tau": float(adjusted.Tau),
        "p_value": float(adjusted.p),
        "trend": str(adjusted.trend),
        "significant_0_05": bool(adjusted.h),
        "sen_slope": slope,
        "sen_slope_ci_lower": slope_lower,
        "sen_slope_ci_upper": slope_upper,
        "sen_slope_confidence_level": confidence_level,
        "sen_slope_unit": scale,
        "sen_slope_interval_method": (
            "Sen (1968) rank-order interval using Hamed-Rao variance"
        ),
    }
    return TrendResult(values_out)

