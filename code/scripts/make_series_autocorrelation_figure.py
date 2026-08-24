#!/usr/bin/env python3
"""Pre-test serial-dependence diagnostics for the two monthly Paper 3 series.

Computes the autocorrelation and partial autocorrelation functions of monthly
General Ukraine Aid Stance polarization and the Party Brand Index, writes the
exact values to CSV, and renders the approved monochrome diagnostic figure.

This is the evidence base for two Method claims: that the corrected
Mann-Kendall procedure was applied because serial dependence was detected
rather than assumed, and that the truncation lags were set from the observed
dependence structure rather than chosen by convention.

Bands are approximate 95% references. The ACF uses cumulative Bartlett bands,
which widen with lag; the PACF uses the flat plus or minus 1.96 over the square
root of N band. Both are the conventional references reported alongside sample
correlograms and are not exact simultaneous tests.

Outputs are derivative research records. They do not alter the immutable
Factiva-to-LLM-annotation provenance chain.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from src.plot_style import (  # noqa: E402
    COLORS,
    CONSTRUCT_LABELS,
    UNCERTAINTY_STYLES,
    add_panel_label,
    apply_apa_style,
    figure_size,
    format_figure_note,
    save_figure,
    style_axes,
    validate_style_contract,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPECTED_MONTHS = pd.period_range("2022-03", "2024-12", freq="M").astype(str)
MAX_LAG = 6
CONFIDENCE_Z = 1.96

REPO_ROOT = PIPELINE_ROOT.parent
DEFAULT_POLARIZATION_FILE = (
    REPO_ROOT / "data" / "aggregated" / "monthly_polarization.csv"
)
DEFAULT_PBI_FILE = (
    REPO_ROOT / "data" / "statistics" / "party_brand_index_timeseries.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "manuscript" / "figures"

# Truncation lags carried into the Hamed-Rao correction, by series.
REPORTED_TRUNCATION_LAG = {"polarization": 2, "party_brand_index": 3}


def _validate_calendar(frame: pd.DataFrame, name: str) -> None:
    months = frame["year_month"].astype(str).to_numpy()
    if len(months) != len(EXPECTED_MONTHS) or not np.array_equal(months, EXPECTED_MONTHS.to_numpy()):
        raise ValueError(f"{name} does not contain the complete 34-month calendar")


def load_series(polarization_file: Path, pbi_file: Path) -> dict[str, np.ndarray]:
    """Return the two monthly series on the validated calendar."""

    polarization = pd.read_csv(polarization_file)
    polarization = polarization.loc[polarization["dimension"].eq("q1")].sort_values(
        "year_month", kind="stable"
    )
    _validate_calendar(polarization, "monthly polarization")

    pbi = pd.read_csv(pbi_file).sort_values("year_month", kind="stable")
    _validate_calendar(pbi, "Party Brand Index")

    series = {
        "polarization": polarization["polarization"].to_numpy(dtype=float),
        "party_brand_index": pbi["party_brand_index"].to_numpy(dtype=float),
    }
    for name, values in series.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")
    return series


def bartlett_band(correlations: np.ndarray, lag: int, n: int) -> float:
    """Cumulative Bartlett 95% band for the ACF at a given lag."""

    return CONFIDENCE_Z * float(np.sqrt((1.0 + 2.0 * np.sum(correlations[1:lag] ** 2)) / n))


def correlogram_table(series: dict[str, np.ndarray]) -> pd.DataFrame:
    """Return exact ACF and PACF values with their reference bands."""

    rows: list[dict[str, object]] = []
    for name, values in series.items():
        n = len(values)
        correlations = acf(values, nlags=MAX_LAG, fft=False)
        partials = pacf(values, nlags=MAX_LAG, method="ywm")
        flat = CONFIDENCE_Z / float(np.sqrt(n))
        for lag in range(1, MAX_LAG + 1):
            band = bartlett_band(correlations, lag, n)
            rows.append(
                {
                    "series": name,
                    "n_observations": n,
                    "lag": lag,
                    "acf": float(correlations[lag]),
                    "acf_band_bartlett_95": band,
                    "acf_outside_band": bool(abs(correlations[lag]) > band),
                    "pacf": float(partials[lag]),
                    "pacf_band_flat_95": flat,
                    "pacf_outside_band": bool(abs(partials[lag]) > flat),
                    "reported_truncation_lag": REPORTED_TRUNCATION_LAG[name],
                }
            )
    return pd.DataFrame(rows)


def _panel(ax, lags, values, bands, ink_label: str | None) -> None:
    fill = UNCERTAINTY_STYLES["measurement"]
    edge = np.concatenate([[bands[0]], bands, [bands[-1]]])
    span = np.concatenate([[lags[0] - 0.6], lags, [lags[-1] + 0.6]])
    ax.fill_between(
        span, -edge, edge,
        facecolor=fill["facecolor"], edgecolor=fill["edgecolor"],
        alpha=fill["alpha"], linewidth=fill["linewidth"], zorder=1,
    )
    ax.axhline(0, color=COLORS["zero"], linewidth=0.8, linestyle=":", zorder=2)
    ax.vlines(lags, 0, values, color=COLORS["ink"], linewidth=1.0, zorder=3)
    outside = np.abs(values) > bands
    ax.scatter(
        lags[~outside], values[~outside], s=20, zorder=4,
        facecolor=COLORS["background"], edgecolor=COLORS["ink"], linewidth=0.7,
    )
    ax.scatter(
        lags[outside], values[outside], s=26, zorder=4,
        facecolor=COLORS["ink"], edgecolor=COLORS["ink"], linewidth=0.7,
    )
    for lag, value, flagged in zip(lags, values, outside):
        if flagged:
            ax.annotate(
                f"{value:.3f}", (lag, value), textcoords="offset points",
                xytext=(6, 3), fontsize=7.2, color=COLORS["ink"],
            )
    ax.set_xlim(lags[0] - 0.6, lags[-1] + 0.6)
    ax.set_ylim(-0.62, 0.72)
    ax.set_xticks(lags)
    ax.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    style_axes(ax, show_y_grid=True)
    if ink_label:
        ax.set_ylabel(ink_label, fontsize=9)


def build_figure(table: pd.DataFrame):
    """Render the two-series correlogram grid under the approved style."""

    apply_apa_style()
    width, height = figure_size("double_column")
    fig, axes = plt.subplots(2, 2, figsize=(width, height + 0.55), sharex=True)
    fig.suptitle(
        "Autocorrelation and Partial Autocorrelation of Polarization and PBI",
        y=0.985,
        fontsize=11.5,
    )

    layout = [
        ("polarization", CONSTRUCT_LABELS["q1_polarization"]),
        ("party_brand_index", CONSTRUCT_LABELS["party_brand_index"]),
    ]
    labels = iter(["A", "B", "C", "D"])

    for row, (name, construct) in enumerate(layout):
        block = table.loc[table["series"].eq(name)].sort_values("lag")
        lags = block["lag"].to_numpy(dtype=float)
        for column, (values, bands, title) in enumerate(
            [
                (block["acf"].to_numpy(float), block["acf_band_bartlett_95"].to_numpy(float), "Autocorrelation"),
                (block["pacf"].to_numpy(float), block["pacf_band_flat_95"].to_numpy(float), "Partial autocorrelation"),
            ]
        ):
            ax = axes[row, column]
            ylabel = construct.replace(" Polarization", "\nPolarization") if column == 0 else None
            _panel(ax, lags, values, bands, ylabel)
            add_panel_label(ax, next(labels))
            if row == 0:
                ax.set_title(title)
            if row == len(layout) - 1:
                ax.set_xlabel("Lag (months)")

    note_body = (
        "Diagnostics computed on the raw monthly series before trend testing. Filled markers fall "
        "outside the approximate 95% band."
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    fig.text(
        0.02, 0.015,
        format_figure_note(textwrap.fill(note_body, width=118)),
        ha="left", va="bottom", fontsize=7.6, color=COLORS["muted_ink"], linespacing=1.5,
    )
    return fig


def run(polarization_file: Path, pbi_file: Path, output_dir: Path) -> pd.DataFrame:
    failures = validate_style_contract()
    if failures:
        raise RuntimeError("style contract violations: " + "; ".join(failures))

    output_dir.mkdir(parents=True, exist_ok=True)
    series = load_series(polarization_file, pbi_file)
    table = correlogram_table(series)

    table_path = output_dir / "series_autocorrelation.csv"
    table.to_csv(table_path, index=False)

    figure = build_figure(table)
    save_figure(figure, output_dir / "figure_1_series_autocorrelation", formats=("png", "pdf"))
    plt.close(figure)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polarization-file", type=Path, default=DEFAULT_POLARIZATION_FILE)
    parser.add_argument("--pbi-file", type=Path, default=DEFAULT_PBI_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    table = run(args.polarization_file.resolve(), args.pbi_file.resolve(), args.output_dir.resolve())
    print(table.to_string(index=False))
    print(f"\nWrote diagnostics to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
