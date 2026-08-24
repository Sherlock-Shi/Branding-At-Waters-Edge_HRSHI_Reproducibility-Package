"""Publication and supplementary figures for validated Paper 3 results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from src.plot_style import (
    BAR_STYLES,
    COLORS,
    CONSTRUCT_LABELS,
    DIMENSION_STYLES,
    REFERENCE_STYLES,
    SERIES_STYLES,
    STYLE_VERSION,
    UNCERTAINTY_STYLES,
    add_breakpoint_lines,
    add_panel_label,
    apa_style_context,
    figure_size,
    format_figure_note,
    format_month_axis,
    plot_structural_mean,
    save_figure,
    style_axes,
)
from src.publication_contract import PublicationInputs


FIGURE_SCHEMA = "paper3-publication-figures-v1"
EXPECTED_MONTHS = pd.period_range("2022-03", "2024-12", freq="M").astype(str).tolist()
STUDY_START = pd.Timestamp(f"{EXPECTED_MONTHS[0]}-01")
STUDY_END = pd.Timestamp(f"{EXPECTED_MONTHS[-1]}-01")


def _dates(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_values("year_month").reset_index(drop=True)
    months = ordered["year_month"].astype(str).tolist()
    if months != EXPECTED_MONTHS:
        raise ValueError("publication figure input does not contain the exact 34-month calendar")
    return pd.to_datetime(ordered["year_month"])


def _model_rows(
    models: pd.DataFrame,
    fitted: pd.DataFrame,
    series: str,
    role: str,
) -> tuple[pd.Series, pd.DataFrame]:
    selected = models.loc[models["series"].eq(series) & models["reporting_role"].eq(role)]
    if len(selected) != 1:
        raise ValueError(f"expected one {series} {role} model")
    model = selected.iloc[0]
    values = fitted.loc[fitted["model_id"].eq(model["model_id"])].sort_values("year_month")
    if len(values) != 34:
        raise ValueError(f"{series} {role} model does not contain 34 fitted rows")
    return model, values.reset_index(drop=True)


def _fill_measurement_interval(
    ax: Axes,
    dates: pd.Series,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    role: str = "measurement",
    label: str | None = None,
    zorder: float = 1,
) -> None:
    """Draw a pointwise article-cluster interval through the shared gray role."""

    style = UNCERTAINTY_STYLES[role]
    ax.fill_between(
        dates,
        lower,
        upper,
        facecolor=style["facecolor"],
        edgecolor=style["edgecolor"],
        alpha=style["alpha"],
        linewidth=style["linewidth"],
        label=label,
        zorder=zorder,
    )


def _dimension_style(series: str) -> dict:
    """Map reporting-series names to reusable non-color construct cues."""

    if series == "Polarization":
        return DIMENSION_STYLES["q1"]
    if series == "Party Brand Index":
        return DIMENSION_STYLES["party_brand_index"]
    raise KeyError(f"no publication dimension style for {series}")


def _temporal_panel(
    ax: Axes,
    *,
    points: pd.DataFrame,
    value: str,
    lower: str,
    upper: str,
    series: str,
    ylabel: str,
    models: pd.DataFrame,
    fitted: pd.DataFrame,
) -> None:
    points = points.sort_values("year_month").reset_index(drop=True)
    dates = _dates(points)
    selected, selected_values = _model_rows(models, fitted, series, "selected_primary")
    alternative, alternative_values = _model_rows(models, fitted, series, "required_alternative")
    if selected_values["year_month"].astype(str).tolist() != EXPECTED_MONTHS:
        raise ValueError(f"{series} selected fitted calendar differs from the measurement calendar")
    if alternative_values["year_month"].astype(str).tolist() != EXPECTED_MONTHS:
        raise ValueError(f"{series} alternative fitted calendar differs from the measurement calendar")

    _fill_measurement_interval(
        ax,
        dates,
        points[lower].to_numpy(float),
        points[upper].to_numpy(float),
        label="95% measurement interval",
        zorder=1,
    )
    ax.plot(
        dates,
        points[value].to_numpy(float),
        color=SERIES_STYLES["observed"]["color"],
        linestyle=SERIES_STYLES["observed"]["linestyle"],
        marker=SERIES_STYLES["observed"]["marker"],
        markerfacecolor=SERIES_STYLES["observed"]["markerfacecolor"],
        markeredgecolor=SERIES_STYLES["observed"]["markeredgecolor"],
        linewidth=1.0,
        markersize=3.2,
        label="Observed value",
        zorder=3,
    )
    plot_structural_mean(
        ax,
        dates,
        selected_values["structural_regime_mean"].to_numpy(float),
        role="selected",
        label="Selected Structural Mean",
        annotate=True,
        zorder=4,
    )
    plot_structural_mean(
        ax,
        dates,
        alternative_values["structural_regime_mean"].to_numpy(float),
        role="alternative",
        label="Alternative Structural Mean",
        annotate=False,
        zorder=2,
    )
    break_dates = [
        item
        for item in str(selected["break_onset_dates"]).split("|")
        if item and item not in {"none", "nan"}
    ]
    add_breakpoint_lines(
        ax,
        [pd.Timestamp(date + "-01") for date in break_dates],
        show_arrows=True,
    )
    ax.text(
        0.01,
        0.98,
        (
            f"Selected: AR({int(selected['ar_order'])}), BIC = {selected['joint_selection_bic']:.3f}\n"
            f"Alternative: AR({int(alternative['ar_order'])}), Delta BIC = {alternative['delta_bic_within_reported_pair']:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=COLORS["muted_ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.5},
    )
    ax.set_ylabel(ylabel)
    style_axes(ax, show_y_grid=True)


def figure_2_temporal_trajectory(inputs: PublicationInputs, output_stem: Path) -> list[Path]:
    frames = inputs.frames
    polarization = frames["polarization"].loc[
        frames["polarization"]["dimension"].eq("q1")
    ].copy()
    pbi = frames["party_brand_index"].copy()
    with apa_style_context():
        fig, axes = plt.subplots(
            2,
            1,
            figsize=figure_size("double_column_tall"),
            sharex=True,
        )
        _temporal_panel(
            axes[0],
            points=polarization,
            value="polarization",
            lower="pol_ci_lower",
            upper="pol_ci_upper",
            series="Polarization",
            ylabel="Polarization (Points)",
            models=frames["breakpoint_models"],
            fitted=frames["breakpoint_fitted"],
        )
        _temporal_panel(
            axes[1],
            points=pbi,
            value="party_brand_index",
            lower="pbi_ci_lower",
            upper="pbi_ci_upper",
            series="Party Brand Index",
            ylabel="Party Brand Index (points)",
            models=frames["breakpoint_models"],
            fitted=frames["breakpoint_fitted"],
        )
        axes[0].set_title(CONSTRUCT_LABELS["q1_polarization"])
        axes[1].set_title("Party Brand Index")
        axes[1].set_xlabel("Month")
        format_month_axis(
            axes[1],
            interval=2,
            start=STUDY_START,
            end=STUDY_END,
        )
        add_panel_label(axes[0], "A")
        add_panel_label(axes[1], "B")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=5,
            frameon=False,
            fontsize=7.3,
            handlelength=2.0,
        )
        fig.suptitle(
            "Temporal Trajectory of Polarization and the Party Brand Index",
            y=0.995,
            fontsize=11.5,
        )
        fig.text(
            0.5,
            0.015,
            format_figure_note(
                "Panel A shows General Ukraine Aid Stance polarization, discussed under RQ1. Panel B shows the "
                "Party Brand Index, discussed under RQ4. Bands are pointwise 95% article-cluster measurement "
                "intervals, and onset lines mark estimated new-regime starts."
            ),
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
        fig.subplots_adjust(top=0.84, bottom=0.15, hspace=0.34)
        outputs = save_figure(
            fig,
            output_stem,
            metadata={
                "Title": "Temporal Trajectory of Polarization and the Party Brand Index",
                "Author": "Paper 3 validated publication pipeline",
                "Subject": "March 2022 to December 2024 structural trajectories",
            },
        )
        plt.close(fig)
    return outputs


def figure_3_party_stance_and_contrast(inputs: PublicationInputs, output_stem: Path) -> list[Path]:
    stance = inputs.frames["stance"].loc[inputs.frames["stance"]["dimension"].eq("q1")].copy()
    trends = inputs.frames["trends"].set_index("series")
    contrast = inputs.frames["party_slope_contrast"].iloc[0]
    with apa_style_context():
        fig, (trajectory_ax, slope_ax) = plt.subplots(
            1,
            2,
            figsize=figure_size("double_column"),
            gridspec_kw={"width_ratios": [1.8, 1.0]},
        )
        for party, label in (("republican", "Republican"), ("democrat", "Democratic")):
            values = stance.loc[stance["party"].eq(party)].sort_values("year_month").reset_index(drop=True)
            dates = _dates(values)
            style = SERIES_STYLES[party]
            _fill_measurement_interval(
                trajectory_ax,
                dates,
                values["ci_lower"].to_numpy(float),
                values["ci_upper"].to_numpy(float),
                role=f"{party}_measurement",
                label="95% measurement interval" if party == "republican" else None,
            )
            trajectory_ax.plot(
                dates,
                values["mean"].to_numpy(float),
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["markeredgecolor"],
                linewidth=style["linewidth"],
                markersize=3.4,
                label=label,
            )
        trajectory_ax.set_title("Monthly General Ukraine Aid Stance by Party")
        trajectory_ax.set_xlabel("Month")
        trajectory_ax.set_ylabel("General Ukraine Aid Stance (Points)")
        trajectory_ax.set_ylim(0, 75)
        format_month_axis(
            trajectory_ax,
            interval=2,
            start=STUDY_START,
            end=STUDY_END,
        )
        style_axes(trajectory_ax, show_y_grid=True)
        trajectory_ax.legend(loc="upper left", ncol=1)
        add_panel_label(trajectory_ax, "A")

        slope_data = [
            (
                "Republican stance",
                float(trends.loc["Q1 Republican stance", "sen_slope"]),
                float(trends.loc["Q1 Republican stance", "sen_slope_ci_lower"]),
                float(trends.loc["Q1 Republican stance", "sen_slope_ci_upper"]),
                SERIES_STYLES["republican"],
            ),
            (
                "Democratic stance",
                float(trends.loc["Q1 Democratic stance", "sen_slope"]),
                float(trends.loc["Q1 Democratic stance", "sen_slope_ci_lower"]),
                float(trends.loc["Q1 Democratic stance", "sen_slope_ci_upper"]),
                SERIES_STYLES["democrat"],
            ),
            (
                "Paired R-D contrast",
                float(contrast["paired_sen_slope_contrast"]),
                float(contrast["paired_sen_slope_ci_lower"]),
                float(contrast["paired_sen_slope_ci_upper"]),
                SERIES_STYLES["comparison"],
            ),
        ]
        y_positions = np.asarray([2, 1, 0], dtype=float)
        for y, (label, estimate, lower, upper, style) in zip(y_positions, slope_data):
            slope_ax.errorbar(
                estimate,
                y,
                xerr=np.asarray([[estimate - lower], [upper - estimate]]),
                fmt=style["marker"],
                color=style["color"],
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["markeredgecolor"],
                markersize=5,
                linewidth=1.0,
                capsize=2.5,
            )
        slope_ax.axvline(0, **REFERENCE_STYLES["zero"])
        slope_ax.set_yticks(y_positions, [item[0] for item in slope_data])
        slope_ax.set_xlabel("Sen Slope (Stance Points per Month)")
        slope_ax.set_title("Temporal-Change Estimates")
        slope_ax.text(
            0.98,
            0.76,
            "Direct paired contrast\n0.289896 [0.060754, 0.584725]\np = .006378",
            transform=slope_ax.transAxes,
            ha="right",
            va="center",
            fontsize=7.3,
            color=COLORS["muted_ink"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        )
        style_axes(slope_ax, show_y_grid=True)
        add_panel_label(slope_ax, "B", x=-0.18)
        fig.suptitle(
            "Party Stance Trajectories and Direct Paired Change",
            y=0.995,
            fontsize=11.5,
        )
        fig.text(
            0.5,
            0.02,
            format_figure_note(
                "Panel A shows monthly General Ukraine Aid Stance for each party. Panel B shows the two marginal "
                "Sen slopes and the direct paired contrast. Bands are pointwise 95% article-cluster measurement "
                "intervals; forest intervals are 95% Sen rank intervals."
            ),
            ha="center",
            va="bottom",
            fontsize=7.4,
        )
        fig.subplots_adjust(top=0.84, bottom=0.23, left=0.10, right=0.98, wspace=0.34)
        outputs = save_figure(
            fig,
            output_stem,
            metadata={
                "Title": "Party Stance Trajectories and Direct Paired Change",
                "Author": "Paper 3 validated publication pipeline",
                "Subject": "RQ2 party attribution from March 2022 to December 2024",
            },
        )
        plt.close(fig)
    return outputs


def figure_s2_breakpoint_evidence(inputs: PublicationInputs, output_stem: Path) -> list[Path]:
    frames = inputs.frames
    models = frames["breakpoint_models"]
    fitted = frames["breakpoint_fitted"]
    regimes = frames["breakpoint_regimes"]
    with apa_style_context():
        fig, axes = plt.subplots(2, 2, figsize=figure_size("supplement_wide"))
        for ax, series in (
            (axes[0, 0], "Polarization"),
            (axes[0, 1], "Party Brand Index"),
        ):
            selected, selected_values = _model_rows(models, fitted, series, "selected_primary")
            alternative, alternative_values = _model_rows(models, fitted, series, "required_alternative")
            dates = pd.to_datetime(selected_values["year_month"])
            dimension_style = _dimension_style(series)
            ax.plot(
                dates,
                selected_values["observed"],
                color=SERIES_STYLES["observed"]["color"],
                linestyle=SERIES_STYLES["observed"]["linestyle"],
                marker=dimension_style["marker"],
                markerfacecolor=(
                    SERIES_STYLES["observed"]["color"]
                    if series == "Polarization"
                    else COLORS["background"]
                ),
                markeredgecolor=SERIES_STYLES["observed"]["color"],
                markersize=3,
                linewidth=0.9,
                label="Observed",
            )
            plot_structural_mean(
                ax,
                dates,
                selected_values["structural_regime_mean"],
                role="selected",
                label=f"Selected, BIC {selected['joint_selection_bic']:.3f}",
                annotate=True,
            )
            plot_structural_mean(
                ax,
                dates,
                alternative_values["structural_regime_mean"],
                role="alternative",
                label=f"Alternative, BIC {alternative['joint_selection_bic']:.3f}",
                annotate=False,
                zorder=2,
            )
            display_series = (
                CONSTRUCT_LABELS["q1_polarization"]
                if series == "Polarization"
                else "Party Brand Index"
            )
            ax.set_title(display_series)
            ax.set_xlabel("Month")
            ax.set_ylabel("Points")
            format_month_axis(
                ax,
                interval=2,
                rotation=50,
                start=STUDY_START,
                end=STUDY_END,
            )
            style_axes(ax, show_y_grid=True)
            ax.legend(loc="best")

        labels = []
        deltas = []
        styles = []
        roles = []
        for _, row in models.iterrows():
            series_short = (
                "Aid Polarization"
                if row["series"] == "Polarization"
                else "Party Brand Index"
            )
            role_short = "selected" if row["reporting_role"] == "selected_primary" else "alternative"
            labels.append(f"{series_short}, {role_short}")
            deltas.append(float(row["delta_bic_within_reported_pair"]))
            styles.append(_dimension_style(str(row["series"])))
            roles.append("selected" if row["reporting_role"] == "selected_primary" else "alternative")
        y = np.arange(len(labels))[::-1]
        for position, delta, style, role in zip(y, deltas, styles, roles):
            role_style = SERIES_STYLES[role]
            axes[1, 0].hlines(
                position,
                0,
                delta,
                color=style["color"],
                linewidth=1.0,
                linestyles=role_style["linestyle"],
            )
            axes[1, 0].plot(
                delta,
                position,
                marker=style["marker"],
                color=style["color"],
                markerfacecolor=style["color"] if role == "selected" else COLORS["background"],
                markeredgecolor=style["color"],
                markersize=5,
            )
            label_x = delta - 0.15 if delta >= 8.0 else delta + 0.15
            label_alignment = "right" if delta >= 8.0 else "left"
            axes[1, 0].text(
                label_x,
                position,
                f"{delta:.3f}",
                ha=label_alignment,
                va="center",
                fontsize=7.5,
            )
        axes[1, 0].set_yticks(y, labels)
        axes[1, 0].set_xlabel("Delta BIC within reported pair")
        axes[1, 0].set_title("Required Model Comparison")
        axes[1, 0].set_xlim(-0.1, 10.4)
        style_axes(axes[1, 0], show_y_grid=True)

        shift_rows = regimes.dropna(subset=["adjacent_structural_shift"]).copy()
        shift_labels = []
        for _, row in shift_rows.iterrows():
            series_short = (
                "Aid Polarization"
                if row["series"] == "Polarization"
                else "Party Brand Index"
            )
            role_short = "sel" if row["reporting_role"] == "selected_primary" else "alt"
            shift_labels.append(f"{series_short} {role_short}, {row['start_date']}")
        shift_values = shift_rows["adjacent_structural_shift"].to_numpy(float)
        y_shift = np.arange(len(shift_rows))[::-1]
        for position, (_, row), value in zip(y_shift, shift_rows.iterrows(), shift_values):
            style = _dimension_style(str(row["series"]))
            role = "selected" if row["reporting_role"] == "selected_primary" else "alternative"
            axes[1, 1].hlines(
                position,
                0,
                value,
                color=style["color"],
                linewidth=1.0,
                linestyles=SERIES_STYLES[role]["linestyle"],
            )
            axes[1, 1].plot(
                value,
                position,
                marker=style["marker"],
                color=style["color"],
                markerfacecolor=style["color"] if role == "selected" else COLORS["background"],
                markeredgecolor=style["color"],
                markersize=4.5,
            )
        axes[1, 1].axvline(0, **REFERENCE_STYLES["zero"])
        axes[1, 1].set_yticks(y_shift, shift_labels)
        axes[1, 1].set_xlabel("Adjacent structural shift (points)")
        axes[1, 1].set_title("Reported Regime Shifts")
        style_axes(axes[1, 1], show_y_grid=True)
        axes[1, 0].tick_params(axis="y", labelsize=7.4)
        axes[1, 1].tick_params(axis="y", labelsize=7.1)
        for index, (label, ax) in enumerate(zip(("A", "B", "C", "D"), axes.flat)):
            add_panel_label(ax, label, x=-0.16 if index >= 2 else -0.12)
        fig.suptitle("Supplementary Structural Model Evidence", y=0.995, fontsize=11.5)
        fig.text(
            0.5,
            0.015,
            format_figure_note(
                "BIC selected the reported models. Visual fit did not select the models, and the panels do not "
                "quantify breakpoint-date uncertainty."
            ),
            ha="center",
            fontsize=7.3,
        )
        fig.subplots_adjust(top=0.92, bottom=0.11, left=0.11, right=0.98, hspace=0.50, wspace=0.38)
        outputs = save_figure(
            fig,
            output_stem,
            metadata={
                "Title": "Supplementary Structural Model Evidence",
                "Author": "Paper 3 validated publication pipeline",
                "Subject": "Selected and required alternative breakpoint models",
            },
        )
        plt.close(fig)
    return outputs


def figure_s3_breakpoint_diagnostics(inputs: PublicationInputs, output_stem: Path) -> list[Path]:
    models = inputs.frames["breakpoint_models"]
    fitted = inputs.frames["breakpoint_fitted"]
    with apa_style_context():
        fig, axes = plt.subplots(2, 2, figsize=figure_size("supplement_grid"))
        for row_index, series in enumerate(("Polarization", "Party Brand Index")):
            _, values = _model_rows(models, fitted, series, "selected_primary")
            dates = pd.to_datetime(values["year_month"])
            innovations = values["standardized_innovation"].to_numpy(float)
            dimension_style = _dimension_style(series)
            time_ax = axes[row_index, 0]
            acf_ax = axes[row_index, 1]
            time_ax.axhline(0, **REFERENCE_STYLES["zero"])
            time_ax.plot(
                dates,
                innovations,
                color=dimension_style["color"],
                linestyle=dimension_style["linestyle"],
                marker=dimension_style["marker"],
                markerfacecolor=dimension_style["markerfacecolor"],
                markeredgecolor=dimension_style["color"],
                markersize=3.2,
                linewidth=1.1,
            )
            display_series = (
                CONSTRUCT_LABELS["q1_polarization"]
                if series == "Polarization"
                else "Party Brand Index"
            )
            time_ax.set_title(
                f"{display_series}\nStandardized Innovations",
                fontsize=9.1,
                linespacing=1.0,
            )
            time_ax.set_ylabel("Standardized innovation")
            time_ax.set_xlabel("Month")
            format_month_axis(
                time_ax,
                interval=4,
                rotation=45,
                start=STUDY_START,
                end=STUDY_END,
            )
            style_axes(time_ax, show_y_grid=True)

            acf_values = acf(innovations, nlags=10, fft=False, missing="raise")
            lags = np.arange(1, len(acf_values))
            acf_ax.axhline(0, color=COLORS["zero"], linewidth=0.8)
            acf_ax.vlines(lags, 0, acf_values[1:], color=dimension_style["color"], linewidth=1.0)
            acf_ax.plot(
                lags,
                acf_values[1:],
                linestyle="None",
                marker=dimension_style["marker"],
                color=dimension_style["color"],
                markerfacecolor=dimension_style["markerfacecolor"],
                markeredgecolor=dimension_style["color"],
            )
            threshold = 1.96 / np.sqrt(len(innovations))
            acf_ax.axhline(threshold, **REFERENCE_STYLES["diagnostic_threshold"])
            acf_ax.axhline(-threshold, **REFERENCE_STYLES["diagnostic_threshold"])
            acf_ax.set_title(
                f"{display_series}\nInnovation ACF",
                fontsize=9.1,
                linespacing=1.0,
            )
            acf_ax.set_xlabel("Lag (months)")
            acf_ax.set_ylabel("Autocorrelation")
            acf_ax.set_xticks(lags)
            acf_ax.set_ylim(-0.65, 0.65)
            style_axes(acf_ax, show_y_grid=True)
        for label, ax in zip(("A", "B", "C", "D"), axes.flat):
            add_panel_label(ax, label, x=-0.16, y=1.14)
        fig.suptitle("Selected Model Innovation Diagnostics", y=0.995, fontsize=11.5)
        fig.text(
            0.5,
            0.015,
            format_figure_note(
                "Dashed ACF guides equal plus or minus 1.96 divided by the square root of 34 and are diagnostic "
                "reference thresholds, not inferential evidence for breakpoint selection."
            ),
            ha="center",
            fontsize=7.3,
        )
        fig.subplots_adjust(top=0.87, bottom=0.11, left=0.10, right=0.98, hspace=0.66, wspace=0.38)
        outputs = save_figure(
            fig,
            output_stem,
            metadata={
                "Title": "Selected Model Innovation Diagnostics",
                "Author": "Paper 3 validated publication pipeline",
                "Subject": "Supplementary breakpoint residual evidence",
            },
        )
        plt.close(fig)
    return outputs


def build_publication_figures(
    inputs: PublicationInputs,
    output_directory: Path,
) -> tuple[list[Path], Path]:
    """Render the manuscript figures and supplementary evidence figures."""

    main_directory = output_directory
    supplement_directory = output_directory / "supplement"
    main_directory.mkdir(parents=True, exist_ok=True)
    supplement_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs.extend(
        figure_2_temporal_trajectory(
            inputs, main_directory / "figure_2_temporal_trajectory"
        )
    )
    outputs.extend(
        figure_3_party_stance_and_contrast(
            inputs, main_directory / "figure_3_party_stance_and_contrast"
        )
    )
    outputs.extend(
        figure_s2_breakpoint_evidence(
            inputs, supplement_directory / "figure_s2_breakpoint_evidence"
        )
    )
    outputs.extend(
        figure_s3_breakpoint_diagnostics(
            inputs, supplement_directory / "figure_s3_breakpoint_diagnostics"
        )
    )
    specs: Dict[str, dict] = {
        "figure_2": {
            "placement": "main",
            "title": "Temporal Trajectory of Polarization and the Party Brand Index",
            "panels": ["General Ukraine Aid Stance Polarization", "Party Brand Index"],
            "period": EXPECTED_MONTHS,
            "uncertainty": "pointwise article-cluster measurement intervals",
            "does_not_cover": ["breakpoint dates", "model selection", "serial-dependence estimation", "LLM scoring"],
            "non_color_cues": ["gray observed circles", "thin dashed black selected means with inline values", "gray dash-dot alternative means", "dashed onset lines with arrows and month labels"],
        },
        "figure_3": {
            "placement": "main",
            "title": "Party Stance Trajectories and Direct Paired Change",
            "panels": ["monthly party stance", "Sen slope intervals"],
            "period": EXPECTED_MONTHS,
            "inferential_role": "direct paired Republican-minus-Democratic slope contrast",
            "non_color_cues": ["solid filled circles", "dashed open squares", "dash-dot open diamond contrast"],
        },
        "figure_s2": {
            "placement": "supplement",
            "title": "Supplementary Structural Model Evidence",
            "panels": ["polarization comparison", "Party Brand Index comparison", "delta BIC", "regime shifts"],
        },
        "figure_s3": {
            "placement": "supplement",
            "title": "Selected Model Innovation Diagnostics",
            "panels": ["polarization innovations", "polarization ACF", "Party Brand Index innovations", "Party Brand Index ACF"],
            "status": "diagnostic evidence, not substantive publication result",
        },
    }
    manifest_path = output_directory / "figure_specs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": FIGURE_SCHEMA,
                "style_version": STYLE_VERSION,
                "palette_mode": "monochrome",
                "reference_grammar": (
                    "gray raw observations and confidence shading; thin dashed labeled structural means; "
                    "patterned open-marker comparisons; dashed breakpoint references with month labels"
                ),
                "figures": specs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return outputs, manifest_path
