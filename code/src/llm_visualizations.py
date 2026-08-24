"""Diagnostic visualizations for the Paper 3 temporal series."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import DATA_PATH
from src.plot_style import (
    CONSTRUCT_LABELS,
    SERIES_STYLES,
    UNCERTAINTY_STYLES,
    add_breakpoint_lines,
    apply_apa_style,
    format_figure_note,
    format_month_axis,
    plot_structural_mean,
    style_axes,
)


apply_apa_style()


class LLMVisualizations:
    """Render the temporal diagnostic figures."""

    AGGREGATED_DIR = os.path.join(DATA_PATH, "aggregated")
    STATISTICS_DIR = os.path.join(DATA_PATH, "statistics")
    OUTPUT_DIR = os.path.join(DATA_PATH, "statistics")

    def __init__(
        self,
        aggregated_dir: Optional[str] = None,
        statistics_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.AGGREGATED_DIR = aggregated_dir or self.AGGREGATED_DIR
        self.STATISTICS_DIR = statistics_dir or self.STATISTICS_DIR
        self.OUTPUT_DIR = output_dir or self.OUTPUT_DIR

    def run(self) -> bool:
        """Render the two temporal diagnostics."""

        statistics = Path(self.STATISTICS_DIR)
        destination = Path(self.OUTPUT_DIR) / "temporal_diagnostics"
        destination.mkdir(parents=True, exist_ok=True)
        models = pd.read_csv(statistics / "breakpoint_models.csv")
        fitted = pd.read_csv(statistics / "breakpoint_fitted_series.csv")

        outputs = {
            "polarization": destination
            / "q1_polarization_measurement_and_breakpoint.png",
            "party_brand_index": destination
            / "party_brand_index_measurement_and_breakpoint.png",
        }
        polarization = pd.read_csv(
            statistics / "monthly_polarization_joint_bootstrap.csv"
        )
        polarization = polarization.loc[
            polarization["dimension"].eq("q1")
        ].sort_values("year_month")
        self._plot_temporal_contract(
            points=polarization,
            value_column="polarization",
            lower_column="pol_ci_lower",
            upper_column="pol_ci_upper",
            series="Polarization",
            ylabel="Polarization (Points)",
            title=f"{CONSTRUCT_LABELS['q1_polarization']} and Selected Structural Account",
            models=models,
            fitted=fitted,
            output=outputs["polarization"],
        )

        pbi = pd.read_csv(statistics / "party_brand_index_timeseries.csv").sort_values(
            "year_month"
        )
        self._plot_temporal_contract(
            points=pbi,
            value_column="party_brand_index",
            lower_column="pbi_ci_lower",
            upper_column="pbi_ci_upper",
            series="Party Brand Index",
            ylabel="Party Brand Index (points)",
            title="Party Brand Index and Selected Structural Account",
            models=models,
            fitted=fitted,
            output=outputs["party_brand_index"],
        )

        print(f"Rendered {len(outputs)} diagnostic figures to {destination}")
        return True

    @staticmethod
    def _plot_temporal_contract(
        points: pd.DataFrame,
        value_column: str,
        lower_column: str,
        upper_column: str,
        series: str,
        ylabel: str,
        title: str,
        models: pd.DataFrame,
        fitted: pd.DataFrame,
        output: Path,
    ) -> None:
        selected = models.loc[
            models["series"].eq(series)
            & models["reporting_role"].eq("selected_primary")
        ].iloc[0]
        model_values = (
            fitted.loc[fitted["model_id"].eq(selected["model_id"])]
            .sort_values("year_month")
            .reset_index(drop=True)
        )
        points = points.reset_index(drop=True)
        if len(points) != 34 or len(model_values) != 34:
            raise ValueError(f"{series} figure inputs do not contain 34 months")
        if points["year_month"].astype(str).tolist() != model_values[
            "year_month"
        ].astype(str).tolist():
            raise ValueError(f"{series} point and breakpoint calendars differ")
        dates = pd.to_datetime(points["year_month"])
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        interval_style = UNCERTAINTY_STYLES["measurement"]
        ax.fill_between(
            dates,
            points[lower_column].to_numpy(float),
            points[upper_column].to_numpy(float),
            facecolor=interval_style["facecolor"],
            edgecolor=interval_style["edgecolor"],
            alpha=interval_style["alpha"],
            linewidth=interval_style["linewidth"],
            label="Pointwise 95% Article-Cluster Interval",
        )
        observed_style = SERIES_STYLES["observed"]
        ax.plot(
            dates,
            points[value_column],
            color=observed_style["color"],
            linestyle=observed_style["linestyle"],
            marker=observed_style["marker"],
            markerfacecolor=observed_style["markerfacecolor"],
            markeredgecolor=observed_style["markeredgecolor"],
            markersize=3.5,
            linewidth=observed_style["linewidth"],
            label="Observed monthly value",
        )
        plot_structural_mean(
            ax,
            dates,
            model_values["structural_regime_mean"],
            role="selected",
            label="Selected Structural Regime Mean",
            annotate=True,
        )
        break_dates = [
            item
            for item in str(selected["break_onset_dates"]).split("|")
            if item and item != "none"
        ]
        add_breakpoint_lines(
            ax,
            [pd.Timestamp(value + "-01") for value in break_dates],
            show_arrows=True,
        )
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        format_month_axis(
            ax,
            interval=2,
            start=dates.iloc[0],
            end=dates.iloc[-1],
        )
        style_axes(ax, show_y_grid=True)
        ax.legend(loc="best", frameon=False)
        fig.text(
            0.5,
            0.01,
            format_figure_note(
                "Bands quantify monthly article-sampling uncertainty, not breakpoint-date or model-selection "
                "uncertainty."
            ),
            ha="center",
            fontsize=8,
        )
        fig.subplots_adjust(bottom=0.22)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
