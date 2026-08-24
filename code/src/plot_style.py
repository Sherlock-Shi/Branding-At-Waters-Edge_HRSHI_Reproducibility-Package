"""Repository-local Seaborn and Matplotlib style system for Paper 3.

This module is the visual source of truth. It implements the approved
monochrome figure grammar through semantic roles rather than chart-specific
colors. The roles support publication, supplement, diagnostic, and
presentation figures without importing analytical meaning from a reference
image.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import isclose
import os
from pathlib import Path
import tempfile
from typing import Iterator, Mapping, Sequence

from cycler import cycler
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
import seaborn as sns


STYLE_VERSION = "paper3-apa-monochrome-v4"
PALETTE_MODE = "monochrome"
FONT_FAMILY = "Times New Roman"
FONT_FALLBACKS = ["Times New Roman", "Liberation Serif", "DejaVu Serif", "serif"]
SAMPLE_PERIOD = "March 2022 to December 2024"
BREAKPOINT_DATE_FORMAT = "%Y-%m"
FIGURE_NOTE_PREFIX = r"$\mathit{Note.}$"


COLORS = {
    "republican": "#111111",
    "democrat": "#555555",
    "q1": "#111111",
    "q2": "#4D4D4D",
    "q3": "#737373",
    "q4": "#969696",
    "party_brand_index": "#4D4D4D",
    "observed": "#737373",
    "selected": "#111111",
    "alternative": "#737373",
    "comparison": "#333333",
    "breakpoint": "#333333",
    "segment_mean": "#808080",
    "zero": "#4D4D4D",
    "ink": "#1A1A1A",
    "muted_ink": "#595959",
    "grid": "#E3E3E3",
    "neutral": "#A6A6A6",
    "confidence": "#D9D9D9",
    "confidence_light": "#ECECEC",
    "bar_fill": "#D0D0D0",
    "background": "#FFFFFF",
}


DIM_LABELS = {
    "q1": "General Ukraine Aid Stance",
    "q2": "Ukraine Characterization",
    "q3": "Threat Priority",
    "q4": "Resource Allocation",
}


DIM_SHORT = {
    "q1": "General Aid Stance",
    "q2": "Ukraine Char.",
    "q3": "Threat Priority",
    "q4": "Resource Alloc.",
}


CONSTRUCT_LABELS = {
    "q1_stance": "General Ukraine Aid Stance",
    "q1_polarization": "General Ukraine Aid Stance Polarization",
    "q1_message_coherence": "General Ukraine Aid Stance Message Coherence",
    "q1_message_dispersion": "General Ukraine Aid Stance Message Dispersion",
    "party_brand_index": "Party Brand Index",
}


SERIES_STYLES = {
    "republican": {
        "color": COLORS["republican"],
        "linestyle": "-",
        "marker": "o",
        "markerfacecolor": COLORS["republican"],
        "markeredgecolor": COLORS["republican"],
        "linewidth": 1.1,
    },
    "democrat": {
        "color": COLORS["democrat"],
        "linestyle": "--",
        "marker": "s",
        "markerfacecolor": COLORS["background"],
        "markeredgecolor": COLORS["democrat"],
        "linewidth": 1.1,
    },
    "observed": {
        "color": COLORS["observed"],
        "linestyle": "-",
        "marker": "o",
        "markerfacecolor": COLORS["observed"],
        "markeredgecolor": COLORS["observed"],
        "linewidth": 0.9,
    },
    "selected": {
        "color": COLORS["selected"],
        "linestyle": "--",
        "marker": None,
        "linewidth": 0.8,
    },
    "alternative": {
        "color": COLORS["alternative"],
        "linestyle": "-.",
        "marker": None,
        "linewidth": 0.9,
    },
    "comparison": {
        "color": COLORS["comparison"],
        "linestyle": "-.",
        "marker": "D",
        "markerfacecolor": COLORS["background"],
        "markeredgecolor": COLORS["comparison"],
        "linewidth": 1.0,
    },
    "segment_mean": {
        "color": COLORS["segment_mean"],
        "linestyle": "--",
        "marker": None,
        "linewidth": 0.8,
    },
    "breakpoint": {
        "color": COLORS["breakpoint"],
        "linestyle": "--",
        "marker": None,
        "linewidth": 0.8,
    },
}


DIMENSION_STYLES = {
    "q1": {"color": COLORS["q1"], "linestyle": "-", "marker": "o", "markerfacecolor": COLORS["q1"]},
    "q2": {"color": COLORS["q2"], "linestyle": "--", "marker": "s", "markerfacecolor": COLORS["background"]},
    "q3": {"color": COLORS["q3"], "linestyle": "-.", "marker": "D", "markerfacecolor": COLORS["background"]},
    "q4": {"color": COLORS["q4"], "linestyle": ":", "marker": "^", "markerfacecolor": COLORS["background"]},
    "party_brand_index": {
        "color": COLORS["party_brand_index"],
        "linestyle": "-.",
        "marker": "D",
        "markerfacecolor": COLORS["background"],
    },
}


UNCERTAINTY_STYLES = {
    "measurement": {
        "facecolor": COLORS["confidence"],
        "edgecolor": "#BDBDBD",
        "alpha": 0.62,
        "linewidth": 0.45,
    },
    "republican_measurement": {
        "facecolor": "#C9C9C9",
        "edgecolor": "#969696",
        "alpha": 0.34,
        "linewidth": 0.35,
    },
    "democrat_measurement": {
        "facecolor": COLORS["confidence_light"],
        "edgecolor": "#B3B3B3",
        "alpha": 0.58,
        "linewidth": 0.35,
    },
    "conditional_interval": {
        "color": COLORS["ink"],
        "alpha": 1.0,
        "linewidth": 1.2,
        "capsize": 2.5,
    },
    "reference": {
        "color": COLORS["zero"],
        "alpha": 1.0,
        "linewidth": 0.8,
        "linestyle": ":",
    },
}


REFERENCE_STYLES = {
    "zero": {"color": COLORS["zero"], "linestyle": ":", "linewidth": 0.8},
    "breakpoint": {"color": COLORS["breakpoint"], "linestyle": "--", "linewidth": 0.8},
    "segment_mean": {"color": COLORS["segment_mean"], "linestyle": "--", "linewidth": 0.8},
    "diagnostic_threshold": {"color": COLORS["alternative"], "linestyle": "--", "linewidth": 0.8},
}


BAR_STYLES = {
    "primary": {"facecolor": COLORS["bar_fill"], "edgecolor": COLORS["ink"], "hatch": None},
    "secondary": {"facecolor": COLORS["background"], "edgecolor": COLORS["ink"], "hatch": "///"},
    "tertiary": {"facecolor": "#AFAFAF", "edgecolor": COLORS["ink"], "hatch": "..."},
}


HATCH_SEQUENCE = (None, "///", "\\\\", "...", "xx", "++")


CHART_FAMILY_DEFAULTS = {
    "time_series": {
        "raw_series": "observed",
        "primary_series": "selected",
        "comparison_series": "comparison",
        "uncertainty": "measurement",
        "structural_reference": "breakpoint",
        "month_tick_interval": 2,
    },
    "dual_axis_time_series": {
        "left_axis_series": "observed",
        "right_axis_series": "comparison",
        "secondary_axis_helper": "style_secondary_y_axis",
        "structural_reference": "breakpoint",
        "month_tick_interval": 2,
    },
    "party_comparison": {
        "primary_group": "republican",
        "secondary_group": "democrat",
        "uncertainty_roles": ("republican_measurement", "democrat_measurement"),
    },
    "coefficient_forest": {
        "interval": "conditional_interval",
        "reference": "zero",
        "group_markers": ("o", "s", "D", "^"),
    },
    "bar_or_count": {
        "fills": ("primary", "secondary", "tertiary"),
        "hatches": HATCH_SEQUENCE,
    },
    "distribution": {
        "outline": "ink",
        "fills": ("confidence", "confidence_light"),
        "hatches": HATCH_SEQUENCE,
    },
    "model_comparison": {
        "selected": "selected",
        "alternative": "alternative",
        "comparison": "comparison",
    },
    "diagnostic": {
        "observed": "observed",
        "zero_reference": "zero",
        "threshold": "diagnostic_threshold",
    },
    "multi_panel": {
        "panel_labels": "add_panel_label",
        "shared_typography": STYLE_VERSION,
    },
}


@dataclass(frozen=True)
class FigurePreset:
    """Physical figure dimensions in inches."""

    width: float
    height: float


FIGURE_PRESETS = {
    "single_column": FigurePreset(3.35, 2.65),
    "single_column_tall": FigurePreset(3.35, 4.25),
    "double_column": FigurePreset(7.20, 4.40),
    "double_column_tall": FigurePreset(7.20, 6.60),
    "wide_diagnostic": FigurePreset(9.20, 5.20),
    "supplement_grid": FigurePreset(7.50, 7.50),
    "supplement_wide": FigurePreset(9.20, 7.20),
    "presentation": FigurePreset(10.00, 5.625),
}


EXPORT_SETTINGS = {
    "dpi": 300,
    "bbox_inches": "tight",
    "pad_inches": 0.08,
    "facecolor": "white",
    "transparent": False,
}


APA_RCPARAMS = {
    "font.family": "serif",
    "font.serif": FONT_FALLBACKS,
    "font.size": 9.5,
    "figure.dpi": 300,
    "figure.facecolor": COLORS["background"],
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "savefig.facecolor": COLORS["background"],
    "savefig.transparent": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.facecolor": COLORS["background"],
    "axes.edgecolor": COLORS["ink"],
    "axes.labelcolor": COLORS["ink"],
    "axes.titlecolor": COLORS["ink"],
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "axes.titleweight": "normal",
    "axes.titlepad": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(
        color=[
            COLORS["q1"],
            COLORS["party_brand_index"],
            COLORS["q2"],
            COLORS["q3"],
            COLORS["q4"],
        ]
    )
    + cycler(linestyle=["-", "--", "-.", ":", (0, (5, 2, 1, 2))]),
    "xtick.color": COLORS["ink"],
    "ytick.color": COLORS["ink"],
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 8.5,
    "legend.frameon": False,
    "legend.handlelength": 2.4,
    "legend.columnspacing": 1.1,
    "lines.linewidth": 1.1,
    "lines.markersize": 4.0,
    "lines.markeredgewidth": 0.7,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "butt",
    "errorbar.capsize": 2.5,
    "patch.edgecolor": COLORS["ink"],
    "patch.linewidth": 0.8,
    "hatch.linewidth": 0.6,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.6,
    "grid.alpha": 0.80,
}


def apply_apa_style(context: str = "paper", font_scale: float = 1.0) -> None:
    """Apply the repository-local Seaborn baseline and Matplotlib overrides."""

    sns.set_theme(
        context=context,
        style="ticks",
        palette=[
            COLORS["q1"],
            COLORS["party_brand_index"],
            COLORS["q2"],
            COLORS["q3"],
            COLORS["q4"],
        ],
        font=FONT_FAMILY,
        font_scale=font_scale,
        color_codes=False,
        rc=APA_RCPARAMS,
    )
    mpl.rcParams.update(APA_RCPARAMS)


@contextmanager
def apa_style_context(context: str = "paper", font_scale: float = 1.0) -> Iterator[None]:
    """Temporarily apply the Paper 3 theme without leaking global changes."""

    previous = mpl.rcParams.copy()
    apply_apa_style(context=context, font_scale=font_scale)
    try:
        yield
    finally:
        mpl.rcParams.update(previous)


def figure_size(preset: str) -> tuple[float, float]:
    """Return a named physical figure size."""

    if preset not in FIGURE_PRESETS:
        raise KeyError(f"unknown figure preset: {preset}")
    value = FIGURE_PRESETS[preset]
    return value.width, value.height


def style_axes(
    ax: Axes,
    *,
    show_x_grid: bool = False,
    show_y_grid: bool = True,
    zero_line: bool = False,
) -> Axes:
    """Apply quiet axes with horizontal background guides only."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    if show_x_grid or show_y_grid:
        ax.grid(visible=False, axis="x")
        ax.grid(
            visible=True,
            axis="y",
            color=COLORS["grid"],
            linewidth=0.6,
            alpha=0.80,
        )
        ax.set_axisbelow(True)
    else:
        ax.grid(visible=False)
    if zero_line:
        ax.axhline(
            0,
            color=COLORS["zero"],
            linewidth=UNCERTAINTY_STYLES["reference"]["linewidth"],
            linestyle=UNCERTAINTY_STYLES["reference"]["linestyle"],
            zorder=0,
        )
    return ax


def style_secondary_y_axis(ax: Axes) -> Axes:
    """Enable a restrained right axis for chart families that require it."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color(COLORS["ink"])
    ax.spines["right"].set_linewidth(0.8)
    ax.tick_params(axis="y", direction="out", colors=COLORS["ink"])
    ax.yaxis.label.set_color(COLORS["ink"])
    return ax


def add_breakpoint_lines(
    ax: Axes,
    dates: Sequence[object],
    *,
    label: str = "Estimated regime onset",
    show_arrows: bool = True,
    show_date_labels: bool = True,
    date_format: str = BREAKPOINT_DATE_FORMAT,
) -> None:
    """Draw model-selected structural onsets with arrows and baseline month labels."""

    style = REFERENCE_STYLES["breakpoint"]
    transform = ax.get_xaxis_transform()
    for index, date in enumerate(dates):
        arrow_x = mdates.date2num(date)
        ax.axvline(
            date,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=label if index == 0 else None,
            zorder=0,
        )
        if show_arrows:
            ax.annotate(
                "",
                xy=(arrow_x, 0.985),
                xytext=(arrow_x, 1.055),
                xycoords=transform,
                textcoords=transform,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": COLORS["ink"],
                    "linewidth": 0.8,
                    "mutation_scale": 7.0,
                },
                annotation_clip=False,
            )
        if show_date_labels:
            ax.text(
                arrow_x,
                0.012,
                mdates.num2date(arrow_x).strftime(date_format),
                transform=transform,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=6.4,
                color=COLORS["ink"],
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": COLORS["background"],
                    "edgecolor": COLORS["muted_ink"],
                    "linewidth": 0.4,
                    "alpha": 0.94,
                },
                zorder=6,
                clip_on=True,
            )


def format_figure_note(note: str) -> str:
    """Prefix a figure note with an italicized APA ``Note.`` label."""

    normalized = note.strip()
    if not normalized:
        raise ValueError("figure note must not be empty")
    if normalized.startswith("Note."):
        normalized = normalized[len("Note.") :].lstrip()
    return f"{FIGURE_NOTE_PREFIX} {normalized}"


def plot_structural_mean(
    ax: Axes,
    dates: Sequence[object],
    values: Sequence[float],
    *,
    role: str = "selected",
    label: str | None = None,
    annotate: bool = True,
    decimals: int = 1,
    zorder: float = 4,
) -> None:
    """Draw thin structural means and place each mean value on its segment."""

    if role not in {"selected", "alternative", "segment_mean"}:
        raise KeyError(f"unsupported structural-mean role: {role}")
    date_values = list(dates)
    mean_values = [float(value) for value in values]
    if not date_values or len(date_values) != len(mean_values):
        raise ValueError("structural-mean dates and values must have equal nonzero length")
    style = SERIES_STYLES[role]
    ax.step(
        date_values,
        mean_values,
        where="post",
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        label=label,
        zorder=zorder,
    )
    if not annotate:
        return

    segment_start = 0
    for index in range(1, len(mean_values) + 1):
        at_end = index == len(mean_values)
        changed = not at_end and not isclose(
            mean_values[index],
            mean_values[segment_start],
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        if not at_end and not changed:
            continue
        segment_end = index - 1
        left = mdates.date2num(date_values[segment_start])
        right = mdates.date2num(date_values[segment_end])
        midpoint = (left + right) / 2.0
        mean = mean_values[segment_start]
        ax.text(
            midpoint,
            mean,
            f"{mean:.{decimals}f}",
            ha="center",
            va="center",
            fontsize=6.8,
            color=COLORS["ink"],
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": COLORS["background"],
                "edgecolor": COLORS["muted_ink"],
                "linewidth": 0.4,
                "alpha": 0.96,
            },
            zorder=zorder + 1,
        )
        segment_start = index


def add_panel_label(ax: Axes, label: str, *, x: float = -0.10, y: float = 1.04) -> None:
    """Add a consistent bold panel label in axes coordinates."""

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
    )


def format_month_axis(
    ax: Axes,
    *,
    interval: int = 2,
    rotation: float = 40,
    start: object | None = None,
    end: object | None = None,
) -> None:
    """Format a monthly axis and prevent ticks outside the observed calendar."""

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    if (start is None) != (end is None):
        raise ValueError("month-axis start and end must be supplied together")
    if start is not None and end is not None:
        ax.set_xlim(start, end)
    else:
        ax.margins(x=0)
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_horizontalalignment("right")


def save_figure(
    fig: Figure,
    output_stem: Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
    metadata: Mapping[str, str] | None = None,
) -> list[Path]:
    """Export one figure through the shared 300 dpi raster and vector contract."""

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for extension in formats:
        normalized = extension.lower().lstrip(".")
        if normalized not in {"png", "pdf", "svg"}:
            raise ValueError(f"unsupported publication format: {extension}")
        path = output_stem.with_suffix(f".{normalized}")
        kwargs = dict(EXPORT_SETTINGS)
        if normalized != "png":
            kwargs.pop("dpi", None)
        if metadata:
            kwargs["metadata"] = dict(metadata)
        temporary_handle = tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.",
            suffix=f".{normalized}.tmp",
            dir=path.parent,
            delete=False,
        )
        temporary_path = Path(temporary_handle.name)
        temporary_handle.close()
        try:
            fig.savefig(temporary_path, format=normalized, **kwargs)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        written.append(path)
    return written


def validate_style_contract() -> list[str]:
    """Return style-contract violations for automated gates."""

    failures: list[str] = []
    if APA_RCPARAMS["savefig.dpi"] != 300 or EXPORT_SETTINGS["dpi"] != 300:
        failures.append("raster export must use 300 dpi")
    if APA_RCPARAMS["axes.spines.top"] or APA_RCPARAMS["axes.spines.right"]:
        failures.append("top and right spines must be disabled")
    if APA_RCPARAMS["legend.frameon"]:
        failures.append("legends must be frameless")
    if not APA_RCPARAMS["axes.grid"] or APA_RCPARAMS["axes.grid.axis"] != "y":
        failures.append("background guides must be horizontal only")
    if FONT_FAMILY not in APA_RCPARAMS["font.serif"]:
        failures.append("Times New Roman must lead the serif fallback chain")
    if PALETTE_MODE != "monochrome":
        failures.append("the active Paper 3 palette must be monochrome")
    for name, value in COLORS.items():
        red, green, blue = to_rgb(value)
        if max(red, green, blue) - min(red, green, blue) > 1e-12:
            failures.append(f"semantic color {name} is not grayscale")
    for first, second in (("republican", "democrat"), ("selected", "alternative")):
        if SERIES_STYLES[first]["linestyle"] == SERIES_STYLES[second]["linestyle"]:
            failures.append(f"{first} and {second} require distinct line styles")
        if SERIES_STYLES[first]["marker"] == SERIES_STYLES[second]["marker"] and first == "republican":
            failures.append("party series require distinct markers")
    if SERIES_STYLES["observed"]["color"] == SERIES_STYLES["selected"]["color"]:
        failures.append("observed and selected series require distinct grayscale tones")
    if SERIES_STYLES["breakpoint"]["linestyle"] != "--":
        failures.append("breakpoints must use the approved dashed structural cue")
    if SERIES_STYLES["selected"]["linestyle"] != "--":
        failures.append("selected structural means must use the approved dashed cue")
    if SERIES_STYLES["selected"]["linewidth"] >= SERIES_STYLES["observed"]["linewidth"]:
        failures.append("selected structural means must be thinner than observed trajectories")
    if REFERENCE_STYLES["breakpoint"]["linewidth"] > 0.8:
        failures.append("breakpoint references must remain thin")
    if BREAKPOINT_DATE_FORMAT != "%Y-%m":
        failures.append("breakpoint date labels must use the numeric YYYY-MM format")
    if format_figure_note("Bands quantify uncertainty.") != (
        r"$\mathit{Note.}$ Bands quantify uncertainty."
    ):
        failures.append("figure notes must begin with an italicized Note. label")
    if CONSTRUCT_LABELS["q1_polarization"].startswith("Q1"):
        failures.append("visible construct labels must not expose internal Q1 codes")
    if UNCERTAINTY_STYLES["measurement"]["facecolor"] != COLORS["confidence"]:
        failures.append("measurement uncertainty must use the shared gray fill")
    for role in ("republican", "democrat", "q1", "q2", "q3", "q4", "party_brand_index"):
        source = SERIES_STYLES if role in SERIES_STYLES else DIMENSION_STYLES
        if not source[role].get("linestyle") or not source[role].get("marker"):
            failures.append(f"{role} requires non-color line and marker cues")
    required_families = {
        "time_series",
        "dual_axis_time_series",
        "party_comparison",
        "coefficient_forest",
        "bar_or_count",
        "distribution",
        "model_comparison",
        "diagnostic",
        "multi_panel",
    }
    if set(CHART_FAMILY_DEFAULTS) != required_families:
        failures.append("the reusable chart-family registry is incomplete")
    if CHART_FAMILY_DEFAULTS["time_series"]["month_tick_interval"] != 2:
        failures.append("full time-series figures must default to two-month ticks")
    return failures


apply_apa_style()
