"""Load Paper 3 publication inputs from the deposit data tree."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class PublicationInputs:
    repository_root: Path
    statistics_directory: Path
    aggregation_directory: Path
    frames: Dict[str, pd.DataFrame]
    metadata: Dict[str, dict]


def load_publication_inputs(data_root: Path) -> PublicationInputs:
    """Load the publication frames and analysis metadata."""

    data_root = data_root.resolve()
    statistics = data_root / "statistics"
    aggregation = data_root / "aggregated"
    trend_path = statistics / "trend_tests.csv"
    winner_path = statistics / "joint_ar_breakpoints" / "joint_ar_bic_winners.csv"
    contrast_path = statistics / "party_slope_contrast.csv"
    article_source = aggregation / "article_level_scores.csv"

    frame_paths = {
        "stance": statistics / "monthly_stance_joint_bootstrap.csv",
        "polarization": statistics / "monthly_polarization_joint_bootstrap.csv",
        "coherence": statistics / "monthly_message_coherence.csv",
        "party_brand_index": statistics / "party_brand_index_timeseries.csv",
        "trends": trend_path,
        "breakpoint_models": statistics / "breakpoint_models.csv",
        "breakpoint_fitted": statistics / "breakpoint_fitted_series.csv",
        "breakpoint_regimes": statistics / "breakpoint_regimes.csv",
        "h3_models": statistics / "h3_model_comparison.csv",
        "h3_coefficients": statistics / "h3_coefficients.csv",
        "h3_contrasts": statistics / "h3_party_contrast.csv",
        "dominance_observed": statistics / "dominance_observed.csv",
        "dominance_contrasts": statistics / "dominance_contrasts.csv",
        "dominance_sensitivity": statistics / "dominance_block_sensitivity.csv",
        "trend_break_models": statistics / "trend_break_model_comparison.csv",
        "trend_break_coefficients": statistics / "trend_break_coefficients.csv",
        "trend_break_residuals": statistics / "trend_break_residual_diagnostics.csv",
        "endpoint_contrasts": statistics / "fitted_endpoint_contrast.csv",
        "s_decomposition": statistics / "mann_kendall_s_decomposition.csv",
        "breakpoint_winners": winner_path,
        "party_slope_contrast": contrast_path,
        "article_scores": article_source,
    }
    frames = {name: pd.read_csv(path) for name, path in frame_paths.items()}
    metadata_paths = {
        "measurement_receipt": statistics / "measurement_bootstrap_metadata.json",
        "breakpoint_receipt": statistics / "breakpoint_analysis_metadata.json",
        "h3_receipt": statistics / "h3_analysis_metadata.json",
        "dominance_receipt": statistics / "dominance_analysis_metadata.json",
        "trend_break_receipt": statistics / "trend_break_analysis_metadata.json",
    }
    metadata = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in metadata_paths.items()
    }
    return PublicationInputs(
        repository_root=data_root.parent,
        statistics_directory=statistics,
        aggregation_directory=aggregation,
        frames=frames,
        metadata=metadata,
    )
