"""Machine-generated manuscript and supplement table contracts for Paper 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.publication_contract import PublicationInputs


TABLE_SCHEMA = "paper3-manuscript-statistical-tables-v1"


def _fmt(value: object, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def _fmt_int(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(int(float(value)))


def _fmt_p(value: object) -> str:
    numeric = float(value)
    return f"{numeric:.6f}" if numeric >= 0.000001 else f"{numeric:.2e}"


def _fmt_ci(lower: object, upper: object) -> str:
    return f"[{_fmt(lower)}, {_fmt(upper)}]"


def _party_label(value: object) -> str:
    labels = {"republican": "Republican", "democrat": "Democratic"}
    text = str(value).strip().lower()
    return labels.get(text, str(value).title())


def _split_pipe_numbers(value: object) -> list[float]:
    text = str(value)
    if not text or text.lower() == "nan":
        return []
    return [float(item) for item in text.split("|") if item]


def _human_dates(value: object) -> str:
    text = str(value)
    if not text or text in {"nan", "none"}:
        return "None"
    return ", ".join(pd.Period(item, freq="M").strftime("%b %Y") for item in text.split("|"))


def _blueprint_table(
    label: str,
    title: str,
    columns: list[str],
    rows: Iterable[Iterable[object]],
    width_weights: list[float],
    notes: list[str],
    alignments: list[str] | None = None,
    font_size_points: float = 8.0,
) -> dict:
    if len(columns) != len(width_weights):
        raise ValueError(f"{label} width count does not match column count")
    if alignments is None:
        alignments = ["left"] + ["center"] * (len(columns) - 1)
    return {
        "label": label,
        "title": title,
        "columns": columns,
        "rows": [[str(value) for value in row] for row in rows],
        "width_weights": width_weights,
        "alignments": alignments,
        "notes": notes,
        "font_size_points": font_size_points,
    }


def build_table_contract(inputs: PublicationInputs) -> tuple[dict[str, pd.DataFrame], dict]:
    """Construct exact CSV tables and formatted DOCX blueprints."""

    frames = inputs.frames
    trends = frames["trends"].copy()
    contrast = frames["party_slope_contrast"].iloc[0]

    trend_rows = trends[
        [
            "series",
            "n_observations",
            "lag",
            "S",
            "variance_ordinary",
            "variance_hamed_rao",
            "hamed_rao_correction_factor",
            "Z",
            "tau",
            "p_value",
            "trend",
            "sen_slope",
            "sen_slope_ci_lower",
            "sen_slope_ci_upper",
            "sen_slope_unit",
            "period_start",
            "period_end",
        ]
    ].copy()
    trend_rows.insert(0, "row_role", "validated_series")
    contrast_row = pd.DataFrame(
        [
            {
                "row_role": "direct_paired_rq2_contrast",
                "series": "RQ2 paired Republican minus Democratic slope contrast",
                "n_observations": contrast["n_observations"],
                "lag": contrast["lag"],
                "S": contrast["S"],
                "variance_ordinary": contrast["variance_ordinary"],
                "variance_hamed_rao": contrast["variance_hamed_rao"],
                "hamed_rao_correction_factor": contrast["hamed_rao_correction_factor"],
                "Z": contrast["Z"],
                "tau": contrast["tau"],
                "p_value": contrast["p_value"],
                "trend": contrast["trend"],
                "sen_slope": contrast["paired_sen_slope_contrast"],
                "sen_slope_ci_lower": contrast["paired_sen_slope_ci_lower"],
                "sen_slope_ci_upper": contrast["paired_sen_slope_ci_upper"],
                "sen_slope_unit": contrast["unit"],
                "period_start": contrast["period_start"],
                "period_end": contrast["period_end"],
            }
        ]
    )
    table_1 = pd.concat([trend_rows, contrast_row], ignore_index=True)

    models = frames["breakpoint_models"].copy()
    regimes = frames["breakpoint_regimes"].copy()
    structural_rows = []
    for _, model in models.iterrows():
        model_regimes = regimes.loc[regimes["model_id"].eq(model["model_id"])].sort_values("regime")
        means = " | ".join(_fmt(value) for value in model_regimes["structural_regime_mean"])
        shifts = " | ".join(
            _fmt(value)
            for value in model_regimes["adjacent_structural_shift"].dropna().tolist()
        )
        structural_rows.append(
            {
                "series": model["series"],
                "reporting_role": model["reporting_role"],
                "model": model["candidate_label"],
                "n_observations": model["n_observations"],
                "n_breakpoints": model["n_breakpoints"],
                "ar_order": model["ar_order"],
                "break_onset_dates": model["break_onset_dates"],
                "joint_selection_bic": model["joint_selection_bic"],
                "delta_bic": model["delta_bic_within_reported_pair"],
                "structural_regime_means": means,
                "adjacent_structural_shifts": shifts,
                "ljung_box_p_lag8": model["ljung_box_p_lag8"],
                "selection_basis": model["selection_basis"],
            }
        )
    table_2 = pd.DataFrame(structural_rows)

    h3_models = frames["h3_models"].copy()
    h3_coefficients = frames["h3_coefficients"].copy()
    h3_contrasts = frames["h3_contrasts"].copy()
    dominance_observed = frames["dominance_observed"].copy()
    dominance_contrasts = frames["dominance_contrasts"].copy()
    dominance_sensitivity = frames["dominance_sensitivity"].copy()
    trend_break_models = frames["trend_break_models"].copy()
    trend_break_coefficients = frames["trend_break_coefficients"].copy()
    endpoint_contrasts = frames["endpoint_contrasts"].copy()
    s_decomposition = frames["s_decomposition"].copy()
    breakpoint_winners = frames["breakpoint_winners"].copy()
    breakpoint_regimes = regimes.copy()

    csv_tables = {
        "table_1_temporal_trends": table_1,
        "table_2_structural_models": table_2,
        "table_s1a_h3_model_comparison": h3_models,
        "table_s1b_h3_coefficients": h3_coefficients,
        "table_s1c_h3_party_contrasts": h3_contrasts,
        "table_s2a_dominance_observed": dominance_observed,
        "table_s2b_dominance_contrasts": dominance_contrasts,
        "table_s2c_dominance_block_sensitivity": dominance_sensitivity,
        "table_s3a_trend_break_model_comparison": trend_break_models,
        "table_s3b_trend_break_coefficients": trend_break_coefficients,
        "table_s3c_fitted_endpoint_contrasts": endpoint_contrasts,
        "table_s3d_mann_kendall_s_decomposition": s_decomposition,
        "table_s4a_breakpoint_duration_sensitivity": breakpoint_winners,
        "table_s4b_breakpoint_regimes": breakpoint_regimes,
    }

    table_1_rows = [
        [
            row["series"],
            _fmt_int(row["n_observations"]),
            _fmt_int(row["lag"]),
            _fmt_int(row["S"]),
            _fmt(row["variance_ordinary"]),
            _fmt(row["variance_hamed_rao"]),
            _fmt(row["hamed_rao_correction_factor"]),
            _fmt(row["Z"]),
            _fmt(row["tau"]),
            _fmt_p(row["p_value"]),
            _fmt(row["sen_slope"]),
            _fmt_ci(row["sen_slope_ci_lower"], row["sen_slope_ci_upper"]),
        ]
        for _, row in table_1.iterrows()
    ]
    table_2_rows = [
        [
            row["series"],
            row["model"],
            _fmt_int(row["ar_order"]),
            _fmt_int(row["n_breakpoints"]),
            _human_dates(row["break_onset_dates"]),
            _fmt(row["joint_selection_bic"]),
            _fmt(row["delta_bic"]),
            row["structural_regime_means"],
            row["adjacent_structural_shifts"] or "None",
            _fmt(row["ljung_box_p_lag8"]),
        ]
        for _, row in table_2.iterrows()
    ]

    main_blueprints = [
        _blueprint_table(
            "Table 1",
            "Temporal trend tests and the direct paired RQ2 slope contrast",
            [
                "Series or estimand",
                "n",
                "Lag",
                "S",
                "Var(S), ordinary",
                "Var(S), H-R",
                "H-R factor",
                "Z",
                "tau",
                "p",
                "Sen slope",
                "95% CI",
            ],
            table_1_rows,
            [24, 4, 4, 5, 9, 9, 7, 6, 6, 7, 9, 14],
            [
                "H-R denotes the Hamed-Rao variance correction. All tests are two-sided, all rows show an increasing direction, and all intervals are 95% rank-order Sen intervals.",
                "The first four rows use the validated 34-month series from March 2022 through December 2024. Q1 slopes use stance points per month; the Party Brand Index slope uses index points per month.",
                "The direct paired RQ2 row takes the median Republican-minus-Democratic difference over all 561 common month-pair slopes. It is algebraically identical to the Sen slope of the polarization series. The Republican and Democratic marginal slopes are 0.423731 and 0.167643; their descriptive difference is 0.256088 and is not the inferential contrast.",
            ],
            font_size_points=7.5,
        ),
        _blueprint_table(
            "Table 2",
            "Selected structural models and required alternatives",
            [
                "Series",
                "Model",
                "AR",
                "Breaks",
                "Estimated regime onsets",
                "BIC",
                "Delta BIC",
                "Structural means",
                "Adjacent shifts",
                "Ljung-Box p, lag 8",
            ],
            table_2_rows,
            [13, 18, 4, 5, 17, 8, 8, 16, 14, 10],
            [
                "BIC is the location-adjusted joint criterion. Delta BIC is calculated within each reported two-model pair.",
                "The selected polarization model contains three AR(2) regime onsets. The competitive four-break AR(2) alternative remains disclosed because Delta BIC equals 1.248392. The selected Party Brand Index model contains one September 2023 onset; its no-break AR(1) comparator has Delta BIC 9.348696.",
                "Breakpoint dates are estimated new-regime onsets. The table does not provide breakpoint-date or model-selection confidence intervals.",
            ],
            font_size_points=7.7,
        ),
    ]

    selected_h3 = h3_coefficients.loc[h3_coefficients["ar_order"].eq(0)].copy()
    h3_model_rows = [
        [
            _fmt_int(row["ar_order"]),
            _fmt(row["phi"]),
            _fmt(row["log_likelihood"]),
            _fmt(row["bic"]),
            _fmt(row["delta_bic"]),
            "Yes" if bool(row["selected"]) else "No",
        ]
        for _, row in h3_models.iterrows()
    ]
    h3_coefficient_rows = [
        [
            _fmt_int(row["ar_order"]),
            _party_label(row["party"]),
            str(row["term"]),
            _fmt(row["estimate"]),
            _fmt(row["standard_error_conditional"]),
            _fmt_p(row["p_two_sided_conditional"]),
            _fmt_ci(row["ci_lower_conditional"], row["ci_upper_conditional"]),
            _fmt(row["measurement_mean"]),
            _fmt_ci(row["measurement_ci_lower"], row["measurement_ci_upper"]),
        ]
        for _, row in h3_coefficients.iterrows()
    ]
    h3_contrast_rows = [
        [
            _fmt_int(row["ar_order"]),
            str(row["contrast"])
            .replace("republican_minus_democrat", "Republican minus Democratic")
            .replace("_", " "),
            _fmt(row["estimate"]),
            _fmt_p(row["p_two_sided_conditional"]),
            _fmt_ci(row["ci_lower_conditional"], row["ci_upper_conditional"]),
            _fmt(row["measurement_mean"]),
            _fmt_ci(row["measurement_ci_lower"], row["measurement_ci_upper"]),
        ]
        for _, row in h3_contrasts.iterrows()
    ]

    dominance_rows = [
        [
            str(row["predictor"]).upper(),
            _fmt(row["general_dominance_contribution"]),
            _fmt(100 * float(row["share_of_model_r_squared"]), 3),
            _fmt(row["correlation_with_q1"]),
            _fmt(row["full_model_r_squared"]),
        ]
        for _, row in dominance_observed.iterrows()
    ]
    dominance_contrast_rows = [
        [
            str(row["contrast"]).replace("_", " ").upper(),
            _fmt(row["observed_difference"]),
            _fmt_ci(row["simultaneous_ci_lower"], row["simultaneous_ci_upper"]),
            _fmt(row["probability_positive"]),
            str(row["interpretation_role"]),
        ]
        for _, row in dominance_contrasts.iterrows()
    ]
    dominance_sensitivity_display = dominance_sensitivity.loc[
        dominance_sensitivity["statistic"].isin(
            ["q4_contribution", "q4_minus_q2", "q4_minus_q3"]
        )
    ].copy()
    dominance_sensitivity_rows = [
        [
            _fmt_int(row["block_length"]),
            str(row["uncertainty_layer"]).replace("_", " ").title(),
            str(row["statistic"]).replace("_", " ").upper(),
            _fmt(row["observed_value"]),
            _fmt(row["bootstrap_mean"]),
            _fmt(row["bootstrap_standard_error"]),
            _fmt_ci(row["ci_lower_pointwise"], row["ci_upper_pointwise"]),
            _fmt(row["probability_positive"]),
        ]
        for _, row in dominance_sensitivity_display.iterrows()
    ]

    trend_break_rows = [
        [
            _fmt_int(row["minimum_regime_duration"]),
            str(row["model_class"]).replace("_", " ").title(),
            _fmt_int(row["ar_order"]),
            _fmt_int(row["n_breakpoints"]),
            _human_dates(row["break_onset_dates"]),
            _fmt(row["joint_bic"]),
            _fmt(row["delta_bic_within_duration"]),
            "Yes" if bool(row["selected_within_duration"]) else "No",
        ]
        for _, row in trend_break_models.iterrows()
    ]
    primary_time = trend_break_coefficients.loc[
        trend_break_coefficients["minimum_regime_duration"].eq(3)
        & trend_break_coefficients["model_class"].eq("break_plus_trend")
        & trend_break_coefficients["term"].eq("time")
    ].iloc[0]
    endpoint_rows = [
        [
            str(row["model_class"]).replace("_", " ").title(),
            _fmt(row["estimate_end_minus_start"]),
            _fmt(row["standard_error_conditional"]),
            _fmt_p(row["p_two_sided_conditional"]),
            _fmt_ci(row["ci_lower_conditional"], row["ci_upper_conditional"]),
            "No" if not bool(row["selection_uncertainty_included"]) else "Yes",
        ]
        for _, row in endpoint_contrasts.iterrows()
    ]
    s_rows = [
        [
            str(row["pair_category"]).replace("_", " ").title(),
            _fmt_int(row["positive"]),
            _fmt_int(row["negative"]),
            _fmt_int(row["ties"]),
            _fmt_int(row["pairs"]),
            _fmt_int(row["mann_kendall_s_contribution"]),
        ]
        for _, row in s_decomposition.iterrows()
    ]

    winner_rows = [
        [
            row["series"],
            _fmt_int(row["minseglen"]),
            _fmt_int(row["n_breakpoints"]),
            _fmt_int(row["ar_order"]),
            _human_dates(row["break_onset_dates"]),
            _fmt(row["ga_objective_bic"]),
            f"{_fmt_int(row['runs_reaching_best'])}/{_fmt_int(row['total_runs'])}",
            _fmt(row["best_recovery_rate"], 3),
        ]
        for _, row in breakpoint_winners.iterrows()
    ]
    regime_rows = [
        [
            row["series"],
            str(row["reporting_role"]).replace("_", " ").title(),
            _fmt_int(row["regime"]),
            str(row["start_date"]),
            str(row["end_date"]),
            _fmt_int(row["duration_months"]),
            _fmt(row["structural_regime_mean"]),
            _fmt(row["adjacent_structural_shift"]),
        ]
        for _, row in breakpoint_regimes.iterrows()
    ]

    supplementary_blueprints = [
        _blueprint_table(
            "Table S1A",
            "H3 observed-data error-structure comparison",
            ["AR order", "phi", "Log likelihood", "BIC", "Delta BIC", "Selected"],
            h3_model_rows,
            [10, 12, 18, 16, 16, 12],
            ["Observed-data BIC selects independent errors, AR(0)."],
            font_size_points=8.5,
        ),
        _blueprint_table(
            "Table S1B",
            "Complete party-aware H3 coefficient contract",
            [
                "AR",
                "Party",
                "Term",
                "Estimate",
                "Conditional SE",
                "Conditional p",
                "Conditional 95% CI",
                "Measurement mean",
                "Measurement 95% interval",
            ],
            h3_coefficient_rows,
            [5, 10, 12, 10, 12, 11, 16, 13, 18],
            [
                "The selected model uses AR(0). Measurement intervals propagate the joint article-cluster measurement stream. Conditional intervals do not include error-structure selection uncertainty.",
                f"The selected polarization coefficients are {_fmt(selected_h3.loc[selected_h3['party'].eq('republican') & selected_h3['term'].eq('polarization'), 'estimate'].iloc[0])} for Republicans and {_fmt(selected_h3.loc[selected_h3['party'].eq('democrat') & selected_h3['term'].eq('polarization'), 'estimate'].iloc[0])} for Democrats. H3 is not supported.",
            ],
            font_size_points=7.7,
        ),
        _blueprint_table(
            "Table S1C",
            "Party contrast in the H3 polarization coefficients",
            [
                "AR",
                "Contrast",
                "Estimate",
                "Conditional p",
                "Conditional 95% CI",
                "Measurement mean",
                "Measurement 95% interval",
            ],
            h3_contrast_rows,
            [6, 24, 11, 12, 17, 13, 18],
            ["Both conditional and measurement intervals include zero. The selected Republican sign also changes under AR(1)."],
            font_size_points=8.0,
        ),
        _blueprint_table(
            "Table S2A",
            "Observed cross-dimensional general dominance",
            ["Predictor", "Contribution", "% of model R-squared", "Correlation with Q1", "Full-model R-squared"],
            dominance_rows,
            [12, 18, 20, 20, 20],
            ["Q4 has the largest observed contribution. The ranking describes contemporaneous observed co-movement."],
            font_size_points=8.5,
        ),
        _blueprint_table(
            "Table S2B",
            "Dependence-preserving Q4 dominance contrasts",
            ["Contrast", "Observed difference", "Simultaneous interval", "Pr(positive)", "Interpretation role"],
            dominance_contrast_rows,
            [16, 16, 20, 14, 34],
            ["Both simultaneous intervals include zero. Do not infer confirmatory pairwise superiority, temporal precedence, or causation."],
            font_size_points=8.2,
        ),
        _blueprint_table(
            "Table S2C",
            "Stationary month-block sensitivity for Q4 contributions and contrasts",
            [
                "Block length",
                "Uncertainty layer",
                "Statistic",
                "Observed",
                "Bootstrap mean",
                "Bootstrap SE",
                "Pointwise 95% interval",
                "Pr(positive)",
            ],
            dominance_sensitivity_rows,
            [9, 18, 16, 10, 13, 12, 17, 11],
            [
                "Each fixed cell contains 5,000 stationary-bootstrap replicates. The complete machine-readable table also retains full-model R-squared, individual contributions, and correlations.",
                "These intervals provide dependence-preserving sensitivity only. The analysis does not claim nominal confirmatory coverage, pairwise superiority, temporal precedence, or causation.",
            ],
            font_size_points=7.6,
        ),
        _blueprint_table(
            "Table S3A",
            "Trend-only, break-only, and break-plus-trend comparison",
            ["Minimum duration", "Model class", "AR", "Breaks", "Estimated onsets", "BIC", "Delta BIC", "Selected"],
            trend_break_rows,
            [10, 17, 5, 7, 23, 12, 12, 9],
            ["Break-only has the lowest location-adjusted BIC under every examined minimum-duration definition."],
            font_size_points=8.0,
        ),
        _blueprint_table(
            "Table S3B",
            "Primary common within-regime slope",
            ["Model", "Estimate", "Conditional SE", "Conditional p", "Conditional 95% CI"],
            [["Break plus trend", _fmt(primary_time["estimate"]), _fmt(primary_time["standard_error_conditional"]), _fmt_p(primary_time["p_two_sided_conditional"]), _fmt_ci(primary_time["ci_lower_conditional"], primary_time["ci_upper_conditional"])]],
            [24, 15, 18, 16, 24],
            ["The interval includes zero. This establishes an absence of evidence for a common gradual within-regime slope, not proof that gradual change was absent."],
            font_size_points=8.5,
        ),
        _blueprint_table(
            "Table S3C",
            "Fitted end-to-start polarization contrasts",
            ["Model class", "Estimate", "Conditional SE", "Conditional p", "Conditional 95% CI", "Selection uncertainty included"],
            endpoint_rows,
            [20, 13, 15, 14, 20, 18],
            ["Intervals condition on the fitted structural model and do not provide post-selection coverage."],
            font_size_points=8.4,
        ),
        _blueprint_table(
            "Table S3D",
            "Exact Mann-Kendall S decomposition",
            ["Pair category", "Positive", "Negative", "Ties", "Pairs", "S contribution"],
            s_rows,
            [24, 13, 13, 10, 13, 17],
            ["The decomposition is descriptive arithmetic on the original series and is not a second significance test."],
            font_size_points=8.5,
        ),
        _blueprint_table(
            "Table S4A",
            "Joint common-AR breakpoint duration sensitivity and optimizer recovery",
            ["Series", "Minimum duration", "Breaks", "AR", "Estimated onsets", "BIC", "Best runs", "Recovery rate"],
            winner_rows,
            [15, 11, 7, 5, 26, 12, 10, 12],
            ["Optimizer starts assess numerical convergence. They are not bootstrap replicates and do not quantify breakpoint-date uncertainty."],
            font_size_points=7.9,
        ),
        _blueprint_table(
            "Table S4B",
            "Reported breakpoint regimes, structural means, and adjacent shifts",
            ["Series", "Role", "Regime", "Start", "End", "Months", "Structural mean", "Adjacent shift"],
            regime_rows,
            [15, 17, 8, 11, 11, 8, 15, 15],
            ["Adjacent shifts compare each regime with the immediately preceding structural mean."],
            font_size_points=7.9,
        ),
    ]

    blueprint = {
        "schema": TABLE_SCHEMA,
        "style_preset": "compact_reference_guide with named apa_manuscript_table_v1 override",
        "style_override": {
            "page": "US Letter landscape, 1 inch margins, 9 inch content width",
            "font": "Times New Roman",
            "table_body_size_points": "specified per table",
            "table_rule": "APA minimal horizontal rules without vertical grid lines",
            "cell_margins_dxa": {"top": 80, "bottom": 80, "start": 90, "end": 90},
        },
        "documents": {
            "main": {
                "filename": "paper3_main_statistical_tables.docx",
                "document_title": "Paper 3 Main Statistical Tables",
                "tables": main_blueprints,
            },
            "supplement": {
                "filename": "paper3_supplementary_statistical_tables.docx",
                "document_title": "Paper 3 Supplementary Statistical Tables",
                "tables": supplementary_blueprints,
            },
        },
    }
    return csv_tables, blueprint


def write_table_contract(
    inputs: PublicationInputs,
    output_directory: Path,
) -> tuple[list[Path], Path]:
    """Write full-precision CSV tables and the formatted document blueprint."""

    data_directory = output_directory
    data_directory.mkdir(parents=True, exist_ok=True)
    tables, blueprint = build_table_contract(inputs)
    outputs: list[Path] = []
    for stem, frame in tables.items():
        path = data_directory / f"{stem}.csv"
        frame.to_csv(path, index=False)
        outputs.append(path)
    blueprint_path = output_directory.parent / "table_blueprints.json"
    blueprint_path.write_text(json.dumps(blueprint, indent=2), encoding="utf-8")
    return outputs, blueprint_path
