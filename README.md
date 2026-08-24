# Paper 3 Ukraine Aid Reproduction

This repository contains the analysis pipeline and derived data for a study of how United States news coverage represented Democratic and Republican positions on Ukraine aid between March 2022 and December 2024.

## What is not included

The Factiva source export, the deduplicated corpus, the extracted checkpoint, and the masked processed text are not redistributable. This repository therefore begins from the scored sentence file. The exact Factiva query is provided so the corpus can be reconstructed.

## Corpus reconstruction

The Factiva retrieval window was 1 March 2022 to 31 December 2024. The expected result is 24,721 records before deduplication and 23,630 after deduplication.

```text
atleast5(ukrain*) and ((fmt=article or fmt=webpage or fmt=blog) not fmt=picture) and (rst=usa) and (rst=en) not (rst=freg or rst=prnasi or rst=prn or rst=twir or rst=tlgl or rst=wc57855 or rst=congdp or rst=uspeo or rst=indfed or rst=usgnd) not (ns=mcat or ns=ccat or ns=redit or ns=gspo or ns=nabst or ns=nadvtr or ns=ncal or ns=ncopro or ns=ncdig or ns=nfact or ns=nlist or ns=nhoc or ns=nimage or ns=nlet or ns=nobt or ns=npeo or ns=npan or ns=ntra or ns=nrmf or ns=nrvw or ns=nran or ns=nprpct) and wc>199 and (congress* or "senate" or "sen." or "gov." or "governor" or republican* or democrat* or "biden" or "trump" or "rep." or "gop" or representative* or "pelosi" or "vance" or "rand paul" or "kirk" or "carlson" or "secretary" or "white house" or "speaker" or "legislator" or "lawmaker" or "capitol hill" or "majority leader" or "minority leader") NOT hl=("live updates" or "live blog" or "ukraine live" or "live coverage" or "daily briefing")
```

The pipeline consumes these Factiva export columns: `排序号`, `文档id`, `出版社`, `标题`, `日期`, `时间`, `作者`, `字数`, `语言`, `公司`, `行业`, `主题`, `地区`, `所在版面`, `ART`, `摘要`, and `全文_1` through `全文_13`.

## Pipeline stages

1. Preprocessing: `python code/reproduce_preprocessing.py --input <export> --output-dir <dir>`. The expected output is exploded sentence rows with `cleaned_sentence` masked.
2. Annotation: `code/src/llm_scorer.py` submits sentences to the OpenAI Batch API and requires `OPENAI_API_KEY` in the environment. The study used model `gpt-5.1-2025-11-13` at temperature 0 with a 100-token cap and JSON object response format. The expected output is 37,312 scored sentences.
3. Aggregation and statistics: `python code/main.py --downstream --data-root <path>`.
4. Figures and tables: `python code/scripts/build_publication_artifacts.py` and `python code/scripts/make_series_autocorrelation_figure.py`.

## Data dictionary

- `data/aggregated/article_level_scores.csv` — 14,564 rows — article-level party, weight, q1–q4 scores, and counts.
- `data/aggregated/monthly_polarization.csv` — 136 rows — monthly party means, polarization, and party article counts by dimension.
- `data/aggregated/monthly_stance.csv` — 272 rows — monthly party stance, raw IQR, and article counts by dimension.
- `data/aggregated/sentence_id_crosswalk.csv` — 37,312 rows — LLM row numbers, ID schemas, stable sentence and article IDs, and party.
- `data/politicians/legislators-current.csv` — 539 rows — current legislator names, party, office, contact fields, and identifiers.
- `data/politicians/legislators-historical.csv` — 12,222 rows — historical legislator names, party, office, contact fields, and identifiers.
- `data/scored/factiva_llmrated_scores.csv` — 37,312 rows — stable IDs, title and body key, sentence metadata, article weights, q1–q4 scores, error type, and batch ID.
- `data/search_string.txt` — 3 lines — the Factiva query and retrieval note.
- `data/statistics/STATISTICAL_TESTS_README.md` — 106 lines — the statistical analysis contract.
- `data/statistics/acf_pacf_q1.png` — N/A rows; 2,970 × 2,365 pixels — Q1 autocorrelation and partial-autocorrelation plot.
- `data/statistics/all_dimensions_polarization_trend.png` — N/A rows; 4,203 × 2,294 pixels — polarization trends for all four dimensions.
- `data/statistics/ar2_residual_acf_pacf_q1.png` — N/A rows; 2,970 × 2,365 pixels — Q1 AR(2) residual diagnostics.
- `data/statistics/articles_with_polarization_q1.png` — N/A rows; 4,224 × 2,294 pixels — Q1 article counts and polarization.
- `data/statistics/breakpoint_analysis_metadata.json` — N/A rows; 34 top-level fields — breakpoint-analysis parameters, counts, and diagnostics.
- `data/statistics/breakpoint_fitted_series.csv` — 136 rows — observed values, regime means, AR fits, innovations, and standardized innovations.
- `data/statistics/breakpoint_models.csv` — 4 rows — breakpoint model specifications, fit criteria, diagnostics, and coefficients.
- `data/statistics/breakpoint_regimes.csv` — 12 rows — breakpoint regime dates, durations, means, and adjacent shifts.
- `data/statistics/data/table_1_temporal_trends.csv` — 5 rows — temporal trend tests and the direct paired slope contrast.
- `data/statistics/data/table_2_structural_models.csv` — 4 rows — selected structural models and required alternatives.
- `data/statistics/data/table_s1a_h3_model_comparison.csv` — 2 rows — H3 error-structure model comparison.
- `data/statistics/data/table_s1b_h3_coefficients.csv` — 12 rows — H3 coefficient estimates and intervals.
- `data/statistics/data/table_s1c_h3_party_contrasts.csv` — 2 rows — H3 party contrasts.
- `data/statistics/data/table_s2a_dominance_observed.csv` — 3 rows — observed dominance contributions and correlations.
- `data/statistics/data/table_s2b_dominance_contrasts.csv` — 2 rows — dominance contrasts and intervals.
- `data/statistics/data/table_s2c_dominance_block_sensitivity.csv` — 54 rows — dominance estimates across block lengths.
- `data/statistics/data/table_s3a_trend_break_model_comparison.csv` — 12 rows — trend and breakpoint model comparisons.
- `data/statistics/data/table_s3b_trend_break_coefficients.csv` — 44 rows — trend-break model coefficients and intervals.
- `data/statistics/data/table_s3c_fitted_endpoint_contrasts.csv` — 2 rows — fitted end-to-start polarization contrasts.
- `data/statistics/data/table_s3d_mann_kendall_s_decomposition.csv` — 3 rows — exact Mann–Kendall S decomposition.
- `data/statistics/data/table_s4a_breakpoint_duration_sensitivity.csv` — 8 rows — breakpoint-duration sensitivity and optimizer recovery.
- `data/statistics/data/table_s4b_breakpoint_regimes.csv` — 12 rows — reported breakpoint regimes, means, and shifts.
- `data/statistics/dominance_analysis_metadata.json` — N/A rows; 24 top-level fields — dominance-analysis parameters, counts, and diagnostics.
- `data/statistics/dominance_block_sensitivity.csv` — 54 rows — block-length sensitivity estimates and intervals.
- `data/statistics/dominance_bootstrap_replicates.csv` — 10,000 rows — dominance bootstrap contributions, contrasts, and correlations.
- `data/statistics/dominance_bootstrap_stability.csv` — 5 rows — production, independent, and prefix interval comparisons.
- `data/statistics/dominance_contrasts.csv` — 2 rows — observed and bootstrap dominance contrasts.
- `data/statistics/dominance_observed.csv` — 3 rows — observed dominance contributions and correlations.
- `data/statistics/dominance_ready.csv` — 34 rows — monthly q1–q4 polarization series.
- `data/statistics/fitted_endpoint_contrast.csv` — 2 rows — fitted endpoint contrasts and conditional intervals.
- `data/statistics/h3_analysis_metadata.json` — N/A rows; 22 top-level fields — H3 analysis parameters, counts, and diagnostics.
- `data/statistics/h3_bootstrap_stability.csv` — 6 rows — H3 bootstrap interval and probability stability.
- `data/statistics/h3_coefficients.csv` — 12 rows — H3 coefficients and conditional and measurement intervals.
- `data/statistics/h3_influence_diagnostics.csv` — 34 rows — leave-one-month-out H3 influence diagnostics.
- `data/statistics/h3_measurement_replicate_coefficients.csv` — 9,978 rows — H3 measurement-replicate coefficients.
- `data/statistics/h3_model_comparison.csv` — 2 rows — H3 AR-order model comparison.
- `data/statistics/h3_party_contrast.csv` — 2 rows — H3 party-effect contrasts.
- `data/statistics/h3_residual_diagnostics.csv` — 8 rows — H3 residual autocorrelation and Ljung–Box diagnostics.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_fitted_series.csv` — 272 rows — joint AR-breakpoint fitted series.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_focused_candidates.csv` — 5 rows — focused joint AR-breakpoint candidates.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_focused_fitted_series.csv` — 170 rows — focused-candidate fitted series.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_metadata.json` — N/A rows; 26 top-level fields — joint AR-breakpoint parameters, counts, and diagnostics.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_model_diagnostics.csv` — 8 rows — joint AR-breakpoint model diagnostics.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_party_brand_index_diagnostics.png` — N/A rows; 5,821 × 3,352 pixels — Party Brand Index breakpoint diagnostics.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_party_brand_index_focused.png` — N/A rows; 3,392 × 3,352 pixels — focused Party Brand Index breakpoint comparison.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_polarization_diagnostics.png` — N/A rows; 5,821 × 3,352 pixels — polarization breakpoint diagnostics.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_polarization_focused.png` — N/A rows; 4,982 × 3,352 pixels — focused polarization breakpoint comparison.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_regime_statistics.csv` — 24 rows — joint AR-breakpoint regime statistics.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_search_convergence.csv` — 20 rows — breakpoint-search convergence frequencies.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_search_runs.csv` — 80 rows — breakpoint-search runs, seeds, configurations, and fit values.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_stability.png` — N/A rows; 4,954 × 2,680 pixels — breakpoint-search stability.
- `data/statistics/joint_ar_breakpoints/joint_ar_bic_winners.csv` — 8 rows — winning breakpoint configurations and recovery rates.
- `data/statistics/mann_kendall_s_decomposition.csv` — 3 rows — positive, negative, tie, and S contributions.
- `data/statistics/measurement_bootstrap_metadata.json` — N/A rows; 17 top-level fields — measurement-bootstrap parameters, counts, and diagnostics.
- `data/statistics/monthly_message_coherence.csv` — 68 rows — monthly party coherence, intervals, counts, and valid replicates.
- `data/statistics/monthly_polarization_joint_bootstrap.csv` — 136 rows — monthly polarization and joint-bootstrap intervals.
- `data/statistics/monthly_stance_joint_bootstrap.csv` — 272 rows — monthly stance and raw-IQR joint-bootstrap intervals.
- `data/statistics/party_brand_index_metadata.json` — N/A rows; 21 top-level fields — Party Brand Index parameters, counts, and diagnostics.
- `data/statistics/party_brand_index_timeseries.csv` — 34 rows — monthly polarization and coherence components, Party Brand Index, intervals, and counts.
- `data/statistics/party_brand_index_trend.csv` — 1 row — Party Brand Index Mann–Kendall and Sen-slope trend result.
- `data/statistics/party_brand_index_trend_metadata.json` — N/A rows; 8 top-level fields — Party Brand Index trend parameters and diagnostics.
- `data/statistics/party_slope_contrast.csv` — 1 row — paired party-slope contrast and marginal Sen slopes.
- `data/statistics/party_slope_contrast_metadata.json` — N/A rows; 9 top-level fields — party-slope contrast parameters, counts, and diagnostics.
- `data/statistics/q1_stance_mmk_trend.png` — N/A rows; 4,233 × 2,295 pixels — Q1 stance trend.
- `data/statistics/statistical_tests_report.txt` — 7 lines — statistical-test report.
- `data/statistics/table_blueprints.json` — N/A rows; 4 top-level fields — document-builder table specifications.
- `data/statistics/trend_break_analysis_metadata.json` — N/A rows; 19 top-level fields — trend-break analysis parameters, counts, and diagnostics.
- `data/statistics/trend_break_coefficients.csv` — 44 rows — trend-break model coefficients and conditional intervals.
- `data/statistics/trend_break_comparison/trend_break_search_runs.csv` — 40 rows — trend-break search runs and fit values.
- `data/statistics/trend_break_comparison/trend_break_search_winners.csv` — 4 rows — winning trend-break configurations.
- `data/statistics/trend_break_fitted_series.csv` — 102 rows — trend-break observed, structural, fitted, and innovation series.
- `data/statistics/trend_break_model_comparison.csv` — 12 rows — trend-break model comparisons by minimum regime duration.
- `data/statistics/trend_break_residual_diagnostics.csv` — 6 rows — trend-break Ljung–Box diagnostics.
- `data/statistics/trend_tests.csv` — 4 rows — dimension-specific Mann–Kendall and Sen-slope trend results.
- `data/validation/dimension_correlations.csv` — 6 rows — pairwise dimension correlations and p-values.
- `data/validation/manual_coding_BLIND.csv` — 50 rows — blind coding worksheet with two coders' q1–q4 fields.
- `data/validation/manual_coding_BLIND_Haoran.csv` — 50 rows — Haoran blind coding worksheet.
- `data/validation/manual_coding_BLIND_wendy_v2.csv` — 50 rows — Wendy blind coding worksheet.
- `data/validation/manual_coding_FULL.csv` — 50 rows — full coding sample with party, date, LLM scores, and error type.
- `data/validation/manual_coding_instruction.txt` — 25 lines — manual-coding instructions.
- `data/validation/test6_manual_report.txt` — 103 lines — test-6 manual-validation report.
- `data/validation/test6_manual_results.csv` — 50 rows — parsed coder and LLM q1–q4 results with party and date.
- `data/validation/test_retest_raw.csv` — 171 rows — repeated sentence-dimension scores.
- `data/validation/validation_report.txt` — 144 lines — validation report.
- `data/validation/validation_report_withoutmanual.txt` — 144 lines — validation report excluding manual results.

## Manuscript

The manuscript is provided as `manuscript/Paper3_Preprint_0820.docx`, `manuscript/Paper3_Preprint_0820.pdf`, and the LaTeX package in `manuscript/latex/`. The three figures in `manuscript/figures/` are the figures used by the manuscript.

## Software versions

| Package | Version |
| --- | --- |
| Python | 3.11.9 |
| pip | 25.3 |
| setuptools | 65.5.0 |
| R | 4.4.3 |
| r-boot | 1.3_32 |
| r-cli | 3.6.6 |
| r-farver | 2.1.2 |
| r-ggplot2 | 4.0.3 |
| r-glue | 1.8.1 |
| r-gtable | 0.3.6 |
| r-isoband | 0.3.0 |
| r-labeling | 0.4.3 |
| r-lifecycle | 1.0.5 |
| r-r6 | 2.6.1 |
| r-rcolorbrewer | 1.1_3 |
| r-rlang | 1.2.0 |
| r-s7 | 0.2.2 |
| r-scales | 1.4.0 |
| r-vctrs | 0.7.3 |
| r-viridislite | 0.4.3 |
| r-withr | 3.0.3 |
| rpy2 | 3.6.4 |

## Citation

```text
[Zenodo DOI to be added on deposit]
```

## Licence

See [LICENSE](LICENSE).
