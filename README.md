# Branding At The Water's Edge Repo

Analysis pipeline and derived data for a study of how United States news coverage represented Democratic and Republican positions on Ukraine aid between March 2022 and December 2024.

**Contents**

- [Scope](#scope)
- [Repository layout](#repository-layout)
- [Corpus reconstruction](#corpus-reconstruction)
- [Pipeline stages](#pipeline-stages)
- [Data dictionary](#data-dictionary)
- [Manuscript](#manuscript)
- [Software versions](#software-versions)
- [Citation](#citation)
- [Licence](#licence)

---

## Scope

### What is included

The complete analysis pipeline, the scored sentence file, every derived dataset, the validation evidence, and the manuscript.

### What is not included

The Factiva source export, the deduplicated corpus, the extracted checkpoint, and the masked processed text are not redistributable. This repository therefore begins from the scored sentence file. The exact Factiva query is provided so the corpus can be reconstructed.

---

## Repository layout

```
.
├── code/
│   ├── main.py                        downstream pipeline entry point
│   ├── reproduce_preprocessing.py     rerun preprocessing on a new export
│   ├── src/                           pipeline modules
│   ├── scripts/                       analysis and publication scripts
│   └── styles/                        matplotlib style
├── data/
│   ├── search_string.txt              Factiva query
│   ├── politicians/                   legislator name lists
│   ├── scored/                        annotated sentences
│   ├── aggregated/                    article and monthly series
│   ├── statistics/                    analysis outputs, tables, diagnostics
│   └── validation/                    human coding and agreement evidence
└── manuscript/
    ├── Paper3_Preprint_0820.docx
    ├── Paper3_Preprint_0820.pdf
    ├── latex/                         LaTeX source of the deposited PDF
    └── figures/                       regenerated figures
```

---

## Corpus reconstruction

Retrieval window: 1 March 2022 to 31 December 2024. Expected result: 24,721 records before deduplication, 23,630 after.

### Query

```text
atleast5(ukrain*) and ((fmt=article or fmt=webpage or fmt=blog) not fmt=picture) and (rst=usa) and (rst=en) not (rst=freg or rst=prnasi or rst=prn or rst=twir or rst=tlgl or rst=wc57855 or rst=congdp or rst=uspeo or rst=indfed or rst=usgnd) not (ns=mcat or ns=ccat or ns=redit or ns=gspo or ns=nabst or ns=nadvtr or ns=ncal or ns=ncopro or ns=ncdig or ns=nfact or ns=nlist or ns=nhoc or ns=nimage or ns=nlet or ns=nobt or ns=npeo or ns=npan or ns=ntra or ns=nrmf or ns=nrvw or ns=nran or ns=nprpct) and wc>199 and (congress* or "senate" or "sen." or "gov." or "governor" or republican* or democrat* or "biden" or "trump" or "rep." or "gop" or representative* or "pelosi" or "vance" or "rand paul" or "kirk" or "carlson" or "secretary" or "white house" or "speaker" or "legislator" or "lawmaker" or "capitol hill" or "majority leader" or "minority leader") NOT hl=("live updates" or "live blog" or "ukraine live" or "live coverage" or "daily briefing")
```

### Export columns

The export was taken from a Chinese-language Factiva interface, so column headers arrive in Chinese. `code/src/data_loader.py` renames them as below. An export with English headers can be used by adjusting that mapping.

| Factiva header | Renamed to | Used |
| --- | --- | --- |
| `排序号` | `index` | dropped |
| `文档id` | `doc_id` | dropped |
| `出版社` | `publisher` | **kept** |
| `标题` | `title` | **kept** |
| `日期` | `date` | **kept** |
| `时间` | `time` | dropped |
| `作者` | `author` | dropped |
| `字数` | `word_count` | **kept** |
| `语言` | `language` | dropped |
| `公司` | `company` | dropped |
| `行业` | `industry` | dropped |
| `主题` | `subject` | dropped |
| `地区` | `region` | dropped |
| `所在版面` | `page` | dropped |
| `ART` | `ART` | dropped |
| `摘要` | `abstract` | dropped |
| `全文_1` … `全文_13` | `body_1` … `body_13` | **kept**, concatenated into `body` |

Four columns and the thirteen body segments are retained. The body segments are concatenated into a single `body` field before sentence segmentation.

---

## Pipeline stages

### 1. Preprocessing

```bash
python code/reproduce_preprocessing.py --input <export> --output-dir <dir>
```

Segments articles into sentences, retains Ukraine-relevant sentences attributable to one party, and masks party and politician references. Output: exploded sentence rows with `cleaned_sentence` masked.

### 2. Annotation

`code/src/llm_scorer.py` submits sentences to the OpenAI Batch API and requires `OPENAI_API_KEY` in the environment. The study used model `gpt-5.1-2025-11-13` at temperature 0, a 100-token cap, and JSON object response format. Output: 37,312 annotated sentences.

### 3. Aggregation and statistics

```bash
python code/main.py --downstream --data-root <path>
```

Aggregates sentences to articles and months, computes polarization, within-party coherence and the Party Brand Index, and runs the temporal and structural models.

### 4. Figures and tables

```bash
python code/scripts/build_publication_artifacts.py
python code/scripts/make_series_autocorrelation_figure.py
```

---

## Data dictionary

### Query and reference lists

| File | Size | Contents |
| --- | --- | --- |
| `data/search_string.txt` | 3 lines | Factiva query and retrieval note |
| `data/politicians/legislators-current.csv` | 539 rows | current legislator names, party, office, contact fields, identifiers |
| `data/politicians/legislators-historical.csv` | 12,222 rows | historical legislator names, party, office, contact fields, identifiers |

### Scored sentences

| File | Size | Contents |
| --- | --- | --- |
| `data/scored/factiva_llmrated_scores.csv` | 37,312 rows | stable identifiers, title and body key, sentence metadata, article weights, scores for General Ukraine Aid Stance (q1), Ukraine Characterization (q2), Threat Priority (q3) and Resource Allocation (q4), error type, batch identifier |

### Aggregated series

| File | Size | Contents |
| --- | --- | --- |
| `data/aggregated/article_level_scores.csv` | 14,564 rows | article-level party, weight, dimension scores, counts |
| `data/aggregated/monthly_stance.csv` | 272 rows | monthly party stance, raw IQR, article counts by dimension |
| `data/aggregated/monthly_polarization.csv` | 136 rows | monthly party means, polarization, party article counts by dimension |
| `data/aggregated/sentence_id_crosswalk.csv` | 37,312 rows | LLM row numbers, identifier schemas, stable sentence and article identifiers, party |

### Statistics: headline results

| File | Size | Contents |
| --- | --- | --- |
| `data/statistics/trend_tests.csv` | 4 rows | dimension-specific Mann–Kendall and Sen-slope trend results |
| `data/statistics/mann_kendall_s_decomposition.csv` | 3 rows | positive, negative, tie and S contributions |
| `data/statistics/fitted_endpoint_contrast.csv` | 2 rows | fitted endpoint contrasts and conditional intervals |
| `data/statistics/party_slope_contrast.csv` | 1 row | paired party-slope contrast and marginal Sen slopes |
| `data/statistics/party_brand_index_timeseries.csv` | 34 rows | monthly polarization and coherence components, Party Brand Index, intervals, counts |
| `data/statistics/party_brand_index_trend.csv` | 1 row | Party Brand Index Mann–Kendall and Sen-slope trend result |
| `data/statistics/monthly_message_coherence.csv` | 68 rows | monthly within-party coherence, intervals, counts, valid replicates |
| `data/statistics/statistical_tests_report.txt` | 7 lines | statistical-test report |
| `data/statistics/STATISTICAL_TESTS_README.md` | 106 lines | statistical analysis specification |

### Statistics: structural models

| File | Size | Contents |
| --- | --- | --- |
| `data/statistics/breakpoint_models.csv` | 4 rows | shift model specifications, fit criteria, diagnostics, coefficients |
| `data/statistics/breakpoint_regimes.csv` | 12 rows | regime onsets, durations, means, adjacent shifts |
| `data/statistics/breakpoint_fitted_series.csv` | 136 rows | observed values, regime means, AR fits, innovations, standardized innovations |
| `data/statistics/trend_break_model_comparison.csv` | 12 rows | trend and shift model comparisons by minimum regime duration |
| `data/statistics/trend_break_coefficients.csv` | 44 rows | trend-shift model coefficients and conditional intervals |
| `data/statistics/trend_break_fitted_series.csv` | 102 rows | trend-shift observed, structural, fitted and innovation series |
| `data/statistics/trend_break_residual_diagnostics.csv` | 6 rows | trend-shift Ljung–Box diagnostics |

### Statistics: within-party coherence model

| File | Size | Contents |
| --- | --- | --- |
| `data/statistics/h3_model_comparison.csv` | 2 rows | error-structure model comparison |
| `data/statistics/h3_coefficients.csv` | 12 rows | coefficients with conditional and measurement intervals |
| `data/statistics/h3_party_contrast.csv` | 2 rows | party-effect contrasts |
| `data/statistics/h3_residual_diagnostics.csv` | 8 rows | residual autocorrelation and Ljung–Box diagnostics |
| `data/statistics/h3_influence_diagnostics.csv` | 34 rows | leave-one-month-out influence diagnostics |
| `data/statistics/h3_bootstrap_stability.csv` | 6 rows | bootstrap interval and probability stability |
| `data/statistics/h3_measurement_replicate_coefficients.csv` | 9,978 rows | measurement-replicate coefficients |

### Statistics: cross-dimensional dominance

| File | Size | Contents |
| --- | --- | --- |
| `data/statistics/dominance_ready.csv` | 34 rows | monthly polarization series for all four dimensions |
| `data/statistics/dominance_observed.csv` | 3 rows | observed contributions and correlations |
| `data/statistics/dominance_contrasts.csv` | 2 rows | observed and bootstrap contrasts |
| `data/statistics/dominance_block_sensitivity.csv` | 54 rows | block-length sensitivity estimates and intervals |
| `data/statistics/dominance_bootstrap_replicates.csv` | 10,000 rows | bootstrap contributions, contrasts, correlations |
| `data/statistics/dominance_bootstrap_stability.csv` | 5 rows | production, independent and prefix interval comparisons |

### Statistics: measurement uncertainty

| File | Size | Contents |
| --- | --- | --- |
| `data/statistics/monthly_stance_joint_bootstrap.csv` | 272 rows | monthly stance and raw-IQR article-cluster bootstrap intervals |
| `data/statistics/monthly_polarization_joint_bootstrap.csv` | 136 rows | monthly polarization and article-cluster bootstrap intervals |
| `data/statistics/series_autocorrelation.csv` | 12 rows | autocorrelation and partial-autocorrelation diagnostics |

### Statistics: run metadata

| File | Fields | Contents |
| --- | --- | --- |
| `data/statistics/measurement_bootstrap_metadata.json` | 17 | article-cluster bootstrap parameters, counts, diagnostics |
| `data/statistics/party_brand_index_metadata.json` | 21 | Party Brand Index parameters, counts, diagnostics |
| `data/statistics/party_brand_index_trend_metadata.json` | 8 | Party Brand Index trend parameters and diagnostics |
| `data/statistics/party_slope_contrast_metadata.json` | 9 | party-slope contrast parameters, counts, diagnostics |
| `data/statistics/breakpoint_analysis_metadata.json` | 34 | shift-analysis parameters, counts, diagnostics |
| `data/statistics/trend_break_analysis_metadata.json` | 19 | trend-shift analysis parameters, counts, diagnostics |
| `data/statistics/h3_analysis_metadata.json` | 22 | coherence-model parameters, counts, diagnostics |
| `data/statistics/dominance_analysis_metadata.json` | 24 | dominance parameters, counts, diagnostics |
| `data/statistics/table_blueprints.json` | 4 | document-builder table specifications |

### Statistics: shift search

| File | Size | Contents |
| --- | --- | --- |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_winners.csv` | 8 rows | winning configurations and recovery rates |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_search_runs.csv` | 80 rows | search runs, seeds, configurations, fit values |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_search_convergence.csv` | 20 rows | search convergence frequencies |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_model_diagnostics.csv` | 8 rows | model diagnostics |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_regime_statistics.csv` | 24 rows | regime statistics |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_fitted_series.csv` | 272 rows | fitted series |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_focused_candidates.csv` | 5 rows | focused candidates |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_focused_fitted_series.csv` | 170 rows | focused-candidate fitted series |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_metadata.json` | 26 fields | search parameters, counts, diagnostics |
| `data/statistics/trend_break_comparison/trend_break_search_runs.csv` | 40 rows | trend-shift search runs and fit values |
| `data/statistics/trend_break_comparison/trend_break_search_winners.csv` | 4 rows | winning trend-shift configurations |

### Statistics: manuscript tables

All under `data/statistics/tables/`.

| File | Size | Contents |
| --- | --- | --- |
| `table_1_temporal_trends.csv` | 5 rows | temporal trend tests and the direct paired slope contrast |
| `table_2_structural_models.csv` | 4 rows | selected structural models and required alternatives |
| `table_s1a_h3_model_comparison.csv` | 2 rows | coherence error-structure comparison |
| `table_s1b_h3_coefficients.csv` | 12 rows | coherence coefficient estimates and intervals |
| `table_s1c_h3_party_contrasts.csv` | 2 rows | coherence party contrasts |
| `table_s2a_dominance_observed.csv` | 3 rows | observed dominance contributions and correlations |
| `table_s2b_dominance_contrasts.csv` | 2 rows | dominance contrasts and intervals |
| `table_s2c_dominance_block_sensitivity.csv` | 54 rows | dominance estimates across block lengths |
| `table_s3a_trend_break_model_comparison.csv` | 12 rows | trend and shift model comparisons |
| `table_s3b_trend_break_coefficients.csv` | 44 rows | trend-shift model coefficients and intervals |
| `table_s3c_fitted_endpoint_contrasts.csv` | 2 rows | fitted end-to-start polarization contrasts |
| `table_s3d_mann_kendall_s_decomposition.csv` | 3 rows | exact Mann–Kendall S decomposition |
| `table_s4a_breakpoint_duration_sensitivity.csv` | 8 rows | minimum-regime-duration sensitivity and optimizer recovery |
| `table_s4b_breakpoint_regimes.csv` | 12 rows | reported regimes, means, shifts |

### Statistics: diagnostic plots

| File | Dimensions | Contents |
| --- | --- | --- |
| `data/statistics/acf_pacf_q1.png` | 2,970 × 2,365 | General Ukraine Aid Stance autocorrelation and partial autocorrelation |
| `data/statistics/ar2_residual_acf_pacf_q1.png` | 2,970 × 2,365 | General Ukraine Aid Stance AR(2) residual diagnostics |
| `data/statistics/all_dimensions_polarization_trend.png` | 4,203 × 2,294 | polarization trends for all four dimensions |
| `data/statistics/articles_with_polarization_q1.png` | 4,224 × 2,294 | General Ukraine Aid Stance article counts and polarization |
| `data/statistics/q1_stance_mmk_trend.png` | 4,233 × 2,295 | General Ukraine Aid Stance trend |
| `data/statistics/temporal_diagnostics/q1_polarization_measurement_and_breakpoint.png` | 2,304 × 1,473 | polarization measurement and shift diagnostic |
| `data/statistics/temporal_diagnostics/party_brand_index_measurement_and_breakpoint.png` | 2,304 × 1,473 | Party Brand Index measurement and shift diagnostic |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_polarization_diagnostics.png` | 5,821 × 3,352 | polarization shift diagnostics |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_party_brand_index_diagnostics.png` | 5,821 × 3,352 | Party Brand Index shift diagnostics |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_polarization_focused.png` | 4,982 × 3,352 | focused polarization shift comparison |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_party_brand_index_focused.png` | 3,392 × 3,352 | focused Party Brand Index shift comparison |
| `data/statistics/joint_ar_breakpoints/joint_ar_bic_stability.png` | 4,954 × 2,680 | shift-search stability |

### Validation

| File | Size | Contents |
| --- | --- | --- |
| `data/validation/manual_coding_instruction.txt` | 25 lines | manual-coding instructions |
| `data/validation/manual_coding_BLIND.csv` | 50 rows | blind coding worksheet with fields for both coders |
| `data/validation/manual_coding_BLIND_Haoran.csv` | 50 rows | first coder returns |
| `data/validation/manual_coding_BLIND_wendy_v2.csv` | 50 rows | second coder returns |
| `data/validation/manual_coding_FULL.csv` | 50 rows | merged sample with party, date, LLM scores, error type |
| `data/validation/test_retest_raw.csv` | 171 rows | repeated sentence-dimension scores |
| `data/validation/test6_manual_results.csv` | 50 rows | parsed coder and LLM results with party and date |
| `data/validation/test6_manual_report.txt` | 103 lines | manual-validation report |
| `data/validation/dimension_correlations.csv` | 6 rows | pairwise dimension correlations and p-values |
| `data/validation/validation_report.txt` | 144 lines | validation report |
| `data/validation/validation_report_withoutmanual.txt` | 144 lines | validation report excluding manual results |

---

## Manuscript

| Item | Path |
| --- | --- |
| Word manuscript | `manuscript/Paper3_Preprint_0820.docx` |
| Compiled PDF | `manuscript/Paper3_Preprint_0820.pdf` |
| LaTeX source | `manuscript/latex/` |
| Figures | `manuscript/figures/` |

The figures in `manuscript/figures/` are regenerated by the pipeline. The copies used to typeset the deposited PDF are in `manuscript/latex/figures/`; see `manuscript/figures/README.md`.

---

## Software versions

| Component | Version |
| --- | --- |
| Python | 3.11.9 |
| pip | 25.3 |
| setuptools | 65.5.0 |
| rpy2 | 3.6.4 |
| R | 4.4.3 |

R packages: `boot` 1.3-32, `cli` 3.6.6, `farver` 2.1.2, `ggplot2` 4.0.3, `glue` 1.8.1, `gtable` 0.3.6, `isoband` 0.3.0, `labeling` 0.4.3, `lifecycle` 1.0.5, `R6` 2.6.1, `RColorBrewer` 1.1-3, `rlang` 1.2.0, `S7` 0.2.2, `scales` 1.4.0, `vctrs` 0.7.3, `viridisLite` 0.4.3, `withr` 3.0.3.

Python package versions are pinned in `requirements.txt`.

---

## Citation

```text
[Zenodo DOI to be added on deposit]
```

---

## Licence

See [LICENSE](LICENSE). This work is licensed under Creative Commons Attribution 4.0 International.