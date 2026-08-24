# Pipeline C Active Statistical Contract

Documentation update timestamp: 2026-07-30 03:21:50 BST (UTC+01:00)

## Status

The active post-aggregation contract uses raw party-month IQR, one joint 5,000-replicate article-cluster measurement bootstrap, the Party Brand Index, a complete Hamed-Rao Mann-Kendall and Sen-slope table, a direct paired RQ2 slope contrast, a joint breakpoint and common-AR likelihood, a party-aware H3 model, dependence-preserving dominance analysis, and a trend-and-break comparison.

The production route uses outcome-independent evenly spaced structured starts and fully random starts for the genetic breakpoint search.

### Stochastic stream contract

Master seed 42 anchors new stochastic procedures, while deterministic versioned child-seed namespaces separate logically distinct streams. Dominance derives a separate named child seed for each stationary-bootstrap block-length specification. Trend-and-break optimization derives a named child seed for each duration, initialization mode, and replicate. The joint common-AR breakpoint diagnostics derive their display-only jitter stream from master seed 42 and record that stream in the breakpoint metadata. Any future stochastic analysis or substantive regeneration must use the versioned child-seed rule and record its namespace, seed, repetition count, software context, inputs, code, and outputs.

## Active files

| File or directory | Role | Status |
|---|---|---|
| `monthly_stance_joint_bootstrap.csv` | Monthly weighted stance points, pointwise intervals, raw IQR, raw-IQR intervals, and party-specific article counts | Validated |
| `monthly_polarization_joint_bootstrap.csv` | Monthly polarization points and pointwise intervals | Validated |
| `monthly_measurement_bootstrap_replicates.csv` | Complete joint stance, raw-IQR, count, and polarization replicate contract | Validated |
| `monthly_message_coherence.csv` | Republican and Democratic coherence transformations derived monotonically from raw IQR | Validated measurement output |
| `party_brand_index_timeseries.csv` | Party Brand Index points, components, calibration values, article counts, and pointwise intervals | Validated |
| `party_brand_index_bootstrap_replicates.csv` | Complete Party Brand Index replicate contract | Validated |
| `statistical_tests_report.txt` | Base-statistics generation log | Generation-stage record |
| `trend_tests.csv` | Four complete Hamed-Rao Mann-Kendall and Sen-slope rows | Validated |
| `party_slope_contrast.csv` | Direct paired Republican-minus-Democratic Sen contrast for RQ2 | Validated |
| `party_slope_contrast_metadata.json` | Paired estimand, algebraic identity, source, code, and output hashes | Current |
| `dominance_ready.csv` | Fixed observed-level 34-month Q1 through Q4 polarization matrix | Validated input |
| `dominance_observed.csv` and `dominance_contrasts.csv` | Observed contributions and paired comparison contract | Validated descriptive result |
| `dominance_bootstrap_replicates.csv` and `dominance_block_sensitivity.csv` | Dependence-preserving temporal and measurement sensitivity | Validated sensitivity |
| `h3_model_comparison.csv`, `h3_coefficients.csv`, and `h3_party_contrast.csv` | Party-aware model selection, effects, and party contrast | Validated; H3 not supported |
| `h3_measurement_replicate_coefficients.csv` | H3 measurement-uncertainty propagation | Validated |
| `trend_break_model_comparison.csv`, `trend_break_coefficients.csv`, and `fitted_endpoint_contrast.csv` | Trend-only, break-only, joint trend comparison, and fitted endpoint | Validated |
| `mann_kendall_s_decomposition.csv` | Exact within-regime and between-regime S arithmetic | Validated descriptive result |
| `breakpoint_models.csv` | Selected and required comparison models for polarization and Party Brand Index | Validated |
| `breakpoint_fitted_series.csv` | Observed values, structural means, one-step fits, and innovations | Validated |
| `breakpoint_regimes.csv` | Regime dates, durations, levels, and adjacent shifts | Validated |
| `joint_ar_breakpoints/` | Complete repeated-search, convergence, diagnostic, focused-comparison, and figure derivatives | Validated diagnostic contract |

## Measurement uncertainty

Within each month, the bootstrap samples unique `stable_article_id` values with replacement and retains every associated party and dimension row. Each replicate recalculates weighted party stances, party-specific raw IQR, polarization, the common coherence calibration constant, the maximum-polarization normalization, and the Party Brand Index. This intact-cluster design preserves dual-party and cross-dimensional covariance and enforces polarization as Republican stance minus Democratic stance in every replicate.

Raw-IQR estimation requires at least four eligible article instances in the party-dimension cell. The observed July 2022 row is available and contains 12 Republican and 121 Democratic articles with valid Q1 scores, Republican raw IQR 72.5, and Party Brand Index 56.19299969599727. Eleven of the 5,000 July draws select only one to three eligible Republican Q1 article instances. Their Republican stance means and polarization values remain finite, but their Republican raw IQR, Republican coherence component, and Party Brand Index are nonestimable. All replicate records remain present, the July valid PBI count is 4,989, and no other month contains a nonestimable PBI value. We retain this resampling nonestimability without imputation and do not describe it as an absent raw month.

The production analysis uses one 5,000-replicate stream under master seed 42. A 500-replicate pilot established runtime feasibility. A nested 2,000-replicate checkpoint left several tied raw-IQR interval lengths unstable relative to the 5,000 sequence. An independent 5,000-replicate child stream then showed that 98.65 percent of 816 interval lengths differed by less than 10 percent. Every Party Brand Index interval met that threshold, with a maximum relative difference of 6.80 percent. The production run required 56.9 seconds in the validated repository environment.

The selection follows the accuracy-based principle in Andrews, D. W. K., and Buchinsky, M. (2000), “A Three-step Method for Choosing the Number of Bootstrap Repetitions,” Econometrica, 68(1), 23-51, https://doi.org/10.1111/1468-0262.00092. We apply its percentage-deviation principle operationally because tied empirical IQR distributions do not support a literal density-based implementation. Cheng, G., Yu, Z., and Huang, J. Z. (2013), “The cluster bootstrap consistency in generalized estimating equations,” Journal of Multivariate Analysis, 115, 33-47, https://doi.org/10.1016/j.jmva.2012.09.003, supports intact-cluster resampling under within-cluster dependence.

Pointwise 95 percent bands quantify uncertainty caused by the observed article sample. They do not quantify LLM scoring uncertainty, monthly serial dependence, breakpoint-date uncertainty, breakpoint-model selection, or political-event attribution.

## Message coherence and Party Brand Index

Raw party-month IQR is the message coherence point measure. Lower raw IQR indicates greater message coherence. Article count does not enter the point calculation. It affects the resampling distribution, interval width, covariance estimates, and subsequent coefficient uncertainty.

For display and index construction, we calculate `C_p = k / (k + IQR_p)`, where `k = 0.5 median(IQR_R) + 0.5 median(IQR_D)` across the 34 Q1 party-month values. The validated observed-data calibration equals 28.343750. The transformation equals 0.5 when raw IQR equals `k` and decreases monotonically as dispersion increases.

The Party Brand Index retains the established polarization component and geometric aggregation:

`P = abs(polarization) / max(abs(polarization))`

`PBI = 100 * (P * C_R * C_D)^(1/3)`

The observed maximum absolute polarization equals 46.482076. Across 34 months, the validated index has mean 57.285840, minimum 43.031747 in March 2022, and maximum 68.233544 in June 2024. Every bootstrap replicate re-estimates `k` and the maximum-polarization normalization because both are data-derived parts of the estimator.

The Party Brand Index cannot test H3 because polarization is one of its components. The party-aware H3 analysis must retain Republican and Democratic raw IQR as separate outcomes.

## Trend contract

We apply the Hamed-Rao Mann-Kendall procedure to each complete raw monthly series. We report Mann-Kendall S, ordinary variance, adjusted variance, the Hamed-Rao correction factor, Z, Kendall's tau, the two-sided p-value, direction, Sen slope, and the 95 percent Sen rank-order interval. Kendall's tau measures ordinal temporal association. Sen slope measures change magnitude.

| Series | Lag evaluated | S | Tau | Z | p-value | Sen slope per month | 95% Sen interval | Correction factor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 polarization | 2 | 185 | 0.329768 | 2.727698 | 0.006378 | 0.289896 | [0.060754, 0.584725] | 1.000000 |
| Q1 Republican stance | 2 | 273 | 0.486631 | 4.032249 | 0.000055 | 0.423731 | [0.233872, 0.666132] | 1.000000 |
| Q1 Democratic stance | 2 | 165 | 0.294118 | 2.431209 | 0.015049 | 0.167643 | [0.065280, 0.277788] | 1.000000 |
| Party Brand Index | 3 | 221 | 0.393939 | 3.261378 | 0.001109 | 0.355445 | [0.149477, 0.540866] | 1.000000 |

The Hamed-Rao method follows Hamed, K. H., and Rao, A. R. (1998), “A modified Mann-Kendall trend test for autocorrelated data,” Journal of Hydrology, 204(1-4), 182-196, https://doi.org/10.1016/S0022-1694(97)00125-X. The magnitude estimator and rank-order interval follow Sen, P. K. (1968), “Estimates of the Regression Coefficient Based on Kendall's Tau,” Journal of the American Statistical Association, 63(324), 1379-1389, https://doi.org/10.1080/01621459.1968.10480934.

### Direct paired RQ2 contrast

For every common month pair, the Republican Q1 pairwise slope minus the Democratic Q1 pairwise slope equals the pairwise slope of the Republican-minus-Democratic polarization series. The median over all 561 paired slopes is therefore exactly the polarization Sen slope. The direct paired estimate equals 0.289896 Q1 stance points per month, with 95 percent interval [0.060754, 0.584725], S = 185, tau = 0.329768, Z = 2.727698, two-sided p = 0.006378, and correction factor = 1.000000.

This direct inferential contrast supports the conclusion that Republican stance increased more than Democratic stance during the study period. The marginal Republican and Democratic Sen slopes equal 0.423731 and 0.167643, and their arithmetic difference equals 0.256088. We retain that marginal difference as descriptive because a difference between marginal medians does not generally equal the median of paired differences. We do not compare separate party p-values or confidence intervals as a test of the between-party difference.

## Joint breakpoint and common-AR contract

We fit piecewise-constant structural means with one common stationary AR order from 0 through 2. We select count, onset dates, and common AR order jointly by minimizing `-2 log L + (2m + p + 2) log N`, where `m` is the number of breakpoints, `p` is the AR order, and `N = 34`. The criterion counts regime shifts, selected locations, AR coefficients, the baseline level, and innovation variance.

For each series and each minimum regime duration from 3 through 6 months, we run five outcome-independent structured starts and five random starts. The resulting 40 starts per series are optimizer replications, not bootstrap replications.

Polarization selects three AR(2) breaks under every duration rule. The 3-, 4-, and 5-month rules select September 2022, April 2023, and September 2023. The 6-month rule changes only the middle onset to March 2023. The primary 3-month result has BIC 218.411528. The required four-break comparison has BIC 219.659920, delta BIC 1.248392.

The Party Brand Index selects one September 2023 break with AR(0) errors under every duration rule. All 40 starts recover the same configuration. The selected model has BIC 215.482048. Its regime levels equal 53.719457 from March 2022 through August 2023 and 61.298021 from September 2023 through December 2024, an upward shift of 7.578565 points. The no-break AR(1) comparison has BIC 224.830744, delta BIC 9.348696. The selected innovations have Ljung-Box p-values of 0.346840 at lag 4 and 0.161681 at lag 8.

We report break dates as estimated new-regime onsets. Monthly article-cluster bands are not intervals for those dates. The method does not provide calibrated post-selection confidence coverage or individual p-values for selected breakpoints. Political-event correspondence remains manuscript interpretation based on independently sourced chronology.

## H3, dominance, and trend-and-break conclusions

Observed general dominance assigns Q4 the largest contribution, 0.180188, followed by Q2 at 0.071620 and Q3 at 0.042453. The stationary bootstrap resamples complete monthly vectors with 5,000 replicates, primary expected block length four, and sensitivities at three and five. Both simultaneous Q4 contrasts include zero, so the result supports a descriptive observed co-movement ranking rather than confirmatory superiority.

The party-aware H3 specification is `raw IQR ~ polarization * party + time * party`, estimated as a bivariate monthly model with unrestricted contemporaneous covariance and an observed-data comparison of independent errors with a common AR(1). BIC selects independent errors. The Republican polarization coefficient equals 0.048660 and the Democratic coefficient equals -0.103962; both measurement intervals include zero. H3 is not supported.

Break-only has the lowest location-adjusted BIC under every examined duration definition. The common slope in the primary break-plus-trend model equals -0.018687 points per month, p = 0.844635, with conditional interval [-0.212469, 0.175096]. The fitted break-only final-minus-initial contrast equals 11.706491 points with conditional interval [9.990686, 13.422296]. Within-regime pairs contribute -18 to Mann-Kendall S, while between-regime pairs contribute 203, yielding total S = 185.
