# Results summary (as reported in the manuscript)

Values from the runs reported in the manuscript. Re-running the scripts regenerates the full tables in this folder; small differences can occur between library versions.

## Dataset
105 counties, 4,091 county-years, harvest years 1982–2023. Anomaly SD = 24.27%.
Years with median county anomaly below −15%: 1989, 1995, 1996, 2007, 2014.

## Year-level correlations, monthly features (n = 42; FDR over 16 tests)
| Weather input | Variable | r | p | q (FDR) |
|---|---|---|---|---|
| County | Hottest temp | −0.208 | 0.186 | 0.429 |
| County | Coldest temp | 0.165 | 0.295 | 0.429 |
| County | Rainfall | 0.193 | 0.221 | 0.429 |
| County | Humidity | 0.339 | 0.028 | 0.266 |
| County | April min temp | 0.051 | 0.749 | 0.749 |
| County | May max temp | −0.135 | 0.395 | 0.451 |
| County | Fall rain | 0.294 | 0.059 | 0.315 |
| County | Spring rain | 0.156 | 0.325 | 0.429 |
| Central point | Hottest temp | −0.127 | 0.423 | 0.451 |
| Central point | Coldest temp | 0.158 | 0.318 | 0.429 |
| Central point | Rainfall | 0.169 | 0.283 | 0.429 |
| Central point | Humidity | 0.273 | 0.081 | 0.323 |
| Central point | April min temp | 0.193 | 0.221 | 0.429 |
| Central point | May max temp | −0.198 | 0.209 | 0.429 |
| Central point | Fall rain | 0.329 | 0.033 | 0.266 |
| Central point | Spring rain | 0.148 | 0.349 | 0.429 |

## Cross-validated R² (mean ± SD, 5 × 5-fold, years held out)
| Weather input | Linear regression | Random forest | Gradient boosting |
|---|---|---|---|
| County: seasonal (monthly) | 0.052 ± 0.026 | −0.021 ± 0.030 | −0.008 ± 0.022 |
| County: seasonal + timing | 0.008 ± 0.046 | −0.099 ± 0.040 | −0.054 ± 0.041 |
| Central point: seasonal | −0.037 ± 0.041 | −0.140 ± 0.065 | −0.097 ± 0.059 |
| Central point: seasonal + timing | −0.126 ± 0.076 | −0.130 ± 0.064 | −0.063 ± 0.033 |
| County: daily extremes | −0.117 ± 0.064 | −0.106 ± 0.028 | −0.151 ± 0.060 |
| County: monthly + daily | 0.000 ± 0.046 | −0.013 ± 0.032 | −0.047 ± 0.050 |
| Central point: daily extremes | −0.120 ± 0.076 | −0.218 ± 0.044 | −0.089 ± 0.047 |

## R² differences (95% year-bootstrap CI)
| Comparison | Model | ΔR² | 95% CI |
|---|---|---|---|
| County vs central (seasonal) | Linear regression | 0.085 | −0.015 to 0.184 |
| County vs central (seasonal + timing) | Linear regression | 0.125 | −0.016 to 0.271 |
| Adding timing features (county) | Linear regression | −0.038 | −0.126 to 0.036 |
| Daily vs monthly (county) | Linear regression | −0.163 | −0.297 to −0.022 |
| Daily vs monthly (county) | Random forest | −0.072 | −0.189 to 0.064 |
| Daily vs monthly (county) | Gradient boosting | −0.129 | −0.277 to 0.016 |
| Monthly + daily vs monthly | Linear regression | −0.040 | −0.184 to 0.090 |
| Monthly + daily vs monthly | Random forest | 0.025 | −0.081 to 0.148 |
| Monthly + daily vs monthly | Gradient boosting | −0.027 | −0.114 to 0.060 |
| County vs central (daily) | Linear regression | −0.004 | −0.109 to 0.136 |
| County vs central (daily) | Random forest | 0.073 | −0.072 to 0.214 |
| County vs central (daily) | Gradient boosting | −0.058 | −0.199 to 0.061 |

## East–west gradient (exploratory; split at 98.5°W)
| Variable | Median r West | Median r East | r with longitude |
|---|---|---|---|
| Oct–Jun precipitation | 0.44 | −0.25 | −0.95 |
| Hottest temperature | −0.45 | 0.03 | 0.92 |
| April minimum temperature | 0.09 | 0.05 | −0.16 (p = 0.099) |

## Precipitation breakpoint
398 mm (95% CI 293–803 mm); below it, yield anomaly changes 11.2% per 100 mm.

## Worst years (statewide means; weather as anomalies)
| Year | Yield anomaly (%) | Apr freeze days | Heat DD >30 °C | Spring rain (mm) | Dry spell (days) |
|---|---|---|---|---|---|
| 1995 | −33.3 | −0.6 | −54.0 | +126.9 | −1.8 |
| 1989 | −29.0 | +0.1 | −28.6 | −73.0 | +6.1 |
| 2007 | −26.9 | +3.7 | −58.4 | +109.9 | −4.2 |
| 2014 | −21.3 | −0.2 | +0.1 | −121.1 | +2.8 |
| 1996 | −19.8 | +0.1 | +39.7 | −14.9 | +1.1 |
| 1986 | −13.1 | −1.5 | −30.5 | −36.4 | −2.3 |

## Mean |SHAP| (random forest, monthly + daily county anomalies; percentage points)
Humidity 5.15 · Oct–Jun rainfall 4.51 · Growing degree-days 3.85 · Winterkill days 1.85 · Fall rain 1.75 · Coldest temp 1.73 · Hottest temp 1.26 · Heat degree-days >30 °C 1.15 · Spring freeze degree-days 1.15 · Spring dry spell 0.91 · Spring rain 0.66 · April freeze days 0.27
