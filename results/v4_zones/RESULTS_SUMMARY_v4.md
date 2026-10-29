# Results summary: agro-climatic zones, zone-specific models and change through time (script 04)

Full console output of the run: `run_output_v4.txt`. Figures: `figures/v4_zones/`.
Data: 105 counties, 4,091 county-years, 1982–2023. Held-out R² uses whole years held out (5-fold × 5 repeats).

## Zones (climate only)
- Five zones (k = 5; silhouette 0.40; k = 6 gives 0.42): Z1 north-west (15 counties, 339 mm), Z2 south-west (17, 374 mm, hottest), Z3 central (25, 457 mm), Z4 north-east (29, 611 mm), Z5 south-east (19, 749 mm).
- Stability under year-bootstrap: mean ARI 0.74 (5–95%: 0.53–0.98).
- 37 of 105 counties: late-period (2003–2023) climate matches a different, warmer zone than early-period (1982–2002).

## Zone-specific vs statewide models (monthly features)
| Model | Statewide R² | Zones R² | ΔR² (95% CI) | East–west split ΔR² (95% CI) |
|---|---|---|---|---|
| Linear | 0.052 | 0.075 | +0.031 (−0.029, 0.096) | +0.037 (−0.017, 0.102) |
| Ridge | 0.054 | 0.089 | +0.040 (−0.020, 0.109) | +0.041 (−0.013, 0.106) |
| Random forest | −0.022 | 0.014 | +0.044 (−0.045, 0.138) | +0.070 (−0.010, 0.159) |

- Real zones vs 300 random zone assignments: permutation p = 0.003 (linear and ridge).
- Out-of-time (train 1982–2002, test 2003–2023): ridge R² 0.080 (statewide) vs 0.162 (zones), ΔR² = +0.082 (95% CI 0.020–0.154); linear +0.088 (0.025–0.162).
- Within-zone ridge R²: Z1 0.19, Z2 0.30, Z3 0.11, Z4 −0.02, Z5 −0.04 (zone-specific); no per-zone gain significant.

## Change through time (1982–2002 vs 2003–2023)
- Hottest temperature trend ≈ +0.5 °C/decade statewide (q = 0.031); significant after FDR in Z2, Z4, Z5.
- Heat-stress years: 10.0% → 17.1% statewide (+7.1 pp, CI −3.0 to 17.6; n.s.); Z2 27.7% → 50.7% (n.s.). Only 3 counties newly exposed.
- Winterkill years: 7.4% → 1.0% statewide (significant); Z4 17.4% → 3.6% (significant).
- No significant shift in yield–weather coefficients (0 of 20 after FDR). Yield-anomaly SD rose in Z1 (22.3 → 28.5%) and Z2 (23.6 → 30.0%) (descriptive).

## R² with 95% year-bootstrap confidence intervals (all configurations)

```
     Features         Model                    Strategy  R2_mean  R2_CI_lo  R2_CI_hi
      monthly        Linear                   Statewide    0.052    -0.085     0.148
      monthly        Linear Statewide + zone intercepts    0.036    -0.103     0.133
      monthly        Linear         East/West split (2)    0.085    -0.047     0.189
      monthly        Linear         Agro-climatic zones    0.075    -0.060     0.177
      monthly         Ridge                   Statewide    0.054    -0.080     0.147
      monthly         Ridge Statewide + zone intercepts    0.038    -0.096     0.131
      monthly         Ridge         East/West split (2)    0.092    -0.031     0.190
      monthly         Ridge         Agro-climatic zones    0.089    -0.026     0.181
      monthly Random Forest                   Statewide   -0.022    -0.206     0.092
      monthly Random Forest Statewide + zone intercepts    0.056    -0.110     0.153
      monthly Random Forest         East/West split (2)    0.052    -0.124     0.159
      monthly Random Forest         Agro-climatic zones    0.014    -0.147     0.116
monthly+daily        Linear                   Statewide    0.000    -0.220     0.155
monthly+daily        Linear Statewide + zone intercepts   -0.016    -0.235     0.138
monthly+daily        Linear         East/West split (2)    0.003    -0.193     0.150
monthly+daily        Linear         Agro-climatic zones   -0.042    -0.254     0.112
monthly+daily         Ridge                   Statewide    0.006    -0.203     0.152
monthly+daily         Ridge Statewide + zone intercepts   -0.010    -0.221     0.136
monthly+daily         Ridge         East/West split (2)    0.019    -0.168     0.155
monthly+daily         Ridge         Agro-climatic zones   -0.005    -0.183     0.129
monthly+daily Random Forest                   Statewide   -0.011    -0.184     0.102
monthly+daily Random Forest Statewide + zone intercepts    0.019    -0.153     0.130
monthly+daily Random Forest         East/West split (2)    0.085    -0.090     0.204
monthly+daily Random Forest         Agro-climatic zones    0.027    -0.140     0.141
```

All intervals include zero: absolute within-period skill is not distinguishable from zero. Written to `results/v4_zones/cv_r2_confidence_intervals.csv` by script 04.
