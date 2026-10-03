# Is Statewide Climate Sensitivity of Winter Wheat an Artefact of Aggregation?

**Opposing East–West Precipitation Responses in Kansas (1982–2023)**

This repository contains the data, code and figures for a county-level study of how weather relates to Kansas winter wheat yields, and why statewide climate–yield models explain so little yield variability.

## Key findings

- **Low predictive skill.** Under cross-validation with whole years held out, the best model (linear regression on county monthly weather anomalies) explained about 5% of yield variance (R² = 0.052). No weather variable was significant at the year level after false discovery rate (FDR) correction.
- **County weather vs. one central point.** County-matched weather outperformed a single central point for every seasonal model, but the gain was not statistically significant (ΔR² = 0.085, 95% CI −0.015 to 0.184).
- **Daily extremes did not help.** Daily freeze, heat, winterkill and dry-spell indices performed worse than monthly averages.
- **Opposing east–west responses (exploratory).** Within-county rainfall–yield correlations declined almost linearly from west to east (r = −0.95 with longitude): positive in semi-arid western Kansas (median r = 0.44) and negative in humid eastern Kansas (median r = −0.25). Heat harmed yields mainly in the west. These opposing responses cancel in statewide models.
- **Multiple routes to failure.** The worst years (1989, 1995, 1996, 2007, 2014) were associated with drought, excess wetness, spring freeze and heat.

## Repository structure

```
├── code/
│   ├── 01_monthly_analysis.py         # main analysis: county vs central point, monthly features
│   ├── 02_daily_extremes_analysis.py  # extension: daily extreme-weather indices vs monthly
│   └── 03_east_west_analysis.py       # exploratory east–west gradient (manuscript Table 3)
├── data/
│   ├── raw/
│   │   └── nass_kansas_winter_wheat_county_1981_2025.csv   # USDA NASS Quick Stats download
│   └── weather/                       # NASA POWER caches (created automatically by the scripts)
├── figures/                           # figures as numbered in the manuscript (1–12)
│   └── supplementary/                 # supplementary figures S1–S6
├── results/                           # output tables (created by the scripts)
│   └── RESULTS_SUMMARY.md             # key results reported in the manuscript
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## How to reproduce

Requires Python 3.9 or later and an internet connection (for U.S. Census county centroids and NASA POWER weather). Run from the repository root:

```bash
pip install -r requirements.txt
python code/01_monthly_analysis.py         # ~5–10 min on first run (monthly weather download)
python code/02_daily_extremes_analysis.py  # ~15–25 min on first run (daily weather download)
python code/03_east_west_analysis.py       # seconds
```

Weather downloads are cached in `data/weather/`, so re-runs take only a few minutes and an interrupted download continues where it stopped. Tables are written to `results/` and figures to `results/run_figures/`; the curated, manuscript-numbered figures are in `figures/`. Small numerical differences can occur between library versions.

**Google Colab:** clone the repository (`!git clone <repo-url>`), change into it (`%cd <repo-name>`), run `!pip install shap -q`, then run each script with `!python code/<script>.py`.

## Data

| Dataset | Source | Details |
| --- | --- | --- |
| County yields | [USDA NASS Quick Stats](https://quickstats.nass.usda.gov) | Kansas, county level, "WHEAT, WINTER – YIELD, MEASURED IN BU / ACRE", domain total. The raw file contains several wheat items; the scripts keep only total winter wheat yield. 105 counties, 4,091 county-years used (1982–2023). |
| Weather | [NASA POWER](https://power.larc.nasa.gov) (Agroclimatology) | MERRA-2 monthly and daily temperature, humidity and precipitation at each county internal point and at 39.83°N, 98.58°W (the single central point). |
| County locations | [U.S. Census Bureau 2023 Gazetteer](https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html) | County internal points, downloaded automatically. |

## Methods summary

1. Yields were detrended per county (linear trend) and expressed as percentage anomalies.
2. Weather features for the October–June season were expressed as within-county anomalies, so models compare wet and dry years rather than wet and dry places.
3. Significance was tested at the year level (n = 42 statewide means) with Benjamini–Hochberg FDR correction, because counties in the same year share weather.
4. Linear regression, random forest and gradient boosting were evaluated by 5 × repeated 5-fold cross-validation with whole years held out; R² differences were tested with year-bootstrap confidence intervals.
5. Models were interpreted with SHAP; within-county correlations were related to longitude (exploratory).

## Citation

Please cite the associated paper (citation to be added on publication) or this repository using `CITATION.cff`, and cite the original data sources (USDA NASS, NASA POWER).

## License

Code is released under the MIT License. Raw data remain subject to the terms of use of USDA NASS and NASA.

## Contact

Gurjant Singh · [Email address]
