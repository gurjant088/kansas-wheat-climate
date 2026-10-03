# =====================================================================
# 01_monthly_analysis.py
# Kansas winter wheat: county weather vs a single central point,
# monthly seasonal and timing features (main analysis; manuscript Sections 3.1-3.5)
#
# Run from the repository root:   python code/01_monthly_analysis.py
# Downloads county centroids (U.S. Census) and monthly NASA POWER weather for
# 105 counties + the central point on the first run (~5-10 min), cached in
# data/weather/weather_cache.csv. Outputs: tables in results/, figures in
# results/run_figures/ (the curated, manuscript-numbered figures are in figures/).
#
# Method notes:
#  1. Weather is expressed as anomalies (each county's value minus that
#     county's own long-term mean), so models compare wet vs. dry YEARS rather
#     than wet vs. dry PLACES (removes the east-west Kansas climate gradient).
#  2. Significance is tested at the YEAR level (n = 42 statewide means),
#     because counties in the same year share weather and are not independent.
#  3. Adds timing-specific features: April minimum temperature (freeze),
#     May maximum temperature (grain-fill heat), fall (Oct-Nov) and spring
#     (Mar-May) rainfall. Models are compared with and without them.
# =====================================================================

import os, glob, time, json, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
FIG, RES = "results/run_figures", "results"
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

# ---------------- Settings ----------------
STATE_FIPS = "20"                              # Kansas
CENTRAL_POINT = (39.83, -98.58)                # single central weather point
YEARS = (1981, 2023)                           # weather years (yield 1982-2023)
MIN_YEARS_PER_COUNTY = 25
SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6]  # Oct (previous year) - Jun
PARAMS = ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M"]
SEASONAL = ["Hottest_C", "Coldest_C", "Precip_mm", "RH_pct"]
TIMING = ["AprTmin_C", "MayTmax_C", "FallPrecip_mm", "SpringPrecip_mm"]
ALL_FEATS = SEASONAL + TIMING
LABEL = {"Hottest_C": "Hottest temp (°C)", "Coldest_C": "Coldest temp (°C)",
         "Precip_mm": "Rainfall Oct–Jun (mm)", "RH_pct": "Humidity (%)",
         "AprTmin_C": "April min temp (°C)", "MayTmax_C": "May max temp (°C)",
         "FallPrecip_mm": "Fall rain Oct–Nov (mm)", "SpringPrecip_mm": "Spring rain Mar–May (mm)"}
SHORT = {"Hottest_C": "Hottest temp", "Coldest_C": "Coldest temp", "Precip_mm": "Rainfall",
         "RH_pct": "Humidity", "AprTmin_C": "April min temp", "MayTmax_C": "May max temp",
         "FallPrecip_mm": "Fall rain", "SpringPrecip_mm": "Spring rain"}
YIELD_ITEM = "WHEAT, WINTER - YIELD, MEASURED IN BU / ACRE"
CACHE = "data/weather/weather_cache.csv"
os.makedirs("data/weather", exist_ok=True)
N_FOLDS, REPEATS = 5, 5                        # year-grouped CV, repeated
BU_AC_TO_KG_HA = 67.25                         # wheat: 1 bu/ac = 67.25 kg/ha
SRC_COLORS = {"County: seasonal": "#9CC3E0", "County: seasonal + timing": "#2A7AB0",
              "Central point: seasonal": "#E6A1A0", "Central point: seasonal + timing": "#C0504D"}


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight")
    plt.show()


# =====================================================================
# PART 1: DATA
# =====================================================================

# ---- 1a. Find the NASS file automatically ----
def find_nass():
    files = sorted(set(glob.glob("data/raw/*.csv") + glob.glob("*.csv")
                       + glob.glob("/content/**/*.csv", recursive=True)))
    for f in files:
        try:
            head = pd.read_csv(f, dtype=str, nrows=5000)
        except Exception:
            continue
        if {"Data Item", "County ANSI", "Value", "Year"} <= set(head.columns) and \
                (head["Data Item"] == YIELD_ITEM).any():
            return f
    raise FileNotFoundError("Upload the NASS county wheat CSV. CSV files found: " + str(files))


NASS_FILE = find_nass()
print("Using NASS file:", NASS_FILE)

nass = pd.read_csv(NASS_FILE, dtype=str)
nass = nass[nass["Data Item"] == YIELD_ITEM]
nass = nass[nass["County ANSI"].notna() & (nass["County ANSI"].str.strip() != "")]
nass = nass[~nass["Value"].str.contains(r"\(", na=True)]          # drop (D), (NA)
nass["yield_bu_ac"] = nass["Value"].str.replace(",", "").astype(float)
nass["Year"] = nass["Year"].astype(int)
nass["FIPS"] = nass["State ANSI"].str.zfill(2) + nass["County ANSI"].str.zfill(3)
nass = nass[nass["Year"].between(YEARS[0] + 1, YEARS[1])]
yld = nass[["FIPS", "County", "Year", "yield_bu_ac"]].drop_duplicates(["FIPS", "Year"])
n_years = yld.groupby("FIPS")["Year"].count()
yld = yld[yld["FIPS"].isin(n_years[n_years >= MIN_YEARS_PER_COUNTY].index)]
print(f"Yield data: {yld['FIPS'].nunique()} counties, {len(yld)} county-years")

# ---- 1b. County centroids (U.S. Census Gazetteer) ----
GAZ = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
       "2023_Gazetteer/2023_Gaz_counties_national.zip")
gaz = pd.read_csv(GAZ, sep="\t", dtype={"GEOID": str})
gaz.columns = gaz.columns.str.strip()
cents = gaz[gaz["GEOID"].str.startswith(STATE_FIPS)].set_index("GEOID")[["INTPTLAT", "INTPTLONG"]]
cents = cents[cents.index.isin(yld["FIPS"].unique())]
print(f"Centroids found for {len(cents)} counties")


# ---- 1c. Weather from NASA POWER (cached) ----
def fetch_power(lat, lon):
    url = ("https://power.larc.nasa.gov/api/temporal/monthly/point?"
           f"parameters={','.join(PARAMS)}&community=AG&latitude={lat:.4f}"
           f"&longitude={lon:.4f}&start={YEARS[0]}&end={YEARS[1]}&format=JSON")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.load(r)["properties"]["parameter"]
            rows = []
            for key in data[PARAMS[0]]:
                yr, mo = int(key[:4]), int(key[4:])
                if mo == 13:                       # annual value, skip
                    continue
                rows.append({"Year": yr, "Month": mo, **{p: data[p][key] for p in PARAMS}})
            return pd.DataFrame(rows)
        except Exception as e:
            print("  retry", attempt + 1, e); time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"NASA POWER failed for {lat}, {lon}")


cache = pd.read_csv(CACHE, dtype={"FIPS": str}) if os.path.exists(CACHE) else pd.DataFrame()
done = set(cache["FIPS"]) if len(cache) else set()
todo = [f for f in list(cents.index) + ["CENTRAL"] if f not in done]
for i, fips in enumerate(todo, 1):
    lat, lon = CENTRAL_POINT if fips == "CENTRAL" else cents.loc[fips].values
    w = fetch_power(lat, lon); w["FIPS"] = fips
    cache = pd.concat([cache, w], ignore_index=True)
    cache.to_csv(CACHE, index=False)
    print(f"Weather {i}/{len(todo)} downloaded ({fips})")
    time.sleep(1)


# ---- 1d. Growing-season and timing features ----
w = cache.replace(-999, np.nan).copy()
w["days"] = pd.to_datetime(dict(year=w["Year"], month=w["Month"], day=1)).dt.days_in_month
w["prec_mm"] = w["PRECTOTCORR"] * w["days"]
w = w[w["Month"].isin(SEASON_MONTHS)]
w["SeasonYear"] = np.where(w["Month"] >= 10, w["Year"] + 1, w["Year"])
g = w.groupby(["FIPS", "SeasonYear"])
season = pd.DataFrame({
    "Hottest_C": g["T2M_MAX"].mean(), "Coldest_C": g["T2M_MIN"].mean(),
    "Precip_mm": g["prec_mm"].sum(min_count=9), "RH_pct": g["RH2M"].mean(),
    "n_months": g.size()})


def month_value(col, months, how):
    sub = w[w["Month"].isin(months)].groupby(["FIPS", "SeasonYear"])[col]
    return sub.sum(min_count=len(months)) if how == "sum" else sub.mean()


season["AprTmin_C"] = month_value("T2M_MIN", [4], "mean")           # spring freeze
season["MayTmax_C"] = month_value("T2M_MAX", [5], "mean")           # grain-fill heat
season["FallPrecip_mm"] = month_value("prec_mm", [10, 11], "sum")   # establishment
season["SpringPrecip_mm"] = month_value("prec_mm", [3, 4, 5], "sum")
season = season.reset_index().rename(columns={"SeasonYear": "Year"})
season = season[season["n_months"] == 9].drop(columns="n_months")
county_w = season[season["FIPS"] != "CENTRAL"]
central_w = season[season["FIPS"] == "CENTRAL"].drop(columns="FIPS")

# ---- 1e. Merge, detrend yields, convert weather to anomalies ----
df = (yld.merge(county_w, on=["FIPS", "Year"])
         .merge(central_w, on="Year", suffixes=("", "_central")).dropna())


def county_anomaly(sub):
    t = np.polyval(np.polyfit(sub["Year"], sub["yield_bu_ac"], 1), sub["Year"])
    return (sub["yield_bu_ac"] - t) / t * 100


df["anomaly"] = df.groupby("FIPS", group_keys=False)[["Year", "yield_bu_ac"]].apply(county_anomaly)
# Weather anomalies: county value minus that county's mean (fix 1);
# the central point minus its own mean over the same years.
for f in ALL_FEATS:
    df[f + "_a"] = df[f] - df.groupby("FIPS")[f].transform("mean")
    df[f + "_ca"] = df[f + "_central"] - df.drop_duplicates("Year")[f + "_central"].mean()
df.to_csv(os.path.join(RES, "county_wheat_dataset.csv"), index=False)
print(f"Final dataset: {len(df)} county-years, {df['FIPS'].nunique()} counties, "
      f"{df['Year'].min()}-{df['Year'].max()}")

SETUPS = {
    "County: seasonal": [f + "_a" for f in SEASONAL],
    "County: seasonal + timing": [f + "_a" for f in ALL_FEATS],
    "Central point: seasonal": [f + "_ca" for f in SEASONAL],
    "Central point: seasonal + timing": [f + "_ca" for f in ALL_FEATS],
}
y, years = df["anomaly"].values, df["Year"].values
uniq_years = np.unique(years)
year_idx = {yr: np.where(years == yr)[0] for yr in uniq_years}


def fdr(p):
    p = np.asarray(p); order = np.argsort(p); q = np.empty(len(p))
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


# =====================================================================
# PART 2: ANALYSIS (tables)
# =====================================================================

# ---- Table 1: year-level correlations (fix 2), county vs central ----
# Each year = mean over counties, so n = number of years (independent units).
yearly = df.groupby("Year").mean(numeric_only=True)
rows = []
for source, suffix in [("County weather", "_a"), ("Central point", "_ca")]:
    for f in ALL_FEATS:
        r, p = pearsonr(yearly[f + suffix], yearly["anomaly"])
        r_pooled = pearsonr(df[f + suffix], df["anomaly"])[0]
        rows.append({"Weather input": source, "Variable": SHORT[f], "r_year": r,
                     "p_year": p, "r_pooled_county_years": r_pooled})
corr = pd.DataFrame(rows)
corr["q_FDR"] = fdr(corr["p_year"])
print(f"=== Table 1: Year-level correlation with yield anomaly (n = {len(yearly)} years, "
      f"FDR over {len(corr)} tests) ===")
print(corr.round(3).to_string(index=False))
corr.to_csv(os.path.join(RES, "correlations_year_level_fdr.csv"), index=False)

# ---- Table 2: collinearity (county weather anomalies) ----
print("\n=== Table 2: Correlation among county weather anomalies ===")
coll = df[[f + "_a" for f in ALL_FEATS]].corr().round(2)
coll.index = coll.columns = [SHORT[f] for f in ALL_FEATS]
print(coll)
coll.to_csv(os.path.join(RES, "collinearity.csv"))

# ---- Table 3: repeated year-grouped cross-validation ----
# Whole years are held out together, so a model can never learn a year's
# weather from neighbouring counties (prevents spatial leakage).
MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=200, max_depth=8,
                                                   min_samples_leaf=10, random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                           learning_rate=0.05, random_state=42),
}
cv_rows, preds = [], {}
for rep in range(REPEATS):
    rng = np.random.default_rng(rep)
    fold_of_year = dict(zip(rng.permutation(uniq_years), np.arange(len(uniq_years)) % N_FOLDS))
    folds = np.array([fold_of_year[yr] for yr in years])
    for setup, cols in SETUPS.items():
        X = df[cols].values
        for name, make in MODELS.items():
            pred = np.empty_like(y)
            for k in range(N_FOLDS):
                tr, te = folds != k, folds == k
                pred[te] = make().fit(X[tr], y[tr]).predict(X[te])
            preds.setdefault((setup, name), []).append(pred)
            cv_rows.append({"Repeat": rep, "Weather input": setup, "Model": name,
                            "R2": r2_score(y, pred), "RMSE": np.sqrt(mean_squared_error(y, pred))})
    print(f"Cross-validation repeat {rep + 1}/{REPEATS} done")
preds = {k: np.mean(v, axis=0) for k, v in preds.items()}     # average over repeats
cv = pd.DataFrame(cv_rows)
cv.to_csv(os.path.join(RES, "cv_all_repeats.csv"), index=False)
cv_sum = cv.groupby(["Weather input", "Model"]).agg(
    R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
    RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std")).round(3)
cv_sum = cv_sum.reindex(list(SETUPS), level="Weather input")
print(f"\n=== Table 3: {REPEATS}x repeated {N_FOLDS}-fold year-grouped CV ===")
print(cv_sum)
print(f"Anomaly SD (error of predicting the mean): {y.std():.2f}%")
cv_sum.to_csv(os.path.join(RES, "cv_summary.csv"))

# ---- Table 4: R² gains with year-bootstrap 95% CI ----
best = cv_sum.loc["County: seasonal + timing"]["R2_mean"].idxmax()
COMPARISONS = [
    ("County vs central point (seasonal)", "County: seasonal", "Central point: seasonal"),
    ("County vs central point (seasonal + timing)", "County: seasonal + timing",
     "Central point: seasonal + timing"),
    ("Adding timing features (county)", "County: seasonal + timing", "County: seasonal"),
]
rng = np.random.default_rng(42)
boot_sets = [np.concatenate([year_idx[yr] for yr in rng.choice(uniq_years, len(uniq_years))])
             for _ in range(1000)]
gain_rows = []
for label, a, b in COMPARISONS:
    pa, pb = preds[(a, best)], preds[(b, best)]
    point = r2_score(y, pa) - r2_score(y, pb)
    boots = [r2_score(y[i], pa[i]) - r2_score(y[i], pb[i]) for i in boot_sets]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    gain_rows.append({"Comparison": label, "Model": best, "R2_gain": point,
                      "CI_low": lo, "CI_high": hi, "Significant": lo > 0 or hi < 0})
gains = pd.DataFrame(gain_rows).round(3)
print(f"\n=== Table 4: R² gains ({best}, predictions averaged over repeats) ===")
print(gains.to_string(index=False))
gains.to_csv(os.path.join(RES, "r2_gains.csv"), index=False)


# ---- Table 5: rainfall threshold (breakpoint regression on absolute rainfall) ----
def hinge_fit(rain, anom, bp):
    X = np.column_stack([np.ones_like(rain), np.minimum(rain - bp, 0)])
    coef, *_ = np.linalg.lstsq(X, anom, rcond=None)
    return coef, np.sum((anom - X @ coef) ** 2)


rain = df["Precip_mm"].values
grid = np.arange(np.percentile(rain, 10), np.percentile(rain, 90), 5)
bp_best = grid[int(np.argmin([hinge_fit(rain, y, b)[1] for b in grid]))]
coef, _ = hinge_fit(rain, y, bp_best)
boot = [grid[int(np.argmin([hinge_fit(rain[i], y[i], b)[1] for b in grid]))]
        for i in boot_sets[:300]]
bp_lo, bp_hi = np.percentile(boot, [2.5, 97.5])
print(f"\n=== Table 5: Rainfall breakpoint ===")
print(f"Breakpoint {bp_best:.0f} mm (95% CI {bp_lo:.0f}-{bp_hi:.0f}); below it, yield "
      f"changes {coef[1] * 100:.1f}% per 100 mm")
pd.DataFrame({"breakpoint_mm": [bp_best], "ci_low": [bp_lo], "ci_high": [bp_hi],
              "slope_pct_per_100mm": [coef[1] * 100]}).to_csv(
    os.path.join(RES, "rainfall_breakpoint.csv"), index=False)

# ---- Table 6: per-county correlations (within each county over time) ----
cc = df.groupby("FIPS").apply(
    lambda d: pd.Series({"r_rain": pearsonr(d["Precip_mm"], d["anomaly"])[0],
                         "r_heat": pearsonr(d["Hottest_C"], d["anomaly"])[0],
                         "r_aprmin": pearsonr(d["AprTmin_C"], d["anomaly"])[0],
                         "mean_yield": d["yield_bu_ac"].mean()}), include_groups=False)
cc = cc.join(cents)
print(f"\n=== Table 6: Per-county correlations ===")
for col, name, sign in [("r_rain", "Rainfall", ">"), ("r_heat", "Hottest temp", "<"),
                        ("r_aprmin", "April min temp", ">")]:
    share = (cc[col] > 0).mean() if sign == ">" else (cc[col] < 0).mean()
    print(f"{name:15s} median r {cc[col].median():+.2f}; "
          f"{'positive' if sign == '>' else 'negative'} in {share * 100:.0f}% of counties")
cc.to_csv(os.path.join(RES, "county_correlations.csv"))


# PART 3: FIGURES
# =====================================================================

# ---- Fig 1: Study area map (counties, centroids, central point) ----
try:
    import geopandas as gpd
    counties = gpd.read_file(
        "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_20m.zip")
    counties = counties[counties["STATEFP"] == STATE_FIPS]
    counties["FIPS"] = counties["STATEFP"] + counties["COUNTYFP"]
    counties = counties.merge(cc, left_on="FIPS", right_index=True, how="left")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    counties.plot(ax=ax, column="mean_yield", cmap="YlGn", edgecolor="white",
                  legend=True, missing_kwds={"color": "#DDDDDD"},
                  legend_kwds={"label": "Mean winter wheat yield (bu/ac)"})
    ax.scatter(cc["INTPTLONG"], cc["INTPTLAT"], s=8, color="black", label="County weather points")
    ax.scatter(CENTRAL_POINT[1], CENTRAL_POINT[0], marker="*", s=450, color="red",
               edgecolor="black", zorder=5, label="Single central point")
    ax.legend(loc="lower left"); ax.set_axis_off()
    ax.set_title("Study area: Kansas counties and weather input locations")
    HAVE_MAP = True
except Exception as e:
    print("County map download failed, drawing simple version:", e)
    HAVE_MAP = False
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(cc["INTPTLONG"], cc["INTPTLAT"], c=cc["mean_yield"], cmap="YlGn",
                    s=120, edgecolor="black")
    plt.colorbar(sc, label="Mean winter wheat yield (bu/ac)")
    ax.scatter(CENTRAL_POINT[1], CENTRAL_POINT[0], marker="*", s=450, color="red",
               edgecolor="black", label="Single central point")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend(loc="lower left")
    ax.set_title("Study area: Kansas county centroids and central weather point")
save("fig01_study_area_map.png")

# ---- Fig 2: Yield trends (state mean and county spread) ----
yr_stats = df.groupby("Year")["yield_bu_ac"].agg(["mean", lambda s: s.quantile(.1),
                                                  lambda s: s.quantile(.9)])
yr_stats.columns = ["mean", "p10", "p90"]
plt.figure(figsize=(10, 4))
plt.fill_between(yr_stats.index, yr_stats["p10"], yr_stats["p90"], color="#A0522D",
                 alpha=0.25, label="County 10th-90th percentile")
plt.plot(yr_stats.index, yr_stats["mean"], color="#A0522D", marker="o", ms=3,
         label="Mean of counties")
plt.ylabel("Yield (bu/ac)"); plt.xlabel("Year"); plt.legend()
plt.title("Kansas county winter wheat yields, 1982–2023")
save("fig02_yield_trends.png")

# ---- Fig 3: Anomalies over time, low-yield years highlighted ----
an = df.groupby("Year")["anomaly"].agg(["median", lambda s: s.quantile(.25),
                                        lambda s: s.quantile(.75)])
an.columns = ["median", "q25", "q75"]
bad = an.index[an["median"] < -15]
plt.figure(figsize=(10, 4))
plt.bar(an.index, an["median"], color=np.where(an["median"] >= 0, "#A0522D", "#999999"))
plt.errorbar(an.index, an["median"], yerr=[an["median"] - an["q25"], an["q75"] - an["median"]],
             fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5)
for b in bad:
    plt.axvspan(b - 0.5, b + 0.5, color="red", alpha=0.15)
plt.axhline(0, color="black", linewidth=0.7)
plt.ylabel("Yield anomaly (% from county trend)"); plt.xlabel("Year")
plt.title("Median county yield anomaly (bars: interquartile range; red: median < −15%)")
save("fig03_yield_anomalies.png")
print("Years with median anomaly below -15%:", list(bad))

# ---- Fig 4: Year-level correlation heatmap (county vs central weather) ----
order_vars = [SHORT[f] for f in ALL_FEATS]
r_mat = corr.pivot(index="Weather input", columns="Variable", values="r_year").loc[
    ["County weather", "Central point"], order_vars]
q_mat = corr.pivot(index="Weather input", columns="Variable", values="q_FDR").loc[
    ["County weather", "Central point"], order_vars]
annot = r_mat.round(2).astype(str) + q_mat.map(
    lambda v: "**" if v < 0.01 else "*" if v < 0.05 else "")
plt.figure(figsize=(11, 2.8))
sns.heatmap(r_mat.astype(float), annot=annot, fmt="", cmap="RdBu", vmin=-0.8, vmax=0.8,
            linewidths=0.5, cbar_kws={"label": "Pearson r"})
plt.ylabel(""); plt.xticks(rotation=20, ha="right")
plt.title(f"Year-level correlation with yield anomaly (n = {len(yearly)} years; "
          "* q < 0.05, ** q < 0.01, FDR)")
save("fig04_correlation_heatmap.png")

# ---- Fig 5: Rainfall and heat anomaly scatter (county vs central) ----
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for row, (setup, suffix) in enumerate([("County weather", "_a"), ("Central point", "_ca")]):
    for col_i, (feat, lab) in enumerate([("Precip_mm" + suffix, "Oct–Jun rainfall anomaly (mm)"),
                                         ("Hottest_C" + suffix, "Oct–Jun hottest temp anomaly (°C)")]):
        ax = axes[row, col_i]
        x = df[feat].values
        r, pv = pearsonr(x, y)
        ax.hexbin(x, y, gridsize=35, cmap="Greys", mincnt=1)
        sl, ic = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, sl * xs + ic, color="#2A7AB0" if row == 0 else "#C0504D", linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_ylim(np.percentile(y, 0.5), np.percentile(y, 99.5))
        ax.set_xlabel(lab); ax.set_ylabel("Yield anomaly (%)")
        ax.set_title(f"{setup}: pooled r = {r:.2f}")
save("fig05_scatter_county_vs_central.png")

# ---- Fig 6: Observed vs predicted (best model, both weather inputs) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
lim = np.percentile(np.abs(y), 99.5)
for ax, setup in zip(axes, ["County: seasonal + timing", "Central point: seasonal"]):
    pr = preds[(setup, best)]
    ax.hexbin(y, pr, gridsize=40, cmap="Blues" if setup.startswith("County") else "Reds",
              mincnt=1, extent=(-lim, lim, -lim, lim))
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("Observed anomaly (%)"); ax.set_ylabel("Predicted anomaly (%)")
    ax.set_title(f"{setup}\n({best}), R² = {r2_score(y, pr):.2f}", fontsize=10)
save("fig06_observed_vs_predicted.png")

# ---- Fig 7: Cross-validated R², county vs central ----
plt.figure(figsize=(10, 4.8))
sns.barplot(data=cv, x="Model", y="R2", hue="Weather input", hue_order=list(SETUPS),
            errorbar="sd", palette=SRC_COLORS, capsize=0.08)
plt.legend(fontsize=8)
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Cross-validated R² (mean ± SD)")
plt.title(f"Predictive skill by weather input ({REPEATS}× {N_FOLDS}-fold, years held out)")
save("fig07_r2_county_vs_central.png")

# ---- Fig 8: Rainfall threshold ----
bins = pd.cut(df["Precip_mm"], 25)
binned = df.groupby(bins, observed=True).agg(rain=("Precip_mm", "mean"), anom=("anomaly", "mean"))
xs = np.linspace(rain.min(), rain.max(), 200)
plt.figure(figsize=(9, 4.5))
plt.scatter(rain, y, s=5, alpha=0.12, color="gray", label="County-years")
plt.scatter(binned["rain"], binned["anom"], color="#A0522D", edgecolor="black", zorder=3,
            label="Binned mean")
plt.plot(xs, coef[0] + coef[1] * np.minimum(xs - bp_best, 0), color="black", linewidth=2,
         label=f"Breakpoint fit ({bp_best:.0f} mm)")
plt.axvspan(bp_lo, bp_hi, color="#A0522D", alpha=0.15, label="95% CI of breakpoint")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.7)
plt.ylim(np.percentile(y, 1), np.percentile(y, 99))
plt.xlabel("Oct–Jun rainfall (mm)"); plt.ylabel("Yield anomaly (%)"); plt.legend(fontsize=8)
plt.title("Kansas winter wheat: rainfall threshold")
save("fig08_rainfall_threshold.png")

# ---- Fig 9: County correlation maps (rainfall and heat) ----
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (colname, title) in zip(axes, [("r_rain", "Rainfall vs yield anomaly"),
                                       ("r_heat", "Hottest temp vs yield anomaly")]):
    if HAVE_MAP:
        counties.plot(ax=ax, column=colname, cmap="RdBu", vmin=-0.8, vmax=0.8,
                      edgecolor="white", legend=True, missing_kwds={"color": "#DDDDDD"},
                      legend_kwds={"label": "Pearson r", "shrink": 0.7})
        ax.set_axis_off()
    else:
        sc = ax.scatter(cc["INTPTLONG"], cc["INTPTLAT"], c=cc[colname], cmap="RdBu",
                        vmin=-0.8, vmax=0.8, s=100, edgecolor="black")
        plt.colorbar(sc, ax=ax, label="Pearson r")
    ax.scatter(CENTRAL_POINT[1], CENTRAL_POINT[0], marker="*", s=300, color="red",
               edgecolor="black", zorder=5)
    ax.set_title(f"County-level r: {title}")
save("fig09_county_correlation_maps.png")

# ---- Fig 10-12: SHAP (county weather, Random Forest) ----
try:
    import shap
    Xs = df[[f + "_a" for f in ALL_FEATS]].copy(); Xs.columns = [LABEL[f] for f in ALL_FEATS]
    rf = MODELS["Random Forest"]().fit(Xs, y)
    sample = Xs.sample(min(1500, len(Xs)), random_state=42)
    sv = shap.TreeExplainer(rf).shap_values(sample)

    shap_imp = pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns).sort_values()
    print("\n=== Table 7: Mean |SHAP| (percentage points of yield anomaly) ===")
    print(shap_imp.round(2)[::-1])
    shap_imp.to_csv(os.path.join(RES, "shap_importance.csv"))
    shap_imp.plot(kind="barh", figsize=(7, 4.5), color="#A0522D")      # Fig 10
    plt.xlabel("Mean |SHAP| (% yield anomaly)")
    plt.title("Climate factor importance (SHAP, county weather anomalies)")
    save("fig10_shap_bar.png")

    shap.summary_plot(sv, sample, show=False)                             # Fig 11
    plt.title("SHAP values (weather anomalies): effect on yield anomaly (%)")
    save("fig11_shap_beeswarm.png")

    # Fig 12
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, feat in zip(axes, [LABEL["Precip_mm"], LABEL["Hottest_C"], LABEL["AprTmin_C"]]):
        shap.dependence_plot(feat, sv, sample, interaction_index=None, ax=ax, show=False)
        ax.set_title(f"Effect of {feat} anomaly", fontsize=10)
    save("fig12_shap_dependence.png")
except ImportError:
    print("\nSHAP not installed. Run  !pip install shap -q  and re-run for Figs 10-12.")

print("\nDone. Tables are in results/ and figures in results/run_figures/.")
