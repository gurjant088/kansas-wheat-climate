# =====================================================================
# 02_daily_extremes_analysis.py
# Kansas winter wheat: do daily weather extremes explain yields better than
# monthly averages? (extension analysis; manuscript Sections 3.6-3.8)
#
# Run from the repository root:   python code/02_daily_extremes_analysis.py
# Reuses data/weather/weather_cache.csv from script 01 and downloads DAILY
# NASA POWER data for 106 locations on the first run (~15-25 min), cached in
# data/weather/daily_cache/. Outputs: tables in results/, figures in
# results/run_figures/ (the curated, manuscript-numbered figures are in figures/).
#
# Daily features (per season, harvest year t; Oct = previous year):
#   FreezeDays_Apr   days in April with Tmin < -2 °C (spring freeze)
#   FreezeDD_spring  sum of degrees below -2 °C, 15 Mar - 15 May (freeze severity)
#   WinterKill       days in Dec-Feb with Tmin < -18 °C (winterkill risk)
#   EDD30_MayJun     sum of (Tmax - 30 °C) over days above 30 °C, May-Jun (heat stress)
#   GDD_MarJun       growing degree days, base 0 °C, capped at 30 °C, Mar-Jun
#   FallRain         Oct-Nov precipitation (establishment)
#   SpringRain       Mar-May precipitation
#   DrySpell_spring  longest run of days with < 1 mm rain, Mar-May
# Degree days use daily Tmax/Tmin (an approximation of hourly methods).
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

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
FIG, RES, DCACHE = "results/run_figures", "results", "data/weather/daily_cache"
for d in (FIG, RES, DCACHE):
    os.makedirs(d, exist_ok=True)

# ---------------- Settings ----------------
STATE_FIPS = "20"
CENTRAL_POINT = (39.83, -98.58)
YEARS = (1981, 2023)
MIN_YEARS_PER_COUNTY = 25
PARAMS = ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M"]
YIELD_ITEM = "WHEAT, WINTER - YIELD, MEASURED IN BU / ACRE"
MONTHLY_CACHE = "data/weather/weather_cache.csv"
N_FOLDS, REPEATS = 5, 5

MONTHLY = ["Hottest_C", "Coldest_C", "Precip_mm", "RH_pct"]
DAILY = ["FreezeDays_Apr", "FreezeDD_spring", "WinterKill", "EDD30_MayJun",
         "GDD_MarJun", "FallRain", "SpringRain", "DrySpell_spring"]
LABEL = {"Hottest_C": "Hottest temp (°C)", "Coldest_C": "Coldest temp (°C)",
         "Precip_mm": "Rainfall Oct–Jun (mm)", "RH_pct": "Humidity (%)",
         "FreezeDays_Apr": "April freeze days (<−2 °C)",
         "FreezeDD_spring": "Spring freeze degree-days",
         "WinterKill": "Winterkill days (<−18 °C)",
         "EDD30_MayJun": "Heat degree-days >30 °C (May–Jun)",
         "GDD_MarJun": "Growing degree-days (Mar–Jun)",
         "FallRain": "Fall rain Oct–Nov (mm)", "SpringRain": "Spring rain Mar–May (mm)",
         "DrySpell_spring": "Longest spring dry spell (days)"}
SHORT = {"Hottest_C": "Hottest temp", "Coldest_C": "Coldest temp", "Precip_mm": "Rainfall",
         "RH_pct": "Humidity", "FreezeDays_Apr": "Apr freeze days",
         "FreezeDD_spring": "Spring freeze DD", "WinterKill": "Winterkill days",
         "EDD30_MayJun": "Heat DD >30", "GDD_MarJun": "GDD", "FallRain": "Fall rain",
         "SpringRain": "Spring rain", "DrySpell_spring": "Dry spell"}
SETUPS = {
    "County: monthly": ("_a", MONTHLY),
    "County: daily extremes": ("_a", DAILY),
    "County: monthly + daily": ("_a", MONTHLY + DAILY),
    "Central point: daily extremes": ("_ca", DAILY),
}
SRC_COLORS = {"County: monthly": "#9CC3E0", "County: daily extremes": "#2A7AB0",
              "County: monthly + daily": "#1B4F72", "Central point: daily extremes": "#C0504D"}


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight")
    plt.show()


def fdr(p):
    p = np.asarray(p); order = np.argsort(p); q = np.empty(len(p))
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


# =====================================================================
# PART 1: DATA
# =====================================================================
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
nass = nass[~nass["Value"].str.contains(r"\(", na=True)]
nass["yield_bu_ac"] = nass["Value"].str.replace(",", "").astype(float)
nass["Year"] = nass["Year"].astype(int)
nass["FIPS"] = nass["State ANSI"].str.zfill(2) + nass["County ANSI"].str.zfill(3)
nass = nass[nass["Year"].between(YEARS[0] + 1, YEARS[1])]
yld = nass[["FIPS", "County", "Year", "yield_bu_ac"]].drop_duplicates(["FIPS", "Year"])
n_years = yld.groupby("FIPS")["Year"].count()
yld = yld[yld["FIPS"].isin(n_years[n_years >= MIN_YEARS_PER_COUNTY].index)]
print(f"Yield data: {yld['FIPS'].nunique()} counties, {len(yld)} county-years")

GAZ = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
       "2023_Gazetteer/2023_Gaz_counties_national.zip")
gaz = pd.read_csv(GAZ, sep="\t", dtype={"GEOID": str})
gaz.columns = gaz.columns.str.strip()
cents = gaz[gaz["GEOID"].str.startswith(STATE_FIPS)].set_index("GEOID")[["INTPTLAT", "INTPTLONG"]]
cents = cents[cents.index.isin(yld["FIPS"].unique())]
LOCS = {f: tuple(cents.loc[f].values) for f in cents.index}
LOCS["CENTRAL"] = CENTRAL_POINT
print(f"Locations: {len(LOCS)} (counties + central point)")


def power_json(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)["properties"]["parameter"]
        except Exception as e:
            print("  retry", attempt + 1, e); time.sleep(5 * (attempt + 1))
    raise RuntimeError("NASA POWER request failed: " + url)


def base_url(kind, lat, lon, start, end):
    return (f"https://power.larc.nasa.gov/api/temporal/{kind}/point?"
            f"parameters={','.join(PARAMS)}&community=AG&latitude={lat:.4f}"
            f"&longitude={lon:.4f}&start={start}&end={end}&format=JSON")


# ---- Monthly data (reuses weather_cache.csv from the previous run) ----
mcache = (pd.read_csv(MONTHLY_CACHE, dtype={"FIPS": str})
          if os.path.exists(MONTHLY_CACHE) else pd.DataFrame())
done = set(mcache["FIPS"]) if len(mcache) else set()
todo = [f for f in LOCS if f not in done]
for i, f in enumerate(todo, 1):
    data = power_json(base_url("monthly", *LOCS[f], YEARS[0], YEARS[1]))
    rows = [{"Year": int(k[:4]), "Month": int(k[4:]), **{p: data[p][k] for p in PARAMS}}
            for k in data[PARAMS[0]] if int(k[4:]) != 13]
    m = pd.DataFrame(rows); m["FIPS"] = f
    mcache = pd.concat([mcache, m], ignore_index=True); mcache.to_csv(MONTHLY_CACHE, index=False)
    print(f"Monthly weather {i}/{len(todo)} ({f})"); time.sleep(1)
print("Monthly weather ready")

# ---- Daily data (cached per location) ----
todo = [f for f in LOCS if not os.path.exists(os.path.join(DCACHE, f"{f}.csv.gz"))]
for i, f in enumerate(todo, 1):
    data = power_json(base_url("daily", *LOCS[f], f"{YEARS[0]}1001", f"{YEARS[1]}0630"))
    d = pd.DataFrame({p: pd.Series(data[p]) for p in PARAMS})
    d.index = pd.to_datetime(d.index, format="%Y%m%d"); d.index.name = "date"
    d.replace(-999, np.nan).to_csv(os.path.join(DCACHE, f"{f}.csv.gz"))
    print(f"Daily weather {i}/{len(todo)} ({f})"); time.sleep(1)
print("Daily weather ready")


# =====================================================================
# PART 2: FEATURES
# =====================================================================
def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def daily_features(d):
    """Season features for each harvest year from one location's daily data."""
    d = d.copy()
    d["SeasonYear"] = np.where(d.index.month >= 10, d.index.year + 1, d.index.year)
    d["m"], d["doy"] = d.index.month, d.index.dayofyear
    tmax, tmin, pr = d["T2M_MAX"], d["T2M_MIN"], d["PRECTOTCORR"]
    tavg = ((tmax.clip(upper=30) + tmin.clip(upper=30)) / 2).clip(lower=0)
    d["freeze_apr"] = ((d["m"] == 4) & (tmin < -2)).astype(int)
    spring = (d["doy"] >= 74) & (d["doy"] <= 135)            # ~15 Mar - 15 May
    d["freeze_dd"] = np.where(spring, (-2 - tmin).clip(lower=0), 0)
    d["winterkill"] = (d["m"].isin([12, 1, 2]) & (tmin < -18)).astype(int)
    d["edd30"] = np.where(d["m"].isin([5, 6]), (tmax - 30).clip(lower=0), 0)
    d["gdd"] = np.where(d["m"].isin([3, 4, 5, 6]), tavg, 0)
    d["fall_rain"] = np.where(d["m"].isin([10, 11]), pr, 0)
    d["spring_rain"] = np.where(d["m"].isin([3, 4, 5]), pr, 0)
    g = d.groupby("SeasonYear")
    out = pd.DataFrame({
        "FreezeDays_Apr": g["freeze_apr"].sum(), "FreezeDD_spring": g["freeze_dd"].sum(),
        "WinterKill": g["winterkill"].sum(), "EDD30_MayJun": g["edd30"].sum(),
        "GDD_MarJun": g["gdd"].sum(), "FallRain": g["fall_rain"].sum(),
        "SpringRain": g["spring_rain"].sum(),
        "DrySpell_spring": d[d["m"].isin([3, 4, 5])].groupby("SeasonYear")["PRECTOTCORR"].apply(
            lambda s: longest_run((s < 1).values)),
        "n_days": g.size()})
    return out[out["n_days"] >= 270].drop(columns="n_days")    # complete seasons only


feats = []
for f in LOCS:
    d = pd.read_csv(os.path.join(DCACHE, f"{f}.csv.gz"), index_col="date", parse_dates=True)
    x = daily_features(d); x["FIPS"] = f
    feats.append(x.reset_index().rename(columns={"SeasonYear": "Year"}))
daily = pd.concat(feats, ignore_index=True)

# Monthly seasonal features (as in v2)
w = mcache.replace(-999, np.nan).copy()
w = w[w["Month"].isin([10, 11, 12, 1, 2, 3, 4, 5, 6])]
w["SeasonYear"] = np.where(w["Month"] >= 10, w["Year"] + 1, w["Year"])
w["days"] = pd.to_datetime(dict(year=w["Year"], month=w["Month"], day=1)).dt.days_in_month
w["prec_mm"] = w["PRECTOTCORR"] * w["days"]
g = w.groupby(["FIPS", "SeasonYear"])
monthly = pd.DataFrame({"Hottest_C": g["T2M_MAX"].mean(), "Coldest_C": g["T2M_MIN"].mean(),
                        "Precip_mm": g["prec_mm"].sum(min_count=9), "RH_pct": g["RH2M"].mean(),
                        "n": g.size()}).reset_index().rename(columns={"SeasonYear": "Year"})
monthly = monthly[monthly["n"] == 9].drop(columns="n")

weather = monthly.merge(daily, on=["FIPS", "Year"])
county_w = weather[weather["FIPS"] != "CENTRAL"]
central_w = weather[weather["FIPS"] == "CENTRAL"].drop(columns="FIPS")
df = (yld.merge(county_w, on=["FIPS", "Year"])
         .merge(central_w, on="Year", suffixes=("", "_central")).dropna())


def county_anomaly(sub):
    t = np.polyval(np.polyfit(sub["Year"], sub["yield_bu_ac"], 1), sub["Year"])
    return (sub["yield_bu_ac"] - t) / t * 100


df["anomaly"] = df.groupby("FIPS", group_keys=False)[["Year", "yield_bu_ac"]].apply(county_anomaly)
ALL = MONTHLY + DAILY
for f in ALL:
    df[f + "_a"] = df[f] - df.groupby("FIPS")[f].transform("mean")
    df[f + "_ca"] = df[f + "_central"] - df.drop_duplicates("Year")[f + "_central"].mean()
df.to_csv(os.path.join(RES, "county_wheat_dataset_daily.csv"), index=False)
print(f"Final dataset: {len(df)} county-years, {df['FIPS'].nunique()} counties, "
      f"{df['Year'].min()}-{df['Year'].max()}")

print("\n=== Daily feature summary (county-years) ===")
print(df[DAILY].describe().T[["mean", "std", "min", "max"]].round(1))

y, years = df["anomaly"].values, df["Year"].values
uniq_years = np.unique(years)
year_idx = {yr: np.where(years == yr)[0] for yr in uniq_years}


# =====================================================================
# PART 3: ANALYSIS (tables)
# =====================================================================

# ---- Table 1: year-level correlations, daily features (county vs central) ----
yearly = df.groupby("Year").mean(numeric_only=True)
rows = []
for source, suffix in [("County weather", "_a"), ("Central point", "_ca")]:
    for f in DAILY:
        r, p = pearsonr(yearly[f + suffix], yearly["anomaly"])
        rows.append({"Weather input": source, "Variable": SHORT[f], "r_year": r, "p_year": p,
                     "r_pooled": pearsonr(df[f + suffix], df["anomaly"])[0]})
corr = pd.DataFrame(rows); corr["q_FDR"] = fdr(corr["p_year"])
print(f"\n=== Table 1: Year-level correlation, daily features (n = {len(yearly)} years, "
      f"FDR over {len(corr)} tests) ===")
print(corr.round(3).to_string(index=False))
corr.to_csv(os.path.join(RES, "daily_correlations_year_level_fdr.csv"), index=False)

# ---- Table 2: collinearity among daily features ----
coll = df[[f + "_a" for f in DAILY]].corr().round(2)
coll.index = coll.columns = [SHORT[f] for f in DAILY]
print("\n=== Table 2: Correlation among county daily-feature anomalies ===")
print(coll)
coll.to_csv(os.path.join(RES, "daily_collinearity.csv"))

# ---- Table 3: repeated year-grouped cross-validation ----
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
    for setup, (suffix, feats_) in SETUPS.items():
        X = df[[f + suffix for f in feats_]].values
        for name, make in MODELS.items():
            pred = np.empty_like(y)
            for k in range(N_FOLDS):
                tr, te = folds != k, folds == k
                pred[te] = make().fit(X[tr], y[tr]).predict(X[te])
            preds.setdefault((setup, name), []).append(pred)
            cv_rows.append({"Repeat": rep, "Weather input": setup, "Model": name,
                            "R2": r2_score(y, pred), "RMSE": np.sqrt(mean_squared_error(y, pred))})
    print(f"Cross-validation repeat {rep + 1}/{REPEATS} done")
preds = {k: np.mean(v, axis=0) for k, v in preds.items()}
cv = pd.DataFrame(cv_rows); cv.to_csv(os.path.join(RES, "cv_all_repeats_daily.csv"), index=False)
cv_sum = cv.groupby(["Weather input", "Model"]).agg(
    R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
    RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std")).round(3)
cv_sum = cv_sum.reindex(list(SETUPS), level="Weather input")
print(f"\n=== Table 3: {REPEATS}x repeated {N_FOLDS}-fold year-grouped CV ===")
print(cv_sum)
print(f"Anomaly SD (error of predicting the mean): {y.std():.2f}%")
cv_sum.to_csv(os.path.join(RES, "cv_summary_daily.csv"))

# ---- Table 4: R² gains with year-bootstrap 95% CI (per model) ----
COMPARISONS = [
    ("Daily extremes vs monthly (county)", "County: daily extremes", "County: monthly"),
    ("Monthly + daily vs monthly (county)", "County: monthly + daily", "County: monthly"),
    ("County vs central point (daily)", "County: daily extremes", "Central point: daily extremes"),
]
rng = np.random.default_rng(42)
boot_sets = [np.concatenate([year_idx[yr] for yr in rng.choice(uniq_years, len(uniq_years))])
             for _ in range(1000)]
gain_rows = []
for model in MODELS:
    for label, a, b in COMPARISONS:
        pa, pb = preds[(a, model)], preds[(b, model)]
        boots = [r2_score(y[i], pa[i]) - r2_score(y[i], pb[i]) for i in boot_sets]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        gain_rows.append({"Model": model, "Comparison": label,
                          "R2_gain": r2_score(y, pa) - r2_score(y, pb),
                          "CI_low": lo, "CI_high": hi, "Significant": lo > 0 or hi < 0})
gains = pd.DataFrame(gain_rows).round(3)
print("\n=== Table 4: R² gains (predictions averaged over repeats, 95% year-bootstrap CI) ===")
print(gains.to_string(index=False))
gains.to_csv(os.path.join(RES, "r2_gains_daily.csv"), index=False)

# ---- Table 5: worst years and their extremes ----
yr_tab = yearly[["anomaly"] + [f + "_a" for f in ["FreezeDays_Apr", "EDD30_MayJun",
                                                   "SpringRain", "DrySpell_spring"]]].copy()
yr_tab.columns = ["Yield anomaly (%)", "Apr freeze days", "Heat DD >30", "Spring rain (mm)",
                  "Dry spell (days)"]
print("\n=== Table 5: Six worst years (statewide means; weather as anomalies) ===")
print(yr_tab.sort_values("Yield anomaly (%)").head(6).round(1))
yr_tab.round(2).to_csv(os.path.join(RES, "yearly_extremes.csv"))


# =====================================================================
# PART 4: FIGURES
# =====================================================================

# ---- Fig D1: year-level correlation heatmap (daily features) ----
order_vars = [SHORT[f] for f in DAILY]
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
plt.title(f"Daily extremes: year-level correlation with yield anomaly "
          f"(n = {len(yearly)}; * q < 0.05, ** q < 0.01, FDR)")
save("figD1_daily_correlation_heatmap.png")

# ---- Fig D2: R² comparison ----
plt.figure(figsize=(10, 4.8))
sns.barplot(data=cv, x="Model", y="R2", hue="Weather input", hue_order=list(SETUPS),
            errorbar="sd", palette=SRC_COLORS, capsize=0.08)
plt.axhline(0, color="black", linewidth=0.8); plt.legend(fontsize=8)
plt.ylabel("Cross-validated R² (mean ± SD)")
plt.title(f"Monthly averages vs. daily extremes ({REPEATS}× {N_FOLDS}-fold, years held out)")
save("figD2_r2_monthly_vs_daily.png")

# ---- Fig D3: year-level scatter for key extremes ----
key = ["FreezeDays_Apr", "EDD30_MayJun", "SpringRain", "DrySpell_spring"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, f in zip(axes, key):
    x = yearly[f + "_a"].values; yy = yearly["anomaly"].values
    r, p = pearsonr(x, yy)
    ax.scatter(x, yy, color="#A0522D", edgecolor="black")
    for yr in yearly.index[yy < -15]:
        ax.annotate(str(yr), (yearly.loc[yr, f + "_a"], yearly.loc[yr, "anomaly"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    sl, ic = np.polyfit(x, yy, 1); xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, sl * xs + ic, color="black")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel(LABEL[f] + " anomaly", fontsize=9); ax.set_ylabel("Mean yield anomaly (%)")
    ax.set_title(f"r = {r:.2f}, p = {p:.3f}", fontsize=10)
fig.suptitle("Statewide yield anomaly vs. daily weather extremes (each point = one year)")
save("figD3_yearly_extremes_scatter.png")

# ---- Fig D4: observed vs predicted, county monthly vs county monthly + daily ----
best_model = cv_sum.loc["County: monthly"]["R2_mean"].idxmax()
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
lim = np.percentile(np.abs(y), 99.5)
for ax, setup in zip(axes, ["County: monthly", "County: monthly + daily"]):
    pr = preds[(setup, best_model)]
    ax.hexbin(y, pr, gridsize=40, cmap="Blues", mincnt=1, extent=(-lim, lim, -lim, lim))
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("Observed anomaly (%)"); ax.set_ylabel("Predicted anomaly (%)")
    ax.set_title(f"{setup} ({best_model})\nR² = {r2_score(y, pr):.2f}", fontsize=10)
save("figD4_observed_vs_predicted.png")

# ---- Fig D5-D7: SHAP for the county monthly + daily model ----
try:
    import shap
    cols = [f + "_a" for f in MONTHLY + DAILY]
    Xs = df[cols].copy(); Xs.columns = [LABEL[f] for f in MONTHLY + DAILY]
    rf = MODELS["Random Forest"]().fit(Xs, y)
    sample = Xs.sample(min(1500, len(Xs)), random_state=42)
    sv = shap.TreeExplainer(rf).shap_values(sample)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns).sort_values()
    print("\n=== Table 6: Mean |SHAP| (monthly + daily, county anomalies) ===")
    print(imp.round(2)[::-1])
    imp.to_csv(os.path.join(RES, "shap_importance_daily.csv"))
    imp.plot(kind="barh", figsize=(8, 5.5),
             color=["#2A7AB0" if c in [LABEL[f] for f in DAILY] else "#9CC3E0" for c in imp.index])
    plt.xlabel("Mean |SHAP| (% yield anomaly)")
    plt.title("Importance: monthly averages (light) vs. daily extremes (dark)")
    save("figD5_shap_bar.png")

    shap.summary_plot(sv, sample, show=False)
    plt.title("SHAP values: monthly + daily weather anomalies")
    save("figD6_shap_beeswarm.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, f in zip(axes, ["FreezeDD_spring", "EDD30_MayJun", "SpringRain"]):
        shap.dependence_plot(LABEL[f], sv, sample, interaction_index=None, ax=ax, show=False)
        ax.set_title(f"Effect of {LABEL[f]} anomaly", fontsize=10)
    save("figD7_shap_dependence.png")
except ImportError:
    print("\nSHAP not installed. Run  !pip install shap -q  and re-run for Figs D5-D7.")

print("\nDone. Tables are in results/ and figures in results/run_figures/.")
