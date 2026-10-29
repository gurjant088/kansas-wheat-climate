# =====================================================================
# Kansas Winter Wheat [v4]: Agro-climatic zones, zone-specific models,
# and change through time (1982-2023)
#
# Questions
#   Q1  Do data-driven agro-climatic zones exist in Kansas, and are they stable?
#   Q2  Do zone-specific models beat one statewide model in whole-year-held-out
#       cross-validation, and more than arbitrary (random) zones would?
#   Q3  How have heat, freeze and drought exposure changed between 1982-2002 and
#       2003-2023 in each zone? Are counties newly exposed to heat stress?
#   Q4  Has the yield-weather relationship itself shifted between the periods?
#
# Run from the repository root:  python code/04_agroclimatic_zones_analysis.py
#   - Reads data/raw/*.csv (NASS) and reuses caches in data/weather/ if present
#     (otherwise downloads NASA POWER data, ~20-30 min on first run)
#   - No SHAP needed. Runtime ~5-10 min after weather is cached.
#   - Outputs: figures/v4_zones/ and results/v4_zones/
# =====================================================================

import os, glob, time, json, urllib.request, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import theilslopes, kendalltau, wilcoxon
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
FIG, RES, DCACHE = "figures/v4_zones", "results/v4_zones", "data/weather/daily_cache"
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
N_FOLDS, REPEATS, N_BOOT, N_PERM = 5, 5, 1000, 300
SPLIT_YEAR = 2002                     # early = 1982-2002, late = 2003-2023
K_RANGE, K_CHOICE_RANGE = range(2, 7), range(3, 6)

MONTHLY = ["Hottest_C", "Coldest_C", "Precip_mm", "RH_pct"]
DAILY = ["FreezeDays_Apr", "FreezeDD_spring", "WinterKill", "EDD30_MayJun",
         "GDD_MarJun", "FallRain", "SpringRain", "DrySpell_spring"]
ZONE_VARS = ["Precip_mm", "Hottest_C", "Coldest_C", "RH_pct", "GDD_MarJun",
             "EDD30_MayJun", "FreezeDays_Apr", "WinterKill"]
SHORT = {"Hottest_C": "Hottest temp", "Coldest_C": "Coldest temp", "Precip_mm": "Rainfall",
         "RH_pct": "Humidity", "FreezeDays_Apr": "Apr freeze days",
         "FreezeDD_spring": "Spring freeze DD", "WinterKill": "Winterkill days",
         "EDD30_MayJun": "Heat DD >30 °C", "GDD_MarJun": "GDD Mar–Jun", "FallRain": "Fall rain",
         "SpringRain": "Spring rain", "DrySpell_spring": "Dry spell"}
ZCOL = ["#C0504D", "#E1A200", "#4C9A2A", "#2A7AB0", "#7B4EA3", "#555555"]


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight")
    plt.show()


def fdr(p):
    p = np.asarray(p, float); order = np.argsort(p); q = np.empty(len(p))
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# =====================================================================
# PART 1: DATA (identical to v3; reuses the caches)
# =====================================================================
def find_nass():
    files = sorted(set(glob.glob("data/raw/*.csv") + glob.glob("*.csv") + glob.glob("/content/**/*.csv", recursive=True)))
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


mcache = (pd.read_csv(MONTHLY_CACHE, dtype={"FIPS": str})
          if os.path.exists(MONTHLY_CACHE) else pd.DataFrame())
done = set(mcache["FIPS"]) if len(mcache) else set()
for i, f in enumerate([f for f in LOCS if f not in done], 1):
    data = power_json(base_url("monthly", *LOCS[f], YEARS[0], YEARS[1]))
    rows = [{"Year": int(k[:4]), "Month": int(k[4:]), **{p: data[p][k] for p in PARAMS}}
            for k in data[PARAMS[0]] if int(k[4:]) != 13]
    m = pd.DataFrame(rows); m["FIPS"] = f
    mcache = pd.concat([mcache, m], ignore_index=True); mcache.to_csv(MONTHLY_CACHE, index=False)
    print(f"Monthly weather download {i} ({f})"); time.sleep(1)
for i, f in enumerate([f for f in LOCS if not os.path.exists(os.path.join(DCACHE, f"{f}.csv.gz"))], 1):
    data = power_json(base_url("daily", *LOCS[f], f"{YEARS[0]}1001", f"{YEARS[1]}0630"))
    d = pd.DataFrame({p: pd.Series(data[p]) for p in PARAMS})
    d.index = pd.to_datetime(d.index, format="%Y%m%d"); d.index.name = "date"
    d.replace(-999, np.nan).to_csv(os.path.join(DCACHE, f"{f}.csv.gz"))
    print(f"Daily weather download {i} ({f})"); time.sleep(1)


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def daily_features(d):
    d = d.copy()
    d["SeasonYear"] = np.where(d.index.month >= 10, d.index.year + 1, d.index.year)
    d["m"], d["doy"] = d.index.month, d.index.dayofyear
    tmax, tmin, pr = d["T2M_MAX"], d["T2M_MIN"], d["PRECTOTCORR"]
    tavg = ((tmax.clip(upper=30) + tmin.clip(upper=30)) / 2).clip(lower=0)
    d["freeze_apr"] = ((d["m"] == 4) & (tmin < -2)).astype(int)
    spring = (d["doy"] >= 74) & (d["doy"] <= 135)
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
    return out[out["n_days"] >= 270].drop(columns="n_days")


feats = []
for f in LOCS:
    d = pd.read_csv(os.path.join(DCACHE, f"{f}.csv.gz"), index_col="date", parse_dates=True)
    x = daily_features(d); x["FIPS"] = f
    feats.append(x.reset_index().rename(columns={"SeasonYear": "Year"}))
daily = pd.concat(feats, ignore_index=True)

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

W = monthly.merge(daily, on=["FIPS", "Year"])
W = W[W["FIPS"] != "CENTRAL"].dropna(subset=MONTHLY + DAILY)          # ALL county-seasons
W = W[W["FIPS"].isin(LOCS)].copy()
W["Period"] = np.where(W["Year"] <= SPLIT_YEAR, "early", "late")
lonlat = pd.DataFrame(LOCS, index=["lat", "lon"]).T.drop("CENTRAL")
W["lat"] = W["FIPS"].map(lonlat["lat"]); W["lon"] = W["FIPS"].map(lonlat["lon"])
print(f"Weather panel: {W['FIPS'].nunique()} counties x {W['Year'].nunique()} seasons "
      f"({W['Year'].min()}-{W['Year'].max()})")

df = yld.merge(W.drop(columns=["lat", "lon", "Period"]), on=["FIPS", "Year"]).dropna()


def county_anomaly(sub):
    t = np.polyval(np.polyfit(sub["Year"], sub["yield_bu_ac"], 1), sub["Year"])
    return (sub["yield_bu_ac"] - t) / t * 100


df["anomaly"] = df.groupby("FIPS", group_keys=False)[["Year", "yield_bu_ac"]].apply(county_anomaly)
for f in MONTHLY + DAILY:
    df[f + "_a"] = df[f] - df.groupby("FIPS")[f].transform("mean")
df["Period"] = np.where(df["Year"] <= SPLIT_YEAR, "early", "late")
df["lat"] = df["FIPS"].map(lonlat["lat"]); df["lon"] = df["FIPS"].map(lonlat["lon"])
print(f"Yield-weather dataset: {len(df)} county-years, {df['FIPS'].nunique()} counties")

# =====================================================================
# PART 2 (Q1): AGRO-CLIMATIC ZONES  (from climate only - no yield information)
# =====================================================================
clim = W.groupby("FIPS")[ZONE_VARS].mean()
clim = clim[clim.index.isin(df["FIPS"].unique())]
pre = make_pipeline(StandardScaler(), PCA(n_components=0.90, random_state=0))
Z = pre.fit_transform(clim.values)
print(f"\n=== Q1: Agro-climatic zoning ({len(clim)} counties, {Z.shape[1]} PCs = 90% variance) ===")
sil = {}
for k in K_RANGE:
    lab = KMeans(k, n_init=50, random_state=0).fit_predict(Z)
    sil[k] = silhouette_score(Z, lab)
print("Silhouette by k:", {k: round(v, 3) for k, v in sil.items()})
MIN_ZONE = 8                                    # every zone needs >= 8 counties
ok_k = [k for k in K_CHOICE_RANGE
        if np.bincount(KMeans(k, n_init=50, random_state=0).fit_predict(Z)).min() >= MIN_ZONE] or list(K_CHOICE_RANGE)
K = max(ok_k, key=lambda k: sil[k])
print(f"Chosen k = {K} (highest silhouette among k={min(K_CHOICE_RANGE)}-{max(K_CHOICE_RANGE)} "
      f"with every zone >= {MIN_ZONE} counties)")
km = KMeans(K, n_init=50, random_state=0).fit(Z)
order = pd.Series(clim["Precip_mm"].values).groupby(km.labels_).mean().sort_values().index
remap = {old: i + 1 for i, old in enumerate(order)}          # Z1 = driest ... ZK = wettest
zone = pd.Series([f"Z{remap[l]}" for l in km.labels_], index=clim.index, name="Zone")
ZONES = sorted(zone.unique())
zcol = dict(zip(ZONES, ZCOL))
clim["Zone"] = zone; clim["lat"] = lonlat["lat"]; clim["lon"] = lonlat["lon"]
prof = clim.groupby("Zone")[ZONE_VARS + ["lon", "lat"]].mean().round(2)
prof.insert(0, "n_counties", clim.groupby("Zone").size())
print(prof.T.to_string())
prof.T.to_csv(os.path.join(RES, "zone_climate_profiles.csv"))
clim.to_csv(os.path.join(RES, "county_zones.csv"))

# Stability: bootstrap over years -> recompute county means -> re-cluster -> ARI vs reference
rng = np.random.default_rng(1)
yrs_all = np.sort(W["Year"].unique())
Wc = W[W["FIPS"].isin(clim.index)]
ari = []
for b in range(100):
    ys = rng.choice(yrs_all, len(yrs_all))
    pieces = [Wc[Wc["Year"] == y].set_index("FIPS")[ZONE_VARS] for y in ys]
    cm = sum(pieces) / len(pieces)
    cm = cm.loc[clim.index]
    pb = make_pipeline(StandardScaler(), PCA(n_components=0.90, random_state=0))
    lb = KMeans(K, n_init=10, random_state=0).fit_predict(pb.fit_transform(cm.values))
    ari.append(adjusted_rand_score(zone.values, lb))
print(f"\nZone stability (100 year-bootstraps): mean ARI = {np.mean(ari):.2f} "
      f"(5-95%: {np.percentile(ari, 5):.2f}-{np.percentile(ari, 95):.2f}; 1 = identical)")

# Climate migration: classify each county's EARLY and LATE climate with the FIXED full-period zones
per = (Wc.groupby(["FIPS", "Period"])[ZONE_VARS].mean().reset_index())
mig = {}
for p in ("early", "late"):
    x = per[per["Period"] == p].set_index("FIPS").loc[clim.index, ZONE_VARS]
    mig[p] = pd.Series([f"Z{remap[l]}" for l in km.predict(pre.transform(x.values))], index=clim.index)
moved = (mig["early"] != mig["late"])
print(f"Counties whose LATE-period climate matches a different zone than their EARLY-period "
      f"climate (fixed zone definitions): {int(moved.sum())} of {len(clim)}")
if moved.any():
    print(pd.crosstab(mig["early"][moved], mig["late"][moved]).rename_axis("early", axis=0)
          .rename_axis("late", axis=1))
shift = (per[per["Period"] == "late"].set_index("FIPS")[ZONE_VARS]
         - per[per["Period"] == "early"].set_index("FIPS")[ZONE_VARS]).loc[clim.index]
shift["Zone"] = zone
print("\nMean climate change (late minus early) by zone:")
print(shift.groupby("Zone").mean().round(2).to_string())
shift.groupby("Zone").mean().round(3).to_csv(os.path.join(RES, "zone_climate_shift_late_minus_early.csv"))

df["Zone"] = df["FIPS"].map(zone)
df = df.dropna(subset=["Zone"]).reset_index(drop=True)
W["Zone"] = W["FIPS"].map(zone)
Wz = W.dropna(subset=["Zone"]).copy()
Wz["Zone"] = Wz["Zone"].astype(str)

# ---- Figures 1-3 ----
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.3, 1.2, 1]})
ax = axes[0]
for z in ZONES:
    s = clim[clim["Zone"] == z]
    ax.scatter(s["lon"], s["lat"], s=95, color=zcol[z], edgecolor="black", label=f"{z} (n={len(s)})")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("(a) Agro-climatic zones"); ax.legend(fontsize=8)
ax = axes[1]
zs = (clim[ZONE_VARS] - clim[ZONE_VARS].mean()) / clim[ZONE_VARS].std()
hm = zs.groupby(clim["Zone"]).mean()
im = ax.imshow(hm.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(ZONE_VARS))); ax.set_xticklabels([SHORT[v] for v in ZONE_VARS], rotation=60, ha="right")
ax.set_yticks(range(len(ZONES))); ax.set_yticklabels(hm.index); ax.set_title("(b) Zone climate (z-score)")
plt.colorbar(im, ax=ax, fraction=0.046)
ax = axes[2]
ax.plot(list(sil), list(sil.values()), "o-", color="black"); ax.axvline(K, color="red", ls="--")
ax.set_xlabel("Number of zones k"); ax.set_ylabel("Silhouette"); ax.set_title(f"(c) Choice of k (chosen = {K})")
save("v01_agroclimatic_zones.png")

# =====================================================================
# PART 3 (Q2): ZONE-SPECIFIC vs STATEWIDE MODELS (whole years held out)
# =====================================================================
y = df["anomaly"].values
years = df["Year"].values
uniq_years = np.unique(years)
year_idx = {yr: np.where(years == yr)[0] for yr in uniq_years}
zarr = df["Zone"].values
west = (df["lon"].values > -98.5)          # False = west of 98.5 W (earlier two-region split)
split2 = np.where(df["lon"].values <= -98.5, "W", "E")

MODELS = {
    "Linear": lambda: LinearRegression(),
    "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=30.0)),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10,
                                                   random_state=42, n_jobs=-1),
}
FEATSETS = {"monthly": MONTHLY, "monthly+daily": MONTHLY + DAILY}
STRATS = ["Statewide", "Statewide + zone intercepts", "East/West split (2)", "Agro-climatic zones"]


def fit_predict(strategy, make, X, y_, grp, tr, te):
    """grp: zone labels (used by zone strategies)."""
    if strategy == "Statewide":
        return make().fit(X[tr], y_[tr]).predict(X[te])
    if strategy == "Statewide + zone intercepts":
        D = pd.get_dummies(grp).astype(float).values
        XD = np.hstack([X, D])
        return make().fit(XD[tr], y_[tr]).predict(XD[te])
    out = np.empty(len(te))
    base = None
    for gname in np.unique(grp):
        mtr, mte = tr[grp[tr] == gname], np.where(grp[te] == gname)[0]
        if len(mte) == 0:
            continue
        if len(mtr) < 60:                                   # too few rows -> fall back to statewide
            if base is None:
                base = make().fit(X[tr], y_[tr])
            out[mte] = base.predict(X[te[mte]])
        else:
            out[mte] = make().fit(X[mtr], y_[mtr]).predict(X[te[mte]])
    return out


def folds_for(rep):
    rng_ = np.random.default_rng(rep)
    f_of_y = dict(zip(rng_.permutation(uniq_years), np.arange(len(uniq_years)) % N_FOLDS))
    return np.array([f_of_y[yr] for yr in years])


print("\n=== Q2: Statewide vs zone-specific models (year-grouped 5-fold CV, "
      f"{REPEATS} repeats) ===")
P = {}                                                      # (featset, model, strategy) -> (REPEATS, N)
t0 = time.time()
for fs, cols in FEATSETS.items():
    X = df[[c + "_a" for c in cols]].values
    for mname, make in MODELS.items():
        for st in STRATS:
            grp = {"Statewide": zarr, "Statewide + zone intercepts": zarr,
                   "East/West split (2)": split2, "Agro-climatic zones": zarr}[st]
            pr = np.zeros((REPEATS, len(y)))
            for rep in range(REPEATS):
                fo = folds_for(rep)
                for k_ in range(N_FOLDS):
                    te = np.where(fo == k_)[0]; tr = np.where(fo != k_)[0]
                    pr[rep, te] = fit_predict(st, make, X, y, grp, tr, te)
            P[(fs, mname, st)] = pr
        print(f"  done {fs} / {mname}  ({time.time() - t0:.0f}s)")

rng = np.random.default_rng(7)
boot_idx = []
for b in range(N_BOOT):
    ys = rng.choice(uniq_years, len(uniq_years))
    boot_idx.append(np.concatenate([year_idx[v] for v in ys]))

rows = []
for (fs, mname, st), pr in P.items():
    r2s = [r2(y, pr[r]) for r in range(REPEATS)]
    pm = pr.mean(0)
    base = P[(fs, mname, "Statewide")].mean(0)
    d = np.array([r2(y[i], pm[i]) - r2(y[i], base[i]) for i in boot_idx]) if st != "Statewide" else np.zeros(1)
    rmse = np.sqrt(np.mean((y - pm) ** 2))
    rows.append({"Features": fs, "Model": mname, "Strategy": st, "R2_mean": np.mean(r2s),
                 "R2_sd": np.std(r2s), "RMSE_%": rmse,
                 "dR2_vs_statewide": r2(y, pm) - r2(y, base) if st != "Statewide" else 0.0,
                 "CI_lo": np.percentile(d, 2.5) if st != "Statewide" else np.nan,
                 "CI_hi": np.percentile(d, 97.5) if st != "Statewide" else np.nan})
cvres = pd.DataFrame(rows)
print(cvres.round(3).to_string(index=False))
cvres.to_csv(os.path.join(RES, "cv_statewide_vs_zones.csv"), index=False)

# ---- Per-zone skill (monthly features) ----
zr = []
for mname in MODELS:
    ps = P[("monthly", mname, "Statewide")].mean(0)
    pz = P[("monthly", mname, "Agro-climatic zones")].mean(0)
    for z in ZONES:
        m = zarr == z
        d = []
        for i in boot_idx[:500]:
            ii = i[m[i]]
            d.append(r2(y[ii], pz[ii]) - r2(y[ii], ps[ii]))
        zr.append({"Model": mname, "Zone": z, "n": int(m.sum()),
                   "R2_statewide": r2(y[m], ps[m]), "R2_zone_specific": r2(y[m], pz[m]),
                   "dR2": r2(y[m], pz[m]) - r2(y[m], ps[m]),
                   "CI_lo": np.percentile(d, 2.5), "CI_hi": np.percentile(d, 97.5)})
zr = pd.DataFrame(zr)
print("\nPer-zone held-out skill (monthly features):")
print(zr.round(3).to_string(index=False))
zr.to_csv(os.path.join(RES, "cv_per_zone.csv"), index=False)

# ---- Permutation null: do REAL zones beat arbitrary partitions of the same sizes? ----
print(f"\n=== Permutation test: real zones vs {N_PERM} random zone assignments ===")
Xm = df[[c + "_a" for c in MONTHLY]].values
cty = np.array(sorted(zone.index))
fo_list = [folds_for(s) for s in (0, 1)]
perm_out = {}
for mname in ("Linear", "Ridge"):
    make = MODELS[mname]
    state_cache = [np.zeros(len(y)) for _ in fo_list]
    for si, fo in enumerate(fo_list):
        for k_ in range(N_FOLDS):
            te = np.where(fo == k_)[0]; tr = np.where(fo != k_)[0]
            state_cache[si][te] = fit_predict("Statewide", make, Xm, y, zarr, tr, te)
    r2_state = np.mean([r2(y, s) for s in state_cache])

    def zone_r2(gvec):
        out = []
        for si, fo in enumerate(fo_list):
            pz = np.zeros(len(y))
            for k_ in range(N_FOLDS):
                te = np.where(fo == k_)[0]; tr = np.where(fo != k_)[0]
                pz[te] = fit_predict("Zone", make, Xm, y, gvec, tr, te)
            out.append(r2(y, pz))
        return np.mean(out)

    obs = zone_r2(zarr) - r2_state
    prng = np.random.default_rng(11)
    null = []
    zvals = zone.loc[cty].values
    for _ in range(N_PERM):
        zmap = dict(zip(cty, prng.permutation(zvals)))
        null.append(zone_r2(df["FIPS"].map(zmap).values) - r2_state)
    null = np.array(null)
    pval = (1 + (null >= obs).sum()) / (1 + N_PERM)
    perm_out[mname] = (obs, null, pval)
    print(f"  {mname}: observed dR2 (zones - statewide) = {obs:+.4f}; random-zone null mean "
          f"{null.mean():+.4f} (95th pct {np.percentile(null, 95):+.4f}); permutation p = {pval:.3f}")
pd.DataFrame({m: {"observed_dR2": v[0], "null_mean": v[1].mean(), "null_p95": np.percentile(v[1], 95),
                  "perm_p": v[2]} for m, v in perm_out.items()}).T.to_csv(
    os.path.join(RES, "permutation_test_zones.csv"))

# ---- Temporal transfer: train 1982-2002, test 2003-2023 ----
print("\n=== Out-of-time test: train 1982-2002 -> predict 2003-2023 ===")
tr_i = np.where(years <= SPLIT_YEAR)[0]; te_i = np.where(years > SPLIT_YEAR)[0]
late_years = np.unique(years[te_i])
oot_rows = []
for mname in ("Linear", "Ridge", "Random Forest"):
    pr_ = {}
    for st in ("Statewide", "Statewide + zone intercepts", "Agro-climatic zones"):
        pr_[st] = fit_predict(st, MODELS[mname], Xm, y, zarr, tr_i, te_i)
    yt = y[te_i]
    bs = []
    for b in range(500):
        ys = np.random.default_rng(b).choice(late_years, len(late_years))
        ii = np.concatenate([np.where(years[te_i] == v)[0] for v in ys])
        bs.append(r2(yt[ii], pr_["Agro-climatic zones"][ii]) - r2(yt[ii], pr_["Statewide"][ii]))
    oot_rows.append({"Model": mname, "R2_statewide": r2(yt, pr_["Statewide"]),
                     "R2_zone_intercepts": r2(yt, pr_["Statewide + zone intercepts"]),
                     "R2_zone_specific": r2(yt, pr_["Agro-climatic zones"]),
                     "dR2_zone_vs_state": r2(yt, pr_["Agro-climatic zones"]) - r2(yt, pr_["Statewide"]),
                     "CI_lo": np.percentile(bs, 2.5), "CI_hi": np.percentile(bs, 97.5)})
oot = pd.DataFrame(oot_rows)
print(oot.round(3).to_string(index=False))
oot.to_csv(os.path.join(RES, "out_of_time_test.csv"), index=False)

# ---- Figures 4-5 ----
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
ax = axes[0]
sub = cvres[cvres["Features"] == "monthly"]
width = 0.2
for j, st in enumerate(STRATS):
    s = sub[sub["Strategy"] == st].set_index("Model").loc[list(MODELS)]
    ax.bar(np.arange(len(MODELS)) + (j - 1.5) * width, s["R2_mean"], width, yerr=s["R2_sd"],
           label=st, color=["#BBBBBB", "#9CC3E0", "#E1A200", "#2A7AB0"][j], capsize=2)
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(list(MODELS)); ax.set_ylabel("Held-out R² (whole years)")
ax.set_title("(a) Statewide vs zone-based models"); ax.legend(fontsize=7)
ax = axes[1]
for (mname, (obs, null, pv)), c in zip(perm_out.items(), ["#2A7AB0", "#C0504D"]):
    ax.hist(null, bins=30, alpha=0.5, color=c, label=f"{mname} random zones")
    ax.axvline(obs, color=c, lw=2.2)
ax.set_xlabel("ΔR² (zone-specific − statewide)"); ax.set_ylabel("Random assignments")
ax.set_title("(b) Real zones (lines) vs random zones"); ax.legend(fontsize=8)
ax = axes[2]
sz = zr[zr["Model"] == "Ridge"]
xx = np.arange(len(sz))
ax.bar(xx - 0.2, sz["R2_statewide"], 0.4, color="#BBBBBB", label="Statewide model")
ax.bar(xx + 0.2, sz["R2_zone_specific"], 0.4, color="#2A7AB0", label="Zone-specific model")
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks(xx); ax.set_xticklabels(sz["Zone"]); ax.set_ylabel("Held-out R² within zone")
ax.set_title("(c) Skill within each zone (Ridge)"); ax.legend(fontsize=8)
save("v02_zone_vs_statewide_cv.png")

# =====================================================================
# PART 4 (Q3): CHANGE THROUGH TIME - exposure to heat, freeze, drought
# =====================================================================
print("\n=== Q3: Change in extreme-weather exposure, 1982-2002 vs 2003-2023 ===")
early = Wz[Wz["Period"] == "early"]
EV = {  # name: (variable, direction, label)
    "Heat-stress year": ("EDD30_MayJun", "gt", "May–Jun heat degree-days >30 °C above early-period P90"),
    "Spring-freeze year": ("FreezeDays_Apr", "gt", "April freeze days above early-period P90"),
    "Dry year": ("Precip_mm", "lt", "Oct–Jun rainfall below early-period P10"),
    "Winterkill year": ("WinterKill", "gt", "Winterkill days above early-period P90"),
}
thr = {}
for name, (v, dirn, _) in EV.items():
    q = early[v].quantile(0.10 if dirn == "lt" else 0.90)
    thr[name] = q
    Wz[name] = (Wz[v] < q).astype(int) if dirn == "lt" else (Wz[v] > q).astype(int)
print("Event thresholds (from early-period county-years):",
      {k: round(float(v), 2) for k, v in thr.items()})

yrs_e = np.arange(YEARS[0] + 1, SPLIT_YEAR + 1); yrs_l = np.arange(SPLIT_YEAR + 1, YEARS[1] + 1)
exp_rows = []
for name in EV:
    zy = Wz.groupby(["Zone", "Year"])[name].mean().unstack(0)        # year x zone event share
    zy["Statewide"] = Wz.groupby("Year")[name].mean()
    brng = np.random.default_rng(5)
    for z in list(ZONES) + ["Statewide"]:
        s = zy[z]
        s = s.reindex(np.arange(YEARS[0] + 1, YEARS[1] + 1))
        fe, fl = np.nanmean(s.loc[yrs_e]), np.nanmean(s.loc[yrs_l])
        bd = [np.nanmean(s.loc[brng.choice(yrs_l, len(yrs_l))]) - np.nanmean(s.loc[brng.choice(yrs_e, len(yrs_e))])
              for _ in range(2000)]
        exp_rows.append({"Event": name, "Zone": z, "freq_early_%": 100 * fe, "freq_late_%": 100 * fl,
                         "change_pp": 100 * (fl - fe), "CI_lo": 100 * np.percentile(bd, 2.5),
                         "CI_hi": 100 * np.percentile(bd, 97.5),
                         "sig_95%": bool(np.percentile(bd, 2.5) > 0 or np.percentile(bd, 97.5) < 0)})
expo = pd.DataFrame(exp_rows)
print(expo.round(1).to_string(index=False))
expo.to_csv(os.path.join(RES, "exposure_change_by_zone.csv"), index=False)

# Newly exposed counties
cnt = Wz.groupby(["FIPS", "Period"])[list(EV)].sum().unstack()
newly = {}
for name in EV:
    e, l = cnt[(name, "early")], cnt[(name, "late")]
    newly[name] = (e <= 1) & (l >= 4)
nw = pd.DataFrame({k: v for k, v in newly.items()})
nw["Zone"] = zone
print("\nCounties 'newly exposed' (<=1 event year in 1982-2002 but >=4 in 2003-2023):")
print(nw.groupby("Zone").sum().assign(n_counties=nw.groupby("Zone").size()).to_string())
nw.to_csv(os.path.join(RES, "newly_exposed_counties.csv"))

# Trends (Theil-Sen + Kendall) on zone-mean annual series
TV = ["EDD30_MayJun", "FreezeDays_Apr", "Precip_mm", "Hottest_C", "Coldest_C", "GDD_MarJun"]
tr_rows = []
for z in list(ZONES) + ["Statewide"]:
    sub = Wz if z == "Statewide" else Wz[Wz["Zone"] == z]
    ann = sub.groupby("Year")[TV].mean()
    for v in TV:
        sl = theilslopes(ann[v].values, ann.index.values)[0] * 10
        tau, p = kendalltau(ann.index.values, ann[v].values)
        tr_rows.append({"Zone": z, "Variable": SHORT[v], "Sen_slope_per_decade": sl, "tau": tau, "p": p})
trd = pd.DataFrame(tr_rows); trd["q_FDR"] = fdr(trd["p"])
print("\nTrends 1982-2023 (zone-mean series; Theil-Sen slope per decade, Kendall p, FDR over "
      f"{len(trd)} tests):")
print(trd.round(3).to_string(index=False))
trd.to_csv(os.path.join(RES, "trends_by_zone.csv"), index=False)

# County-level heat exposure change for map
cfreq = (Wz.groupby(["FIPS", "Period"])["Heat-stress year"].mean().unstack() * 100)
cfreq["change"] = cfreq["late"] - cfreq["early"]
cfreq = cfreq.join(lonlat)

# ---- Figures 6-8 ----
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1.2, 1, 1.2]})
ax = axes[0]
sc = ax.scatter(cfreq["lon"], cfreq["lat"], c=cfreq["change"], cmap="RdBu_r", s=100, edgecolor="black",
                vmin=-30, vmax=30)
plt.colorbar(sc, ax=ax, label="Change in heat-stress-year frequency (pp)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("(a) Heat-stress frequency, late − early")
ax = axes[1]
h = expo[expo["Event"] == "Heat-stress year"]
xx = np.arange(len(h))
ax.bar(xx - 0.2, h["freq_early_%"], 0.4, color="#9CC3E0", label="1982–2002")
ax.bar(xx + 0.2, h["freq_late_%"], 0.4, color="#C0504D", label="2003–2023")
ax.set_xticks(xx); ax.set_xticklabels(h["Zone"]); ax.set_ylabel("% of county-seasons")
ax.set_title("(b) Heat-stress years by zone"); ax.legend(fontsize=8)
ax = axes[2]
ev_names = list(EV)
for j, name in enumerate(ev_names):
    e = expo[(expo["Event"] == name) & (expo["Zone"].isin(ZONES))]
    ax.errorbar(np.arange(len(e)) + (j - 1.5) * 0.17, e["change_pp"],
                yerr=[e["change_pp"] - e["CI_lo"], e["CI_hi"] - e["change_pp"]], fmt="o", capsize=3,
                color=["#C0504D", "#2A7AB0", "#E1A200", "#555555"][j], label=name)
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks(range(len(ZONES))); ax.set_xticklabels(ZONES)
ax.set_ylabel("Change in event frequency (pp, late − early)"); ax.set_title("(c) Exposure change with 95% CI")
ax.legend(fontsize=7)
save("v03_exposure_change.png")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
for ax, v in zip(axes, ["EDD30_MayJun", "Precip_mm", "FreezeDays_Apr"]):
    for z in ZONES:
        a = Wz[Wz["Zone"] == z].groupby("Year")[v].mean()
        ax.plot(a.index, a.values, color=zcol[z], lw=1.2, label=z)
        sl, ic = theilslopes(a.values, a.index.values)[:2]
        ax.plot(a.index, ic + sl * a.index.values, color=zcol[z], lw=2.2, ls="--")
    ax.set_title(SHORT[v] + " (zone means; dashed = Theil–Sen)"); ax.set_xlabel("Season (harvest year)")
axes[0].legend(fontsize=8)
save("v04_trends_by_zone.png")

# =====================================================================
# PART 5 (Q4): HAS THE YIELD-WEATHER RESPONSE SHIFTED?
# =====================================================================
print("\n=== Q4: Shift in yield-weather response (late minus early), year-block bootstrap ===")
RV = MONTHLY
resp_rows = []
brng = np.random.default_rng(21)
for z in ZONES:
    sub = df[df["Zone"] == z]
    sd = sub[[c + "_a" for c in RV]].std()
    def coefs(s):
        Xs = s[[c + "_a" for c in RV]].values / sd.values
        return LinearRegression().fit(Xs, s["anomaly"].values).coef_
    se, sl_ = sub[sub["Period"] == "early"], sub[sub["Period"] == "late"]
    if len(se) < 60 or len(sl_) < 60:
        continue
    d0 = coefs(sl_) - coefs(se)
    ge = {yv: se[se["Year"] == yv] for yv in yrs_e}; gl = {yv: sl_[sl_["Year"] == yv] for yv in yrs_l}
    bs = []
    for _ in range(500):
        be = pd.concat([ge[v] for v in brng.choice(yrs_e, len(yrs_e)) if len(ge[v])])
        bl = pd.concat([gl[v] for v in brng.choice(yrs_l, len(yrs_l)) if len(gl[v])])
        bs.append(coefs(bl) - coefs(be))
    bs = np.array(bs)
    for j, v in enumerate(RV):
        p = 2 * min((bs[:, j] >= 0).mean(), (bs[:, j] <= 0).mean())
        resp_rows.append({"Zone": z, "Variable": SHORT[v], "coef_early": coefs(se)[j],
                          "coef_late": coefs(sl_)[j], "shift": d0[j],
                          "CI_lo": np.percentile(bs[:, j], 2.5), "CI_hi": np.percentile(bs[:, j], 97.5),
                          "p_boot": max(p, 1 / 500)})
resp = pd.DataFrame(resp_rows); resp["q_FDR"] = fdr(resp["p_boot"])
print("(coefficients: % yield anomaly per 1 SD of within-county weather anomaly)")
print(resp.round(2).to_string(index=False))
resp.to_csv(os.path.join(RES, "response_shift_by_zone.csv"), index=False)
n_sig = int((resp["q_FDR"] < 0.05).sum())
print(f"Shifts significant after FDR (q<0.05): {n_sig} of {len(resp)}")

sdv = df.groupby(["Zone", "Period"])["anomaly"].std().unstack().round(1)
print("\nYield-anomaly SD (%) by zone and period:"); print(sdv.to_string())
sdv.to_csv(os.path.join(RES, "yield_anomaly_sd_by_period.csv"))

fig, axes = plt.subplots(1, len(ZONES), figsize=(3.6 * len(ZONES), 4), sharex=True, sharey=True)
axes = np.atleast_1d(axes)
for ax, z in zip(axes, ZONES):
    r_ = resp[resp["Zone"] == z]
    yy = np.arange(len(r_))
    ax.errorbar(r_["shift"], yy, xerr=[r_["shift"] - r_["CI_lo"], r_["CI_hi"] - r_["shift"]], fmt="o",
                color=zcol[z], capsize=3)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(yy); ax.set_yticklabels(r_["Variable"]); ax.set_title(z); ax.set_xlabel("Shift in response")
save("v05_response_shift.png")

# Migration map
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, p, t in zip(axes, ("early", "late"), ("1982–2002", "2003–2023")):
    for z in ZONES:
        s = clim[mig[p] == z]
        ax.scatter(s["lon"], s["lat"], s=95, color=zcol[z], edgecolor="black", label=z)
    ax.scatter(clim.loc[moved, "lon"], clim.loc[moved, "lat"], s=230, facecolors="none",
               edgecolors="black", lw=2)
    ax.set_title(f"Climate matched to fixed zones: {t}"); ax.set_xlabel("Longitude")
axes[0].set_ylabel("Latitude"); axes[0].legend(fontsize=8)
save("v06_zone_migration.png")

# =====================================================================
# SUMMARY (paste this block back)
# =====================================================================
print("\n" + "=" * 70 + "\nKEY NUMBERS\n" + "=" * 70)
print(f"Zones k={K}; counties per zone: {clim.groupby('Zone').size().to_dict()}; ARI={np.mean(ari):.2f}")
for m in ("Linear", "Ridge", "Random Forest"):
    s = cvres[(cvres.Features == "monthly") & (cvres.Model == m)].set_index("Strategy")
    print(f"{m:14s} R2 statewide {s.loc['Statewide','R2_mean']:+.3f} | zones {s.loc['Agro-climatic zones','R2_mean']:+.3f} "
          f"(dR2 {s.loc['Agro-climatic zones','dR2_vs_statewide']:+.3f}, CI {s.loc['Agro-climatic zones','CI_lo']:+.3f} to "
          f"{s.loc['Agro-climatic zones','CI_hi']:+.3f})")
print("Permutation p:", {m: round(v[2], 3) for m, v in perm_out.items()})
print("Heat-stress year change (pp):", expo[expo.Event == 'Heat-stress year'][['Zone', 'change_pp', 'CI_lo', 'CI_hi']].round(1).to_dict('records'))
print("Counties changing zone:", int(moved.sum()), "| response shifts sig after FDR:", n_sig)
print("\nSaved: figures/v4_zones/ (v01-v06) and results/v4_zones/ (csv).")


# =====================================================================

# =====================================================================
# EXTRA: 95% CI for the R² of EVERY configuration (incl. statewide baseline)
# Year-bootstrap (1,000 resamples of whole years), same statistic as the
# reported "mean of 5 repeats". Needs P, y, boot_idx, r2 from the v4 run.
# =====================================================================
rows = []
for (fs, mname, st), pr in P.items():
    bs = np.array([np.mean([r2(y[i], pr[r][i]) for r in range(REPEATS)]) for i in boot_idx])
    rows.append({"Features": fs, "Model": mname, "Strategy": st,
                 "R2_mean": np.mean([r2(y, pr[r]) for r in range(REPEATS)]),
                 "R2_CI_lo": np.percentile(bs, 2.5), "R2_CI_hi": np.percentile(bs, 97.5)})
r2ci = pd.DataFrame(rows)
print("\n=== R² with 95% year-bootstrap CI, all configurations ===")
print(r2ci.round(3).to_string(index=False))
r2ci.to_csv(os.path.join(RES, "cv_r2_confidence_intervals.csv"), index=False)


# =====================================================================
# SAVE PROCESSED DATASETS
# =====================================================================
df.to_csv(os.path.join(RES, "county_wheat_dataset_v4.csv"), index=False)       # county-year model dataset (yield + weather + zone)
Wz.to_csv(os.path.join(RES, "county_weather_panel_v4.csv"), index=False)       # all county-seasons weather + event flags
print("Processed dataset: results/v4_zones/county_wheat_dataset_v4.csv")
