# =====================================================================
# Kansas winter wheat, script 05: COUNTY MAPS + SHAP for the zones analysis
#
# Run from the repository root AFTER script 04:
#   pip install shap geopandas
#   python code/05_county_maps_and_shap.py
# Reuses data/weather caches and the NASS CSV (no new weather downloads).
# Downloads only the Census county boundary file for the maps (if that fails,
# maps fall back to dots at county points).
# Outputs (figures/v4_zones/ and results/v4_zones/):
#   m1-m4  county maps (study area, zones, within-county correlations, exposure change)
#   s1-s4  SHAP for a random forest with zone information
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
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

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

Wc = W[W["FIPS"].isin(clim.index)]
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


# =====================================================================
# PART 3: COUNTY STATISTICS FOR THE MAPS
# =====================================================================
saved_zones = os.path.join(RES, "county_zones.csv")
if os.path.exists(saved_zones):
    sz = pd.read_csv(saved_zones, dtype={0: str}, index_col=0)["Zone"]
    same = (sz.reindex(zone.index) == zone).mean()
    print(f"\nZones recomputed here match results_v4/county_zones.csv for {same:.0%} of counties")

Wz["Heat"] = (Wz["EDD30_MayJun"] > Wz.loc[Wz["Period"] == "early", "EDD30_MayJun"].quantile(0.90)).astype(int)
Wz["WK"] = (Wz["WinterKill"] > Wz.loc[Wz["Period"] == "early", "WinterKill"].quantile(0.90)).astype(int)
per = Wz.groupby(["FIPS", "Period"]).agg(Heat=("Heat", "mean"), WK=("WK", "mean"),
                                         EDD=("EDD30_MayJun", "mean")).unstack()
cs = pd.DataFrame({
    "heat_change_pp": 100 * (per[("Heat", "late")] - per[("Heat", "early")]),
    "edd_change": per[("EDD", "late")] - per[("EDD", "early")],
    "winterkill_change_pp": 100 * (per[("WK", "late")] - per[("WK", "early")])})
sdp = df.groupby(["FIPS", "Period"])["anomaly"].std().unstack()
cs["yield_sd_change"] = sdp["late"] - sdp["early"]
cs["r_rain"] = df.groupby("FIPS").apply(lambda g: g["anomaly"].corr(g["Precip_mm_a"]))
cs["r_heat"] = df.groupby("FIPS").apply(lambda g: g["anomaly"].corr(g["Hottest_C_a"]))
cs["mean_yield"] = df.groupby("FIPS")["yield_bu_ac"].mean()
cs["Zone"] = zone; cs["zone_early"] = mig["early"]; cs["zone_late"] = mig["late"]
cs["lat"] = lonlat["lat"]; cs["lon"] = lonlat["lon"]
cs = cs.loc[cs.index.isin(zone.index)]
cs.to_csv(os.path.join(RES, "county_statistics_for_maps.csv"))
print(cs.groupby("Zone")[["r_rain", "r_heat", "heat_change_pp", "winterkill_change_pp",
                          "yield_sd_change"]].mean().round(2).to_string())

# =====================================================================
# PART 4: COUNTY MAPS
# =====================================================================
COUNTY_SHP = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_20m.zip"
COUNTIES, ZB = None, None
try:
    import geopandas as gpd
    COUNTIES = gpd.read_file(COUNTY_SHP)
    COUNTIES = COUNTIES[COUNTIES["STATEFP"] == STATE_FIPS].copy()
    COUNTIES["FIPS"] = COUNTIES["STATEFP"] + COUNTIES["COUNTYFP"]
    ZB = COUNTIES.merge(zone.rename("Zone"), left_on="FIPS", right_index=True).dissolve(by="Zone").boundary
    print("County boundaries loaded:", len(COUNTIES))
except Exception as e:
    print("County boundary download/geopandas failed -> dot maps will be drawn instead:", str(e)[:120])
ASPECT = 1 / np.cos(np.deg2rad(38.5))


def finish(ax, title):
    ax.set_title(title, fontsize=10)
    if COUNTIES is not None:
        ax.set_axis_off()
    else:
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect(ASPECT)


def value_map(ax, vals, cmap, vmin, vmax, title, label, outline=None, zone_lines=True):
    """Continuous county map (polygons if available, dots otherwise)."""
    vals = vals.reindex(zone.index)
    if COUNTIES is not None:
        g = COUNTIES.merge(vals.rename("v"), left_on="FIPS", right_index=True, how="left")
        g.plot(ax=ax, column="v", cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="white",
               linewidth=0.4, missing_kwds={"color": "#DDDDDD"})
        if zone_lines and ZB is not None:
            ZB.plot(ax=ax, color="black", linewidth=1.1)
        if outline is not None and len(outline):
            sel = g[g["FIPS"].isin(outline)]
            if len(sel):
                sel.boundary.plot(ax=ax, color="black", linewidth=2.2)
    else:
        ax.scatter(lonlat.loc[vals.index, "lon"], lonlat.loc[vals.index, "lat"], c=vals.values,
                   cmap=cmap, vmin=vmin, vmax=vmax, s=90, edgecolor="black")
        if outline is not None and len(outline):
            o = lonlat.loc[list(outline)]
            ax.scatter(o["lon"], o["lat"], s=260, facecolors="none", edgecolors="black", lw=2)
    plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap), ax=ax,
                 fraction=0.04, pad=0.02, label=label)
    finish(ax, title)


def zone_map(ax, zseries, title, outline=None):
    """Categorical county map coloured by zone."""
    zseries = zseries.reindex(zone.index)
    if COUNTIES is not None:
        g = COUNTIES.merge(zseries.rename("z"), left_on="FIPS", right_index=True, how="left")
        g["c"] = g["z"].map(zcol).fillna("#DDDDDD")
        g.plot(ax=ax, color=g["c"], edgecolor="white", linewidth=0.4)
        if outline is not None and len(outline):
            sel = g[g["FIPS"].isin(outline)]
            if len(sel):
                sel.boundary.plot(ax=ax, color="black", linewidth=2.2)
    else:
        ax.scatter(lonlat.loc[zseries.index, "lon"], lonlat.loc[zseries.index, "lat"],
                   c=zseries.map(zcol).values, s=90, edgecolor="black")
        if outline is not None and len(outline):
            o = lonlat.loc[list(outline)]
            ax.scatter(o["lon"], o["lat"], s=260, facecolors="none", edgecolors="black", lw=2)
    ax.legend(handles=[Patch(color=zcol[z], label=z) for z in ZONES], loc="lower left", fontsize=8,
              frameon=True)
    finish(ax, title)


# ---- m1: study area (mean yield, weather points, central point) ----
fig, ax = plt.subplots(figsize=(10, 5))
value_map(ax, cs["mean_yield"], "YlGn", cs["mean_yield"].min(), cs["mean_yield"].max(),
          "Study area: Kansas counties, mean winter wheat yield and weather input locations",
          "Mean winter wheat yield (bu/ac)", zone_lines=False)
ax.scatter(lonlat["lon"], lonlat["lat"], s=7, color="black", label="County weather points", zorder=4)
ax.scatter(CENTRAL_POINT[1], CENTRAL_POINT[0], marker="*", s=380, color="red", edgecolor="black",
           zorder=5, label="Central point (v1 comparison)")
ax.legend(loc="lower left", fontsize=8)
save("m1_study_area_yield.png")

# ---- m2: zones and migration ----
changed = list(moved[moved].index)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
zone_map(axes[0], zone, "(a) Agro-climatic zones (1982–2023 climate)")
zone_map(axes[1], mig["early"], "(b) Zone matched by 1982–2002 climate")
zone_map(axes[2], mig["late"], f"(c) Zone matched by 2003–2023 climate\n(outlined: {len(changed)} counties that changed zone)",
         outline=changed)
save("m2_zones_and_migration.png")

# ---- m3: within-county correlations ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
value_map(axes[0], cs["r_rain"], "RdBu", -0.8, 0.8, "(a) Rainfall vs yield anomaly",
          "Pearson r (blue = rain helps)")
value_map(axes[1], cs["r_heat"], "RdBu", -0.8, 0.8, "(b) Hottest temperature vs yield anomaly",
          "Pearson r (red = heat hurts)")
save("m3_county_correlations.png")

# ---- m4: change in exposure and variability ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
value_map(axes[0, 0], cs["heat_change_pp"], "RdBu_r", -40, 40,
          "(a) Heat-stress years: 2003–2023 minus 1982–2002", "Change (percentage points)")
value_map(axes[0, 1], cs["edd_change"], "RdBu_r", -60, 60,
          "(b) May–Jun heat degree-days >30 °C: change in mean", "Change (°C-days)")
value_map(axes[1, 0], cs["winterkill_change_pp"], "RdBu_r", -40, 40,
          "(c) Winterkill years: 2003–2023 minus 1982–2002", "Change (percentage points)")
value_map(axes[1, 1], cs["yield_sd_change"], "RdBu_r", -15, 15,
          "(d) Yield-anomaly SD: 2003–2023 minus 1982–2002", "Change (% points)")
save("m4_exposure_change_maps.png")

# =====================================================================
# PART 5: SHAP (random forest with zone information)
# =====================================================================
try:
    import shap
except ImportError:
    shap = None
    print("\nSHAP not installed. Run  !pip install shap -q  and re-run for the SHAP figures.")

if shap is not None:
    wlabels = [SHORT[c] for c in MONTHLY]
    Xw = df[[c + "_a" for c in MONTHLY]].copy(); Xw.columns = wlabels
    Xz = pd.get_dummies(df["Zone"]).astype(float); Xz.columns = [f"Zone {c}" for c in Xz.columns]
    X = pd.concat([Xw, Xz], axis=1)
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10,
                               random_state=42, n_jobs=-1).fit(X, df["anomaly"])
    sample = X.sample(n=min(1500, len(X)), random_state=0)
    sv = shap.TreeExplainer(rf).shap_values(sample)
    zs = df.loc[sample.index, "Zone"]
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values()
    print("\n=== SHAP mean |value| (random forest with zone indicators; % yield anomaly) ===")
    print(imp.round(2)[::-1].to_string())
    imp.to_csv(os.path.join(RES, "shap_importance_zones_rf.csv"))

    plt.figure(figsize=(7, 4.5))
    imp.plot(kind="barh", color=["#2A7AB0" if c.startswith("Zone") else "#A0522D" for c in imp.index])
    plt.xlabel("Mean |SHAP value| (% yield anomaly)")
    plt.title("Feature importance (brown = weather anomaly, blue = zone indicator)")
    save("s1_shap_bar.png")

    shap.summary_plot(sv, sample, show=False)
    plt.title("SHAP values: weather anomalies and zone indicators")
    save("s2_shap_beeswarm.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, feat in zip(axes, ["Rainfall", "Hottest temp"]):
        j = list(X.columns).index(feat)
        for z in ZONES:
            m = (zs == z).values
            ax.scatter(sample[feat].values[m], sv[m, j], s=10, alpha=0.45, color=zcol[z])
            if m.sum() > 20:
                sl, ic = np.polyfit(sample[feat].values[m], sv[m, j], 1)
                xs = np.linspace(sample[feat].values[m].min(), sample[feat].values[m].max(), 50)
                ax.plot(xs, ic + sl * xs, color=zcol[z], lw=2.4, label=z)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xlabel(f"{feat} anomaly (within-county)"); ax.set_ylabel("SHAP value (% yield anomaly)")
        ax.set_title(f"{feat}: effect on yield by zone")
    axes[0].legend(title="Zone", fontsize=8)
    save("s3_shap_rain_heat_by_zone.png")

    rows_abs, rows_dir = {}, {}
    for z in ZONES:
        m = (zs == z).values
        rows_abs[z] = {f: np.abs(sv[m, list(X.columns).index(f)]).mean() for f in wlabels}
        rows_dir[z] = {f: np.corrcoef(sample[f].values[m], sv[m, list(X.columns).index(f)])[0, 1]
                       for f in wlabels}
    A = pd.DataFrame(rows_abs).T; D = pd.DataFrame(rows_dir).T
    print("\nMean |SHAP| by zone:"); print(A.round(2).to_string())
    print("\nDirection (corr. of feature value with its SHAP value; + = higher value raises yield):")
    print(D.round(2).to_string())
    A.to_csv(os.path.join(RES, "shap_by_zone_mean_abs.csv")); D.to_csv(os.path.join(RES, "shap_by_zone_direction.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, M, t, cm, vl in zip(axes, [A, D], ["(a) Importance: mean |SHAP|", "(b) Direction of effect"],
                                ["Blues", "RdBu"], [None, (-1, 1)]):
        im = ax.imshow(M.values, cmap=cm, aspect="auto", vmin=None if vl is None else vl[0],
                       vmax=None if vl is None else vl[1])
        ax.set_xticks(range(len(M.columns))); ax.set_xticklabels(M.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index); ax.set_title(t)
        for (i, j), v in np.ndenumerate(M.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)
    save("s4_shap_by_zone_heatmaps.png")

print("\nDone. Maps (m1-m4) and SHAP figures (s1-s4) saved in figures/v4_zones/.")
