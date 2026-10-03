# =====================================================================
# 03_east_west_analysis.py
# Exploratory analysis of the east-west gradient in county weather responses
# (manuscript Section 3.4, Table 3). Run after 01_monthly_analysis.py.
#
# Run from the repository root:   python code/03_east_west_analysis.py
# Input:  results/county_correlations.csv (written by script 01)
# Output: results/east_west_summary.csv
#
# Note: county correlations are spatially dependent (neighbouring counties
# share weather), so the p-values below overstate the effective sample size.
# =====================================================================

import pandas as pd
from scipy.stats import pearsonr

cc = pd.read_csv("results/county_correlations.csv")
SPLIT_LON = -98.5     # counties west of 98.5°W = "West"

rows = []
for col, name in [("r_rain", "Oct–Jun precipitation"), ("r_heat", "Hottest temperature"),
                  ("r_aprmin", "April minimum temperature")]:
    r, p = pearsonr(cc["INTPTLONG"], cc[col])
    west = cc.loc[cc["INTPTLONG"] < SPLIT_LON, col].median()
    east = cc.loc[cc["INTPTLONG"] >= SPLIT_LON, col].median()
    rows.append({"Weather variable": name, "Median r (West)": round(west, 2),
                 "Median r (East)": round(east, 2), "r with longitude": round(r, 2),
                 "p (nominal)": round(p, 4)})
    print(f"{col} vs longitude: r = {r:.2f}, p = {p:.4f}")

summary = pd.DataFrame(rows)
print("\n=== East-west summary (manuscript Table 3) ===")
print(summary.to_string(index=False))
n_w = (cc["INTPTLONG"] < SPLIT_LON).sum()
print(f"\nCounties: {n_w} west, {len(cc) - n_w} east of {abs(SPLIT_LON)}°W")
summary.to_csv("results/east_west_summary.csv", index=False)
