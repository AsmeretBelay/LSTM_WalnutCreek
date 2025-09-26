import json
import numpy as np
import pandas as pd

INPUT_FILE = "combined_usgs_hourly.csv"
TARGET_COL = "Walnut_Creek_at_South_State_Street_0208735460"  #  gage height target
MAX_LAG_H = 48  # test up to 2 days; increase to 72/120 

df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"]).set_index("datetime")
df = df.loc[:, ~df.columns.duplicated()]
df = df.dropna(subset=[TARGET_COL])

def best_lag(series_in: pd.Series, target: pd.Series, max_lag: int) -> tuple[int, float]:
    best = (1, -np.inf)
    for lag in range(1, max_lag + 1):
        shifted = series_in.shift(lag)
        valid = target.notna() & shifted.notna()
        corr = target[valid].corr(shifted[valid])
        if corr > best[1]:
            best = (lag, corr)
    return best

results = []
best_lags = {}
for col in df.columns:
    if col == TARGET_COL or df[col].isna().all():
        continue
    lag, corr = best_lag(df[col], df[TARGET_COL], MAX_LAG_H)
    best_lags[col] = lag
    results.append((col, lag, corr))

res_df = pd.DataFrame(results, columns=["Input", "Best Lag (h)", "Max Corr"]).sort_values("Best Lag (h)")
res_df.to_csv("lag_analysis_results.csv", index=False)
with open("best_lags.json","w") as f:
    json.dump(best_lags, f, indent=2)

print(res_df)
print(" Saved best_lags.json and lag_analysis_results.csv")
