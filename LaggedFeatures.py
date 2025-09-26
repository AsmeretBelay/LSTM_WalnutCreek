import json
import pandas as pd

INPUT_FILE = "combined_usgs_hourly.csv"
LAGS_FILE  = "best_lags.json"
TARGET_COL = "Walnut_Creek_at_South_State_Street_0208735460"  # 

df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"]).set_index("datetime")
df = df.loc[:, ~df.columns.duplicated()]

with open(LAGS_FILE) as f:
    best_lags = json.load(f)

final_df = pd.DataFrame(index=df.index)

# 1) add lagged inputs using accurate lags
for col, lag in best_lags.items():
    if col in df.columns:
        final_df[f"{col}_lag{lag}"] = df[col].shift(int(lag))

# 2) target at time t
final_df["target_south_state_t"] = df[TARGET_COL]

# 3) (optional but recommended) South State persistence: t-1..t-6
for k in range(1, 7):
    final_df[f"{TARGET_COL}_lag{k}"] = df[TARGET_COL].shift(k)

# 4) clean & save
final_df = final_df.dropna()
final_df.to_csv("lagged_features_south_state.csv")
print("saved lagged_features_south_state.csv")
print(final_df.head())
