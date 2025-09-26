# import os
# import pandas as pd

# RAIN_DIR = "usgs_data"
# # exact filename created by script 1:
# RAIN_FILE = "Rain_Gage_at_Lake_Johnson_354546078422045.csv"

# path = os.path.join(RAIN_DIR, RAIN_FILE)
# df = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")
# df = df.rename(columns={"value":"lake_johnson_rain"})

# # Daily total from (likely) sub-hourly/hourly precip
# drt = df["lake_johnson_rain"].resample("1D").sum().to_frame("drt")

# drt.to_csv("daily_total_rainfall_drt.csv")
# print(" saved daily_total_rainfall_drt.csv")
# print(drt.head())
import pandas as pd

lagged = pd.read_csv("lagged_features_south_state.csv", parse_dates=["datetime"]).set_index("datetime")
drt = pd.read_csv("daily_total_rainfall_drt.csv", parse_dates=["datetime"]).set_index("datetime")

# align to hourly index by ffill
drt_hourly = drt.reindex(lagged.index.date, method="ffill")
drt_hourly.index = lagged.index

lagged["drt"] = drt_hourly["drt"]
lagged.reset_index().to_csv("lagged_features_south_state_with_drt.csv", index=False)

print(" saved lagged_features_south_state_with_drt.csv")
print(lagged.head())
