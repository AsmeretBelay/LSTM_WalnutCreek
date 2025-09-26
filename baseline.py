# baseline_model_south_state.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========= CONFIG =========
DATA_FILE  = "lagged_features_south_state_with_drt.csv"  # produced earlier
TARGET_COL = "target_south_state_t"
TEST_START = "2024-08-01"                   # <-- set your test split date
RIDGE_ALPHA = 1.0                           # regularization strength

# ========= LOAD =========
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"]).set_index("datetime").sort_index()

# pick features: all lagged streams + rain features if present
stream_cols = [c for c in df.columns if "_lag" in c and c != TARGET_COL]
rain_cols   = [c for c in ["drt_yesterday", "drt_so_far"] if c in df.columns]
feat_cols   = stream_cols + rain_cols

if not feat_cols:
    raise SystemExit(" No features found. Ensure your file has *_lagX columns and/or drt features.")

print(f"\nUsing {len(feat_cols)} features:")
for c in feat_cols:
    print(" -", c)

# drop rows with NaN (from shifting / rain alignment)
data = df[feat_cols + [TARGET_COL]].dropna().copy()
if data.empty:
    raise SystemExit("After dropping NaNs, no rows left. Check your inputs/merges.")

# ========= TIME SPLIT =========
train = data.loc[data.index < TEST_START]
test  = data.loc[data.index >= TEST_START]

if len(train) == 0 or len(test) == 0:
    raise SystemExit(f" Train or test split empty. Adjust TEST_START (currently {TEST_START}).")

X_train, y_train = train[feat_cols].values, train[TARGET_COL].values
X_test,  y_test  = test[feat_cols].values,  test[TARGET_COL].values

# ========= SCALE =========
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# ========= MODEL =========
model = Ridge(alpha=RIDGE_ALPHA)  # no random_state to keep older versions happy
model.fit(X_train_s, y_train)

y_pred_train = model.predict(X_train_s)
y_pred_test  = model.predict(X_test_s)

# ========= METRICS =========
def metrics(y_true, y_pred, tag=""):
    mse  = mean_squared_error(y_true, y_pred)   # older sklearn: no 'squared' kw
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{tag} RMSE={rmse:.3f}  MAE={mae:.3f}  R^2={r2:.3f}")
    return rmse, mae, r2

print("\n=== Metrics ===")
metrics(y_train, y_pred_train, "Train")
metrics(y_test,  y_pred_test,  "Test ")

# ========= SAVE PREDICTIONS =========
out = pd.DataFrame({
    "datetime": test.index,
    "y_true": y_test,
    "y_pred": y_pred_test
}).set_index("datetime")
out.to_csv("baseline_predictions_south_state.csv")
print("\n Saved predictions: baseline_predictions_south_state.csv")

# ========= PLOTS =========
# Actual vs Predicted (Test)
plt.figure(figsize=(14,5))
plt.plot(out.index, out["y_true"], label="Actual", linewidth=2)
plt.plot(out.index, out["y_pred"], label="Predicted", alpha=0.9)
plt.title("South State — Actual vs Predicted (Test)")
plt.xlabel("Date"); plt.ylabel("Gage height")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# Residuals over time (Test)
res = out["y_true"] - out["y_pred"]
plt.figure(figsize=(14,3.5))
plt.plot(out.index, res)
plt.axhline(0, color="k", linewidth=1)
plt.title("Residuals (Actual - Predicted) — Test")
plt.xlabel("Date"); plt.grid(True); plt.tight_layout(); plt.show()

# Feature coefficients (interpretability)
coef = pd.Series(model.coef_, index=feat_cols).sort_values(key=np.abs, ascending=False)
print("\nTop coefficients (by absolute value):")
print(coef.head(12))

# Optional bar plot of coefficients if seaborn available
try:
    import seaborn as sns
    plt.figure(figsize=(8, max(3, len(coef)*0.25)))
    sns.barplot(x=coef.values, y=coef.index, orient="h")
    plt.title("Ridge coefficients")
    plt.tight_layout(); plt.show()
except Exception:
    pass
