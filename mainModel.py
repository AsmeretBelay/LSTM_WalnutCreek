# lstm_model_south_state.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ===================== CONFIG =====================
DATA_FILE   = "lagged_features_south_state_with_drt.csv"
TARGET_COL  = "target_south_state_t"          # <- South State gage height at t
SEQ_LEN     = 24                               # hours of history per sample (try 24 or 48)
VAL_START   = "2024-07-20"                     # chronological split (adjust to your data window)
TEST_START  = "2024-08-01"                     # leave Aug+ for testing (adjust as needed)
BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 1e-3
MODEL_OUT   = "lstm_south_state.keras"
PRED_OUT    = "lstm_predictions_south_state.csv"

# ===================== LOAD =====================
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"]).set_index("datetime").sort_index()

if TARGET_COL not in df.columns:
    raise SystemExit(f"{TARGET_COL} not found in {DATA_FILE}")

# Feature set = all numeric columns except target
num_df = df.select_dtypes(include=[np.number])
feat_cols = [c for c in num_df.columns if c != TARGET_COL]
if not feat_cols:
    raise SystemExit(" No numeric features found besides target.")

print("\nUsing features:")
for c in feat_cols: print(" -", c)

# Drop rows with any NaN in features/target
data = num_df[feat_cols + [TARGET_COL]].dropna().copy()
if data.empty:
    raise SystemExit(" No rows after dropping NaNs.")

# ===================== SPLIT (time-based) =====================
train = data.loc[data.index < VAL_START]
val   = data.loc[(data.index >= VAL_START) & (data.index < TEST_START)]
test  = data.loc[data.index >= TEST_START]

for name, part in [("train", train), ("val", val), ("test", test)]:
    if len(part) == 0:
        raise SystemExit(f" {name} split empty. Adjust VAL_START/TEST_START.")
print(f"\nSplit sizes: train={len(train)}, val={len(val)}, test={len(test)}")

X_train_raw, y_train = train[feat_cols].values, train[TARGET_COL].values
X_val_raw,   y_val   = val[feat_cols].values,   val[TARGET_COL].values
X_test_raw,  y_test  = test[feat_cols].values,  test[TARGET_COL].values

# ===================== SCALE =====================
scaler = StandardScaler().fit(X_train_raw)
X_train_s = scaler.transform(X_train_raw)
X_val_s   = scaler.transform(X_val_raw)
X_test_s  = scaler.transform(X_test_raw)

# ===================== MAKE SEQUENCES =====================
def make_sequences(X2D, y1D, seq_len):
    """
    Turn (T, F) arrays into (T - seq_len + 1, seq_len, F) and align y to end of each window.
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X2D) + 1):
        X_seq.append(X2D[i - seq_len:i, :])
        y_seq.append(y1D[i - 1])  # predict value at the window end (time t)
    return np.array(X_seq), np.array(y_seq)

Xtr, ytr = make_sequences(X_train_s, y_train, SEQ_LEN)
Xva, yva = make_sequences(X_val_s,   y_val,   SEQ_LEN)
Xte, yte = make_sequences(X_test_s,  y_test,  SEQ_LEN)

print(f"\nShapes -> Xtr:{Xtr.shape}, Xva:{Xva.shape}, Xte:{Xte.shape}  (seq_len={SEQ_LEN}, features={Xtr.shape[-1]})")

# ===================== MODEL =====================
tf.keras.utils.set_random_seed(42)

model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, Xtr.shape[-1]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.1),
    Dense(1)
])

opt = tf.keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=opt, loss="mse")
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=1),
    ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True, verbose=1),
]

# ===================== TRAIN =====================
history = model.fit(
    Xtr, ytr,
    validation_data=(Xva, yva),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=callbacks
)

# ===================== EVALUATE =====================
def eval_metrics(y_true, y_pred, label=""):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{label} RMSE={rmse:.3f}  MAE={mae:.3f}  R^2={r2:.3f}")
    return rmse, mae, r2

ytr_pred = model.predict(Xtr, verbose=0).ravel()
yva_pred = model.predict(Xva, verbose=0).ravel()
yte_pred = model.predict(Xte, verbose=0).ravel()

print("\n=== Metrics ===")
eval_metrics(ytr, ytr_pred, "Train")
eval_metrics(yva, yva_pred, "Val  ")
eval_metrics(yte, yte_pred, "Test ")

# ===================== SAVE PREDICTIONS =====================
# Align sequence predictions back to index timestamps
test_idx = test.index[SEQ_LEN-1:]  # align to end of windows
pred_df = pd.DataFrame({"datetime": test_idx, "y_true": yte, "y_pred": yte_pred}).set_index("datetime")
pred_df.to_csv(PRED_OUT)
print(f"\n Saved predictions: {PRED_OUT}")
print(f" Saved model: {MODEL_OUT}")

# ===================== PLOTS =====================
# plt.figure(figsize=(14,5))
# plt.plot(pred_df.index, pred_df["y_true"], label="Actual", linewidth=2)
# plt.plot(pred_df.index, pred_df["y_pred"], label="Predicted", alpha=0.9)
# plt.title(f"LSTM — South State (SEQ_LEN={SEQ_LEN})")
# plt.xlabel("Date"); plt.ylabel("Gage height (ft)")
# plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# # Residuals
# res = pred_df["y_true"] - pred_df["y_pred"]
# plt.figure(figsize=(14,3.5))
# plt.plot(pred_df.index, res); plt.axhline(0, color="k", linewidth=1)
# plt.title("Residuals (Actual - Predicted) — Test")
# plt.xlabel("Date"); plt.grid(True); plt.tight_layout(); plt.show()
