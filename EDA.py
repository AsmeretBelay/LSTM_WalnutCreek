# eda_overlay_all.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========= CONFIG =========
DATA_FILE = "lagged_features_south_state_with_drt.csv"  
TARGET_HINTS = ["target_south_state_t", "South_State", "South_State_Street", "0208735460"]

# ========= LOAD =========
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
df = df.set_index("datetime").sort_index()

# ========= DETECT COLUMNS =========
# Target
target_col = None
for hint in TARGET_HINTS:
    matches = [c for c in df.columns if hint in c]
    if matches:
        target_col = matches[0]
        break
if target_col is None:
    
    candidates = [c for c in df.columns if "_lag" not in c and "drt" not in c.lower()]
    target_col = candidates[0] if candidates else None

if target_col is None:
    raise SystemExit(" Could not find target column. Set TARGET_HINTS or specify explicitly.")

# Streams (lagged inputs): any column with '_lag' and not the target
stream_cols = [c for c in df.columns if "_lag" in c and c != target_col]

# Rain features 
rain_cols = [c for c in df.columns if "drt" in c.lower()]  # e.g., drt_so_far, drt_yesterday

print("\n=== Detected columns ===")
print("Target:", target_col)
print("Streams:", stream_cols if stream_cols else "(none found)")
print("Rain:", rain_cols if rain_cols else "(none found)")

if not stream_cols:
    print(" No stream lag columns detected. Make sure your feature builder created *_lagX columns.")

# ========= BASIC STATS =========
print("\nShape:", df.shape)
print("\nMissing values (top 20):\n", df.isna().sum().sort_values(ascending=False).head(20))
print("\nDescribe (selected):\n", df[[target_col] + stream_cols[:5]].describe())

# ========= 1) TARGET OVER TIME =========
plt.figure(figsize=(14, 4))
plt.plot(df.index, df[target_col], label=target_col, linewidth=2)
plt.title("Target over time")
plt.ylabel("Gage height"); plt.grid(True); plt.tight_layout(); plt.show()

# ========= 2) OVERLAY (STANDARDIZED) =========
plot_df = df[[target_col] + stream_cols].dropna().copy()
for c in plot_df.columns:
    mu, sd = plot_df[c].mean(), plot_df[c].std()
    if sd > 0:
        plot_df[c] = (plot_df[c] - mu) / sd

plt.figure(figsize=(14, 6))
plt.plot(plot_df.index, plot_df[target_col], label="TARGET (z)", linewidth=2)
for c in stream_cols:
    plt.plot(plot_df.index, plot_df[c], label=c, alpha=0.7)
plt.title("Target + all upstream streams (standardized)")
plt.xlabel("Date"); plt.ylabel("Z-score")
plt.legend(ncol=2, fontsize=9)
plt.grid(True); plt.tight_layout(); plt.show()

# ========= 3) RAW OVERLAY + RAIN (TWIN AXIS) =========
plt.figure(figsize=(14, 5))
ax1 = plt.gca()
ax1.plot(df.index, df[target_col], label="Target", linewidth=2)
for c in stream_cols:
    ax1.plot(df.index, df[c], label=c, alpha=0.6)
ax1.set_ylabel("Raw units"); ax1.grid(True)

# twin axis for rain (bars preferred for drt_so_far)
if rain_cols:
    ax2 = ax1.twinx()
    if "drt_so_far" in df.columns:
        ax2.bar(df.index, df["drt_so_far"], alpha=0.25, label="DRT so far", width=0.03)
    for rc in rain_cols:
        if rc != "drt_so_far":
            ax2.plot(df.index, df[rc], alpha=0.6, label=rc)
    ax2.set_ylabel("Rain (in)")
    # build a combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, ncol=2, fontsize=9, loc="upper left")
plt.title("Streams + Target (raw) with Rain (right axis)")
plt.tight_layout(); plt.show()

# ========= 4) CORRELATION HEATMAP =========
plt.figure(figsize=(10, 8))
corr = df[[target_col] + stream_cols + rain_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation heatmap")
plt.tight_layout(); plt.show()

# ========= 5) SCATTER: EACH STREAM VS TARGET =========
for c in stream_cols:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[c], df[target_col], alpha=0.5)
    plt.xlabel(c); plt.ylabel(target_col); plt.grid(True)
    plt.title(f"{c} vs {target_col}")
    plt.tight_layout(); plt.show()

# ========= 6) SMALL MULTIPLES (SUBPLOTS) =========
cols = [target_col] + stream_cols
n = len(cols)
rows = int(np.ceil(n / 2))
plt.figure(figsize=(14, 3*rows))
for i, c in enumerate(cols, 1):
    plt.subplot(rows, 2, i)
    plt.plot(df.index, df[c])
    plt.title(c); plt.grid(True)
plt.tight_layout(); plt.show()
