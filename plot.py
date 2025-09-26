import matplotlib.pyplot as plt
import pandas as pd

# y_test = actual values
# y_pred_drt = predictions from the LSTM model with DRT feature
# time_index = timestamps for test data

# Example if stored in arrays:
# y_test: shape (n_samples,)
# y_pred_drt: shape (n_samples,)
# time_index: DatetimeIndex or list of timestamps

plt.figure(figsize=(14,6))
plt.plot(time_index, y_true, label="Actual", linewidth=2)
plt.plot(time_index, y_pred, label="Predicted (Hourly + DRT)", linewidth=2)

plt.xlabel("Time")
plt.ylabel("Gage Height (ft)")
plt.title("Actual vs Predicted Streamflow (Hourly + DRT)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
