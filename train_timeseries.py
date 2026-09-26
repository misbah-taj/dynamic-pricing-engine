import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib

# Load historical demand data
data = pd.read_csv("time_series/historical_demand.csv")

# Convert date column
data["date"] = pd.to_datetime(data["date"])

# Set date as index
data = data.set_index("date")

# Get demand series
demand = data["demand"]

# Create ARIMA model
model = ARIMA(demand, order=(1, 1, 1))

# Train model
model_fit = model.fit()

# Forecast next 7 days
forecast = model_fit.forecast(steps=7)

print("Time-Series model trained successfully!")
print("\nPredicted demand for the next 7 days:")

for i, value in enumerate(forecast, start=1):
    print(f"Day {i}: {value:.2f}")

# Save model
joblib.dump(model_fit, "model/timeseries_model.pkl")

print("\nTime-Series model saved successfully!")