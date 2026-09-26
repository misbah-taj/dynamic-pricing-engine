import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv("data/pricing_data.csv")

# Features
X = data[
    [
        "current_price",
        "demand",
        "inventory",
        "competitor_price",
        "seasonal_factor"
    ]
]

# Target
y = data["optimal_price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create XGBoost model
model = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Calculate error
mae = mean_absolute_error(y_test, predictions)

print("XGBoost model trained successfully!")
print("Mean Absolute Error:", mae)

# Save model
joblib.dump(model, "model/xgboost_model.pkl")

print("XGBoost model saved successfully!")