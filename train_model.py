import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load the dataset
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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate error
mae = mean_absolute_error(y_test, predictions)

print("Model trained successfully!")
print("Mean Absolute Error:", mae)

# Save the model
joblib.dump(model, "model/pricing_model.pkl")

print("Model saved successfully!")