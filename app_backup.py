import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model/pricing_model.pkl")

# Page title
st.title("🛒 Dynamic Pricing Engine")

st.write("Enter product information to predict the optimal price.")

# User inputs
current_price = st.number_input(
    "Current Price (₹)",
    min_value=1.0,
    value=500.0
)

demand = st.number_input(
    "Demand",
    min_value=0.0,
    value=80.0
)

inventory = st.number_input(
    "Inventory / Stock",
    min_value=0.0,
    value=30.0
)

competitor_price = st.number_input(
    "Competitor Price (₹)",
    min_value=1.0,
    value=520.0
)

seasonal_factor = st.number_input(
    "Seasonal Factor",
    min_value=0.1,
    value=1.2
)

# Prediction button
if st.button("Predict Optimal Price"):

    input_data = pd.DataFrame({
        "current_price": [current_price],
        "demand": [demand],
        "inventory": [inventory],
        "competitor_price": [competitor_price],
        "seasonal_factor": [seasonal_factor]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Recommended Optimal Price: ₹{prediction:.2f}")