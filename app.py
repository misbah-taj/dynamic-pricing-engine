import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load Models
# -----------------------------

random_forest = joblib.load("model/pricing_model.pkl")
xgboost_model = joblib.load("model/xgboost_model.pkl")
timeseries_model = joblib.load("model/timeseries_model.pkl")
rl_model = joblib.load("reinforcement_learning/pricing_rl_model.pkl")


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Dynamic Pricing Engine",
    page_icon="icon.png",
    layout="wide"
)

st.title("🛒 Dynamic Pricing Engine")

st.write(
    "An intelligent pricing system using Random Forest, XGBoost, "
    "Time-Series Forecasting and Reinforcement Learning."
)

st.divider()


# -----------------------------
# Product Inputs
# -----------------------------

st.subheader("📦 Product Information")

col1, col2 = st.columns(2)

with col1:
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

with col2:
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


# -----------------------------
# Prediction
# -----------------------------

if st.button("🚀 Predict Optimal Price", use_container_width=True):

    input_data = pd.DataFrame({
        "current_price": [current_price],
        "demand": [demand],
        "inventory": [inventory],
        "competitor_price": [competitor_price],
        "seasonal_factor": [seasonal_factor]
    })

    # Random Forest prediction
    rf_prediction = random_forest.predict(input_data)[0]

    # XGBoost prediction
    xgb_prediction = xgboost_model.predict(input_data)[0]

    # Combined ML prediction
    final_price = (rf_prediction + xgb_prediction) / 2

    # -----------------------------
    # Results
    # -----------------------------

    st.divider()

    st.subheader("💰 Price Prediction")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Random Forest",
            f"₹{rf_prediction:.2f}"
        )

    with result_col2:
        st.metric(
            "XGBoost",
            f"₹{xgb_prediction:.2f}"
        )

    with result_col3:
        st.metric(
            "Recommended Price",
            f"₹{final_price:.2f}"
        )

    # Price difference
    difference = final_price - current_price

    if difference > 0:
        st.success(
            f"Recommended price increase: ₹{difference:.2f}"
        )
    elif difference < 0:
        st.warning(
            f"Recommended price decrease: ₹{abs(difference):.2f}"
        )
    else:
        st.info("Recommended price is the same as the current price.")

    # -----------------------------
    # Time-Series Forecast
    # -----------------------------

    st.divider()

    st.subheader("📈 7-Day Demand Forecast")

    forecast = timeseries_model.forecast(steps=7)

    forecast_df = pd.DataFrame({
        "Day": [f"Day {i}" for i in range(1, 8)],
        "Predicted Demand": forecast.values
    })

    st.line_chart(
        forecast_df.set_index("Day")
    )

    # -----------------------------
    # RL Pricing Decision
    # -----------------------------

    st.divider()

    st.subheader("🤖 Reinforcement Learning Decision")

    action = rl_model["decision"]

    st.info(
        f"RL Pricing Recommendation: **{action}**"
    )

    # -----------------------------
    # Price Comparison Chart
    # -----------------------------

    st.divider()

    st.subheader("📊 Price Comparison")

    chart_data = pd.DataFrame({
        "Model": [
            "Current Price",
            "Random Forest",
            "XGBoost",
            "Final Price"
        ],
        "Price": [
            current_price,
            rf_prediction,
            xgb_prediction,
            final_price
        ]
    })

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.bar(
        chart_data["Model"],
        chart_data["Price"]
    )

    ax.set_ylabel("Price (₹)")
    ax.set_title("Dynamic Pricing Model Comparison")

    plt.xticks(rotation=20)

    st.pyplot(fig, width=500)


# -----------------------------
# About
# -----------------------------

st.divider()

st.subheader("ℹ️ About the Project")

st.write(
    "This Dynamic Pricing Engine combines machine learning, "
    "time-series forecasting and reinforcement learning to "
    "support intelligent pricing decisions."
)