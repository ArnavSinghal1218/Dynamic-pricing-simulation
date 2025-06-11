import streamlit as st
import joblib
import numpy as np
import os
print("Current Working Directory:", os.getcwd())


# Load model
model = joblib.load('D:/dynamic-pricing-simulation-updated/scripts/model.pkl')

# Mapping for categorical features
demand_map = {'low': 0, 'medium': 1, 'high': 2}
area_map = {'residential': 0, 'business': 1, 'suburban': 2, 'rural': 3}
weather_map = {'clear': 0, 'rainy': 1, 'stormy': 2, 'snowy': 3}
vehicle_map = {'bike': 0, 'scooter': 1, 'car': 2}
day_map = {'weekday': 0, 'weekend': 1}

# Sidebar inputs
st.title("📦 Delivery Pricing Predictor")
st.write("Estimate dynamic delivery fees based on various real-world parameters.")

st.sidebar.header("📋 Delivery Inputs")
time_of_day = st.sidebar.slider("Time of Day", 0, 23, 12)
distance_km = st.sidebar.slider("Distance (km)", 0.0, 10.0, 2.5)
weather_score = st.sidebar.slider("Weather Score (0–1)", 0.0, 1.0, 0.5)

demand_level = st.sidebar.selectbox("Demand Level", list(demand_map.keys()))
area_type = st.sidebar.selectbox("Area Type", list(area_map.keys()))
weather_condition = st.sidebar.selectbox("Weather Condition", list(weather_map.keys()))
vehicle_type = st.sidebar.selectbox("Vehicle Type", list(vehicle_map.keys()))
day_of_week = st.sidebar.selectbox("Day of Week", list(day_map.keys()))

# Prepare input for prediction
features = np.array([[
    time_of_day,
    distance_km,
    weather_score,
    demand_map[demand_level],
    area_map[area_type],
    weather_map[weather_condition],
    vehicle_map[vehicle_type],
    day_map[day_of_week]
]])

# Prediction
predicted_price = model.predict(features)[0]

st.subheader("💰 Predicted Delivery Price")
st.success(f"€{predicted_price:.2f}")
