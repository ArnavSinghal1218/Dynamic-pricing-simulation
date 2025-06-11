import numpy as np
import pandas as pd

np.random.seed(42)
n = 10000

# Simulated numerical features
time_of_day = np.random.randint(0, 24, n)
distance_km = np.round(np.random.exponential(scale=3.0, size=n), 2)
weather_score = np.clip(np.random.normal(loc=0.5, scale=0.2, size=n), 0, 1)

# Simulated categorical features
demand_level = np.random.choice(['low', 'medium', 'high'], n, p=[0.3, 0.5, 0.2])
area_type = np.random.choice(['residential', 'business', 'suburban', 'rural'], n, p=[0.4, 0.3, 0.2, 0.1])
weather_condition = np.random.choice(['clear', 'rainy', 'stormy', 'snowy'], n, p=[0.6, 0.2, 0.1, 0.1])
vehicle_type = np.random.choice(['bike', 'scooter', 'car'], n, p=[0.5, 0.3, 0.2])
day_of_week = np.random.choice(['weekday', 'weekend'], n, p=[0.7, 0.3])

# Demand encoding
demand_map = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
demand_encoded = np.array([demand_map[d] for d in demand_level])

# Base price formula
base_price = 3 + 0.7 * distance_km + 0.3 * demand_encoded + 2 * weather_score + 0.1 * time_of_day
noise = np.random.normal(0, 2, n)
price = np.round(base_price + noise, 2)

# Combine into DataFrame
df = pd.DataFrame({
    'time_of_day': time_of_day,
    'distance_km': distance_km,
    'weather_score': weather_score,
    'demand_level': demand_level,
    'area_type': area_type,
    'weather_condition': weather_condition,
    'vehicle_type': vehicle_type,
    'day_of_week': day_of_week,
    'price': price
})

# Save
df.to_csv('delivery_pricing.csv', index=False)
print("✅ Data saved to 'delivery_pricing.csv'")
