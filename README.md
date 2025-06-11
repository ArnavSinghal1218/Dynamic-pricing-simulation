# 🛒 Dynamic Pricing Optimization for Delivery Services

**Predicting real-time delivery prices based on real-world conditions using machine learning.**

---

## 📘 Project Summary

This project simulates and models a dynamic pricing system used by food delivery platforms (like Delivery Hero or Uber Eats). It predicts optimal delivery prices based on multiple contextual features using a machine learning regression model.

---

## 🎯 Objective

Enable businesses to estimate delivery prices by considering factors such as:

- Time of day
- Distance
- Demand level
- Weather conditions
- Vehicle type
- Delivery area
- Day of the week

---

## 🧱 Project Structure

```
dynamic-pricing-simulation/
│
├── data/
│   └── generate_data.py
│   └── delivery_pricing.csv
├── notebooks/
│   └── pricing_model_dev.ipynb
├── scripts/
│   └── pricing_model_dev.py
│   └── evaluate_model.py
│   └── model.pkl
├── dashboard/
│   └── app.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧪 Dataset

The dataset is synthetically generated using `generate_data.py`. It contains:

- `time_of_day`: Hour of delivery (0–23)
- `distance_km`: Distance in kilometers
- `weather_score`: Numeric weather impact score
- `demand_level`: ['low', 'medium', 'high']
- `area_type`: ['residential', 'business', 'suburban', 'rural']
- `weather_condition`: ['clear', 'rainy', 'stormy', 'snowy']
- `vehicle_type`: ['bike', 'scooter', 'car']
- `day_of_week`: ['weekday', 'weekend']
- `price`: Simulated final delivery price

---

## 🧠 Model Details

- Model: `RandomForestRegressor`
- Feature encoding: LabelEncoding
- Metrics:
  - RMSE
  - MAE
  - R² Score
- Visuals:
  - Feature importance
  - Residual histogram
  - Actual vs. Predicted scatter plot

---

## 📊 Streamlit Dashboard

Launch the dashboard with:

```bash
streamlit run dashboard/app.py
```

Users can input different delivery conditions and instantly receive a price estimate.

---

## 🐳 Docker Support

To run using Docker:

```bash
docker build -t delivery-pricing-app .
docker run -p 8501:8501 delivery-pricing-app
```

---

## 🛠 Tech Stack

- Python
- pandas, numpy, scikit-learn, seaborn, matplotlib
- Streamlit
- Docker
- joblib (for model serialization)

---

## 📈 Future Improvements

- Use real-world data (e.g., Uber, NYC taxi)
- Add surge pricing logic
- Integrate API with FastAPI
- A/B testing for price strategies

---

## 👤 Author

**Arnav Singhal**  
[LinkedIn](https://www.linkedin.com/in/arnav-singhal-5a34471b0) | [GitHub](https://github.com/ArnavSinghal1218)

---
