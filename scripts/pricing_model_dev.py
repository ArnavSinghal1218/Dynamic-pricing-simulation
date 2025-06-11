import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('../data/delivery_pricing.csv')

# Inspect the data
print(df.head())
print(df.info())

# Encode categorical variables
categorical_cols = ['demand_level', 'area_type', 'weather_condition', 'vehicle_type', 'day_of_week']
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    df.drop(columns=col, inplace=True)

# Define features and target
X = df.drop(columns='price')
y = df['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation Metrics:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}")

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values().plot(kind='barh', figsize=(8, 5), title='Feature Importances')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('../scripts/feature_importance.png')
plt.show()

# Residuals
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residual (Actual - Predicted)")
plt.savefig('../scripts/residuals.png')
plt.show()

# Optional: Visualize actual vs predicted by time_of_day
X_test_copy = X_test.copy()
X_test_copy['actual_price'] = y_test
X_test_copy['predicted_price'] = y_pred

X_test_copy.sort_values(by='time_of_day', inplace=True)
sns.lineplot(data=X_test_copy, x='time_of_day', y='actual_price', label='Actual')
sns.lineplot(data=X_test_copy, x='time_of_day', y='predicted_price', label='Predicted')
plt.title("Price vs Time of Day")
plt.xlabel("Hour")
plt.ylabel("Price")
plt.savefig('../scripts/price_by_hour.png')
plt.show()

# Save model
joblib.dump(model, '../scripts/model.pkl')
print("✅ Model saved to '../scripts/model.pkl'")
