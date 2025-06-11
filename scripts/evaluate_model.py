import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('../data/delivery_pricing.csv')

# Encode categorical features
categorical_cols = ['demand_level', 'area_type', 'weather_condition', 'vehicle_type', 'day_of_week']
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    df.drop(columns=col, inplace=True)

# Features and target
X = df.drop(columns='price')
y = df['price']

# Train/test split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
model = joblib.load('model.pkl')

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Residuals
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.savefig('residuals_eval.png')
plt.show()

# Actual vs Predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.title("Actual vs. Predicted Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.savefig('actual_vs_predicted.png')
plt.tight_layout()
plt.show()
