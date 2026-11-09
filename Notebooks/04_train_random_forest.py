import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# Step 1: Load Data & Create Feature Set
# ---------------------------------------------------------
df = pd.read_csv("data/merged_wind_2017.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract time features (wind generation follows daily & seasonal patterns)
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month

df = df.dropna()

# Select multiple input features
features = ['mean_wind_speed', 'mean_temperature', 'mean_pressure', 'hour', 'month']
X = df[features]
y = df['IEWindGeneration']

# ---------------------------------------------------------
# Step 2: Train/Test Split (Matching Baseline 80/20)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# Step 3: Train Random Forest Regressor
# ---------------------------------------------------------
print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ---------------------------------------------------------
# Step 4: Evaluate Model
# ---------------------------------------------------------
y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\n" + "=" * 50)
print("MODEL COMPARISON RESULTS")
print("=" * 50)
print(f"Linear Baseline MAE: 186.67 MW | Random Forest MAE: {mae_rf:.2f} MW")
print(f"Linear Baseline RMSE: 250.90 MW | Random Forest RMSE: {rmse_rf:.2f} MW")
print(f"Linear Baseline R²:   0.8246    | Random Forest R²:   {r2_rf:.4f}")

# ---------------------------------------------------------
# Step 5: Explanatory Model Analysis (Feature Importance)
# ---------------------------------------------------------
# Required for FYP meeting notes: identify which variables matter most
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

print("\n--- Feature Importance Ranking ---")
for feature, importance in importances.items():
    print(f"{feature:<20}: {importance:.4f}")

# ---------------------------------------------------------
# Step 6: Plot Predictions vs Actual Values
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.3, color='forestgreen', label='Random Forest Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')

plt.title('Actual vs. Predicted Wind Generation (Random Forest)')
plt.xlabel('Actual Generation (MW)')
plt.ylabel('Predicted Generation (MW)')
plt.legend()
plt.tight_layout()
plt.show()