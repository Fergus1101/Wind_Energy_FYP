import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# Step 1: Load Dataset & Feature Engineering
# ---------------------------------------------------------
df = pd.read_csv("data/merged_wind_2017.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# Extract temporal features
df["hour"] = df["datetime"].dt.hour
df["month"] = df["datetime"].dt.month
df = df.dropna()

features = ["mean_wind_speed", "mean_temperature", "mean_pressure", "hour", "month"]
X = df[features]
y = df["IEWindGeneration"]

# ---------------------------------------------------------
# Step 2: Train/Test Split (80/20)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# Step 3: Train Gradient Boosting Regressor
# ---------------------------------------------------------
print("Training Gradient Boosting model...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)

# ---------------------------------------------------------
# Step 4: Evaluate & Compare with Random Forest
# ---------------------------------------------------------
y_pred_gb = gb_model.predict(X_test)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)

# Previous Random Forest metrics for direct comparison
mae_rf = 100.87
rmse_rf = 142.35
r2_rf = 0.9435

print("\n" + "=" * 65)
print("MODEL COMPARISON: GRADIENT BOOSTING vs. RANDOM FOREST")
print("=" * 65)
print(f"Random Forest     -> R²: {r2_rf:.4f} | MAE: {mae_rf:.2f} MW | RMSE: {rmse_rf:.2f} MW")
print(f"Gradient Boosting -> R²: {r2_gb:.4f} | MAE: {mae_gb:.2f} MW | RMSE: {rmse_gb:.2f} MW")
print("=" * 65)

# ---------------------------------------------------------
# Step 5: Feature Importance Analysis
# ---------------------------------------------------------
importances = pd.Series(gb_model.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- Gradient Boosting Feature Importance ---")
for feature, importance in importances.items():
    print(f"{feature:<20}: {importance:.4f}")

# ---------------------------------------------------------
# Step 6: Plot Actual vs. Predicted
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_gb, alpha=0.3, color='crimson', label='Gradient Boosting Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction Line')

plt.title('Actual vs. Predicted Wind Generation (Gradient Boosting)')
plt.xlabel('Actual Generation (MW)')
plt.ylabel('Predicted Generation (MW)')
plt.legend()
plt.tight_layout()
plt.show()