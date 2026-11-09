import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("Evaluating Random Forest directly on 2025 Data (80/20 Train/Test Split)...")

# 1. Load 2025 Data
df_2025 = pd.read_csv("data/merged_wind_2025.csv").dropna()
df_2025["datetime"] = pd.to_datetime(df_2025["datetime"])
df_2025["hour"] = df_2025["datetime"].dt.hour
df_2025["month"] = df_2025["datetime"].dt.month

features = ["mean_wind_speed", "mean_temperature", "mean_pressure", "hour", "month"]
X = df_2025[features]
y = df_2025["IEWindGeneration"]

# 2. Train/Test Split on 2025 Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Fit Random Forest
rf_2025 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_2025.fit(X_train, y_train)

# 4. Evaluate
y_pred = rf_2025.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "=" * 55)
print("INTERNAL 2025 MODEL RESULTS (80/20 Split)")
print("=" * 55)
print(f"2025 R² Score : {r2:.4f}")
print(f"2025 MAE      : {mae:.2f} MW")
print("=" * 55)