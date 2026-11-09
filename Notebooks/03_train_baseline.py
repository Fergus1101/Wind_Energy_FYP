import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# Step 1: Load Merged Dataset
# ---------------------------------------------------------
print("Loading merged dataset...")
df = pd.read_csv("data/merged_wind_2017.csv")

# Drop any remaining missing values if present
df = df.dropna(subset=['mean_wind_speed', 'IEWindGeneration'])

# ---------------------------------------------------------
# Step 2: Define Feature Matrix (X) and Target Vector (y)
# ---------------------------------------------------------
X = df[['mean_wind_speed']]  # Features (DataFrame)
y = df['IEWindGeneration']  # Target (Series)

# ---------------------------------------------------------
# Step 3: Train/Test Split (80% Train, 20% Test)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train):,} samples")
print(f"Test set size: {len(X_test):,} samples")

# ---------------------------------------------------------
# Step 4: Train Linear Regression Model
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------
# Step 5: Evaluate Model on Test Set
# ---------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 40)
print("LINEAR REGRESSION BASELINE RESULTS")
print("=" * 40)
print(f"Mean Absolute Error (MAE): {mae:.2f} MW")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} MW")
print(f"R² Score: {r2:.4f}")

# ---------------------------------------------------------
# Step 6: Plot Regression Line vs Actual Data
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['mean_wind_speed'], y=y_test, alpha=0.3, label='Actual Data')
plt.plot(X_test['mean_wind_speed'], y_pred, color='red', linewidth=2, label='Linear Regression Fit')

plt.title('Wind Speed vs. National Wind Generation (Test Set)')
plt.xlabel('Mean Wind Speed (knots)')
plt.ylabel('Wind Generation (MW)')
plt.legend()
plt.tight_layout()
plt.show()