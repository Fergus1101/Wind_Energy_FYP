import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# 1. Load Data
df = pd.read_csv("data/merged_wind_2017.csv").dropna()
X = df[["mean_wind_speed"]]
y = df["IEWindGeneration"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 2. Polynomial Regression (Degree 2 - Quadratic)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)


# 3. Gompertz Model Definition: y = a * exp(-b * exp(-c * x))
def gompertz(x, a, b, c):
    return a * np.exp(-b * np.exp(-c * x))


# Estimate initial parameters [max_capacity, shift, growth_rate]
p0 = [2500, 5, 0.2]
popt, _ = curve_fit(
    gompertz, X_train["mean_wind_speed"], y_train, p0=p0, maxfev=10000
)
y_pred_gomp = gompertz(X_test["mean_wind_speed"], *popt)

# 4. Print Results
print("=" * 55)
print("NON-LINEAR MODEL EVALUATION")
print("=" * 55)
print(
    f"Polynomial (Degree 2) -> R²: {r2_score(y_test, y_pred_poly):.4f} | MAE: {mean_absolute_error(y_test, y_pred_poly):.2f} MW"
)
print(
    f"Gompertz Sigmoidal    -> R²: {r2_score(y_test, y_pred_gomp):.4f} | MAE: {mean_absolute_error(y_test, y_pred_gomp):.2f} MW"
)
print("=" * 55)

# 5. Plot Comparison Curves
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_test["mean_wind_speed"],
    y=y_test,
    alpha=0.2,
    color="gray",
    label="Actual Data",
)

# Sort values for smooth curve plotting
x_smooth = np.linspace(X_test["mean_wind_speed"].min(), X_test["mean_wind_speed"].max(), 300)
plt.plot(
    x_smooth,
    poly_model.predict(poly.transform(pd.DataFrame(x_smooth, columns=['mean_wind_speed']))),
    color="orange",
    linewidth=2,
    label="Polynomial Fit (Degree 2)",
)
plt.plot(
    x_smooth,
    gompertz(x_smooth, *popt),
    color="purple",
    linewidth=2,
    label="Gompertz Sigmoidal Fit",
)

plt.title("Polynomial vs. Gompertz Curve Fitting on Wind Data")
plt.xlabel("Mean Wind Speed (knots)")
plt.ylabel("Wind Generation (MW)")
plt.legend()
plt.tight_layout()
plt.show()