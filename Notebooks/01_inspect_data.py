import pandas as pd

# ---------------------------------------------------------
# Step 1: Load the CSV files into pandas DataFrames
# ---------------------------------------------------------
print("Loading CSV files...")
weather_df = pd.read_csv("data/weather_2017.csv")
eirgrid_df = pd.read_csv("data/eirgrid_2017.csv")

# ---------------------------------------------------------
# Step 2: Check dataset dimensions (rows, columns)
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("1. DATASET DIMENSIONS")
print("=" * 50)
print(f"Weather dataset shape: {weather_df.shape[0]:,} rows, {weather_df.shape[1]} columns")
print(f"EirGrid dataset shape: {eirgrid_df.shape[0]:,} rows, {eirgrid_df.shape[1]} columns")

# ---------------------------------------------------------
# Step 3: Preview the first few rows of each dataset
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("2. WEATHER DATASET SAMPLE (First 5 rows)")
print("=" * 50)
print(weather_df.head())

print("\n" + "=" * 50)
print("3. EIRGRID DATASET SAMPLE (First 5 rows)")
print("=" * 50)
print(eirgrid_df.head())

# ---------------------------------------------------------
# Step 4: Inspect column names and data types
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("4. WEATHER DATA TYPES & NON-NULL COUNTS")
print("=" * 50)
print(weather_df.info())

print("\n" + "=" * 50)
print("5. EIRGRID DATA TYPES & NON-NULL COUNTS")
print("=" * 50)
print(eirgrid_df.info())

# ---------------------------------------------------------
# Step 5: Check for missing values (NaNs)
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("6. MISSING VALUES COUNT (Weather)")
print("=" * 50)
print(weather_df.isna().sum())

# ---------------------------------------------------------
# Step 6: Numerical Summary Statistics
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("7. WEATHER NUMERICAL SUMMARY (Wind Speed, Temp, etc.)")
print("=" * 50)
print(weather_df.describe().T[['mean', 'std', 'min', '50%', 'max']])