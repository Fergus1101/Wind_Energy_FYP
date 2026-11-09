import pandas as pd
import numpy as np

print("Loading 2025 master CSV files...")

# ---------------------------------------------------------
# Step 1: Load Raw 2025 Combined Files
# ---------------------------------------------------------
weather_df = pd.read_csv("data/weather_2025_full.csv", low_memory=False)
eirgrid_df = pd.read_csv("data/eirgrid_2025_full.csv", low_memory=False)

# Convert numeric weather columns, coercing invalid spaces/text to NaN
for col in ["wdsp", "temp", "msl"]:
    if col in weather_df.columns:
        weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")

# Parse date strings into datetime objects
weather_df["datetime"] = pd.to_datetime(weather_df["date"], format="mixed")
eirgrid_df["datetime"] = pd.to_datetime(
    eirgrid_df["DATE & TIME"], format="mixed"
)

# ---------------------------------------------------------
# Step 2: Clean Missing Weather Readings
# ---------------------------------------------------------
# Forward-fill / backward-fill missing values per station
for col in ["wdsp", "temp", "msl"]:
    if col in weather_df.columns:
        weather_df[col] = weather_df.groupby("station")[col].transform(
            lambda x: x.ffill().bfill()
        )

# ---------------------------------------------------------
# Step 3: National Hourly Weather Aggregation
# ---------------------------------------------------------
# Average readings across all active synoptic stations for each hour
hourly_weather = (
    weather_df.groupby("datetime")
    .agg(
        mean_wind_speed=("wdsp", "mean"),
        mean_temperature=("temp", "mean"),
        mean_pressure=("msl", "mean"),
    )
    .reset_index()
)

# ---------------------------------------------------------
# Step 4: Downsample EirGrid 15-Minute Readings to Hourly
# ---------------------------------------------------------
# Floor 15-minute timestamps to the start of each hour
eirgrid_df["datetime"] = eirgrid_df["datetime"].dt.floor("h")

# Find actual wind column name (ACTUAL WIND(MW))
grid_col = [c for c in eirgrid_df.columns if "ACTUAL" in c.upper()][0]
eirgrid_df[grid_col] = pd.to_numeric(eirgrid_df[grid_col], errors="coerce")

hourly_eirgrid = (
    eirgrid_df.groupby("datetime")
    .agg(IEWindGeneration=(grid_col, "mean"))
    .reset_index()
)

# ---------------------------------------------------------
# Step 5: Merge Datasets & Save
# ---------------------------------------------------------
merged_2025 = pd.merge(
    hourly_weather, hourly_eirgrid, on="datetime", how="inner"
).dropna()

output_path = "data/merged_wind_2025.csv"
merged_2025.to_csv(output_path, index=False)

print("\n" + "=" * 55)
print("2025 DATA CLEANING & MERGING COMPLETE")
print("=" * 55)
print(f"Total merged hourly rows : {len(merged_2025):,}")
print(
    f"Date range               : {merged_2025['datetime'].min()} to {merged_2025['datetime'].max()}"
)
print(f"Saved dataset to         : {output_path}")
print("=" * 55)