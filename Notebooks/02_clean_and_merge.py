import pandas as pd

# ---------------------------------------------------------
# Step 1: Load Raw Datasets
# ---------------------------------------------------------
print("Loading raw datasets...")
weather_df = pd.read_csv("data/weather_2017.csv")
eirgrid_df = pd.read_csv("data/eirgrid_2017.csv")

# ---------------------------------------------------------
# Step 2: Convert Dates to Datetime Objects (Handling Mixed Formats)
# ---------------------------------------------------------
print("Parsing dates...")
weather_df['datetime'] = pd.to_datetime(weather_df['date'], format='mixed')
eirgrid_df['datetime'] = pd.to_datetime(eirgrid_df['date'], format='mixed')

# ---------------------------------------------------------
# Step 3: Handle Missing Weather Values
# ---------------------------------------------------------
# Forward-fill missing wind speed values within weather stations
weather_df['wdsp'] = weather_df.groupby('station')['wdsp'].transform(lambda x: x.ffill().bfill())

# ---------------------------------------------------------
# Step 4: Aggregate Weather Data (National Hourly Average)
# ---------------------------------------------------------
print("Aggregating weather data...")
hourly_weather = weather_df.groupby('datetime').agg({
    'wdsp': 'mean',
    'temp': 'mean',
    'msl': 'mean'
}).reset_index()

hourly_weather.rename(columns={
    'wdsp': 'mean_wind_speed',
    'temp': 'mean_temperature',
    'msl': 'mean_pressure'
}, inplace=True)

# ---------------------------------------------------------
# Step 5: Downsample EirGrid Data from 15-Min to Hourly
# ---------------------------------------------------------
print("Downsampling EirGrid data...")
eirgrid_df['datetime'] = eirgrid_df['datetime'].dt.floor('h')

hourly_eirgrid = eirgrid_df.groupby('datetime').agg({
    'IEWindGeneration': 'mean',
    'IEDemand': 'mean'
}).reset_index()

# ---------------------------------------------------------
# Step 6: Merge Weather and EirGrid on Datetime
# ---------------------------------------------------------
print("Merging datasets...")
merged_df = pd.merge(hourly_weather, hourly_eirgrid, on='datetime', how='inner')

# Print summary
print("\n--- Merged Dataset Summary ---")
print(f"Shape: {merged_df.shape[0]:,} rows, {merged_df.shape[1]} columns")
print("\nFirst 5 Rows:")
print(merged_df.head())

# ---------------------------------------------------------
# Step 7: Save Cleaned Dataset
# ---------------------------------------------------------
output_path = "data/merged_wind_2017.csv"
merged_df.to_csv(output_path, index=False)
print(f"\nSuccessfully saved merged dataset to: {output_path}")