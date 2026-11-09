import glob
import pandas as pd


def parse_met_eireann_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Step 1: Extract Station Name from Line 1 (e.g. "Station Name: CLAREMORRIS")
    station_name = "UNKNOWN"
    if len(lines) > 0 and "Station Name:" in lines[0]:
        station_name = lines[0].split("Station Name:")[1].strip()

    # Step 2: Find the exact row index where the table headers start
    header_row_index = None
    for idx, line in enumerate(lines):
        line_clean = line.strip().lower()
        # Look for the row containing the main column headers
        if line_clean.startswith("date") and "temp" in line_clean:
            header_row_index = idx
            break

    if header_row_index is None:
        print(
            f"⚠️ Skipping {filepath}: Could not locate table header row."
        )
        return None

    # Step 3: Load CSV starting from the detected header line
    df = pd.read_csv(filepath, skiprows=header_row_index)

    # Clean up column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Step 4: Verify if wind speed ('wdsp') exists in this station
    if "wdsp" not in df.columns:
        print(
            f"ℹ️ Skipping {station_name}: No 'wdsp' (wind speed) sensor at this station."
        )
        return None

    # Attach station name as a column
    df["station"] = station_name

    # Keep only the essential columns needed for modeling
    cols_to_keep = ["station", "date", "temp", "wdsp", "msl"]
    cols_to_keep = [col for col in cols_to_keep if col in df.columns]

    return df[cols_to_keep]


# ---------------------------------------------------------
# Execute parsing on all 2025 Met Éireann CSVs
# ---------------------------------------------------------
weather_files = glob.glob("data/raw_weather_2025/*.csv")
print(f"Found {len(weather_files)} weather files. Processing...")

parsed_dfs = []
for file in weather_files:
    df = parse_met_eireann_file(file)
    if df is not None:
        parsed_dfs.append(df)

if parsed_dfs:
    combined_weather = pd.concat(parsed_dfs, ignore_index=True)
    output_path = "data/weather_2025_full.csv"
    combined_weather.to_csv(output_path, index=False)
    print(f"\n✅ Successfully combined {len(parsed_dfs)} valid wind stations into {output_path}!")