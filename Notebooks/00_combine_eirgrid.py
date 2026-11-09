import glob
import pandas as pd

# 1. Find all CSV files in the folder
csv_files = glob.glob("data/raw_eirgrid_2025/*.csv")

print(f"Found {len(csv_files)} EirGrid files. Combining...")

# 2. Read each CSV and stack them into a single DataFrame
df_list = [pd.read_csv(file) for file in csv_files]
eirgrid_combined = pd.concat(df_list, ignore_index=True)

# 3. Clean column names & drop duplicate timestamp rows
# (EirGrid monthly exports often overlap by a day or two)
eirgrid_combined = eirgrid_combined.drop_duplicates(subset=["DATE & TIME"])

# 4. Sort chronologically by date
eirgrid_combined["datetime"] = pd.to_datetime(
    eirgrid_combined["DATE & TIME"], format="mixed"
)
eirgrid_combined = eirgrid_combined.sort_values("datetime").drop(
    columns=["datetime"]
)

# 5. Save the combined master file
output_path = "data/eirgrid_2025_full.csv"
eirgrid_combined.to_csv(output_path, index=False)

print(f"Successfully combined {len(eirgrid_combined):,} rows into {output_path}!")