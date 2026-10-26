
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path.home() / "Downloads/projects/nsef"
INPUT_DIR = BASE_DIR / "era5_monthly"
OUTPUT_DIR = BASE_DIR / "formatted_era5"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "era5_weather_cleaned.csv"


def find_netcdf_files(folder):
    """Find NetCDF files, including files in subfolders."""
    files = list(folder.rglob("*.nc"))
    files += list(folder.rglob("*.nc4"))
    files += list(folder.rglob("*.netcdf"))
    return sorted(set(files))


def main():
    files = find_netcdf_files(INPUT_DIR)

    if not files:
        print(f"No NetCDF files found in: {INPUT_DIR}")
        print("Check that the .nc files are extracted into era5_monthly.")
        print("Files currently in the folder:")
        if INPUT_DIR.exists():
            for path in INPUT_DIR.iterdir():
                print(" -", path.name)
        else:
            print("The era5_monthly folder does not exist.")
        return

    print(f"Found {len(files)} NetCDF file(s).")
    dataframes = []

    for file in files:
        print(f"\nReading: {file.name}")

        try:
            with xr.open_dataset(file) as ds:
                df = ds.to_dataframe().reset_index()

            # Standardise column names.
            df.columns = [
                str(c).strip().lower().replace(" ", "_")
                for c in df.columns
            ]

            # Remove duplicate records within this file.
            df = df.drop_duplicates()

            # Convert Kelvin to Celsius for temperature variables.
            for col in df.columns:
                if col in (
                    "t2m",
                    "d2m",
                    "skt",
                    "stl1",
                    "stl2",
                    "stl3",
                    "stl4",
                ):
                    df[col] = df[col] - 273.15

            # Convert precipitation from metres to millimetres.
            if "tp" in df.columns:
                df["tp"] = df["tp"] * 1000

            # Keep the source filename for traceability.
            df["source_file"] = file.name

            dataframes.append(df)
            print(f"Loaded {len(df):,} rows.")

        except Exception as error:
            print(f"Could not read {file.name}: {error}")

    if not dataframes:
        print("No files could be processed.")
        return

    result = pd.concat(dataframes, ignore_index=True)

    # Remove duplicate records across files.
    key_columns = [
        col for col in ("valid_time", "time", "latitude", "longitude")
        if col in result.columns
    ]

    if key_columns:
        result = result.drop_duplicates(subset=key_columns)

    # Save a clean CSV.
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nFormatting complete.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total rows: {len(result):,}")
    print(f"Columns: {len(result.columns)}")
    print("\nColumn names:")
    print(", ".join(result.columns))


if __name__ == "__main__":
    main()
