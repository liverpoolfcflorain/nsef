
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

MANIFEST = ROOT / "formatted_nwrd" / "nwrd_image_manifest.csv"
WEATHER = ROOT / "formatted_era5" / "era5_weather_cleaned.csv"
OUTPUT_DIR = ROOT / "merged_data"

OUTPUT = OUTPUT_DIR / "nwrd_weather_merged.csv"
MISSING_DATES = OUTPUT_DIR / "nwrd_missing_dates.csv"

WEATHER_FEATURES = ["t2m", "d2m", "tp", "u10", "v10"]


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = pd.read_csv(MANIFEST)

    if "image_date" not in images.columns:
        raise ValueError("The image manifest has no image_date column.")

    # Parse dates explicitly. Never replace invalid dates with year 0001.
    images["image_date"] = pd.to_datetime(
        images["image_date"], errors="coerce"
    )

    images["image_date"] = images["image_date"].dt.strftime("%Y-%m-%d")
    images["month_day"] = pd.to_datetime(
        images["image_date"], errors="coerce"
    ).dt.strftime("%m-%d")

    images["weather_match"] = "not_matched_historical_weather"

    for feature in WEATHER_FEATURES:
        images[feature] = float("nan")

    images["matched_weather_latitude"] = float("nan")
    images["matched_weather_longitude"] = float("nan")

    # Check the weather file for date compatibility, but do not merge
    # unrelated 2026 weather into images captured in 2022.
    if WEATHER.exists():
        weather = pd.read_csv(WEATHER)

        if "valid_time" in weather.columns:
            weather["valid_time"] = pd.to_datetime(
                weather["valid_time"], errors="coerce"
            )

            valid_times = weather["valid_time"].dropna()

            if not valid_times.empty:
                print(
                    "Weather timestamps:",
                    valid_times.min(),
                    "to",
                    valid_times.max(),
                )

                image_dates = pd.to_datetime(
                    images["image_date"], errors="coerce"
                ).dropna()

                if not image_dates.empty:
                    print(
                        "Image dates:",
                        image_dates.min(),
                        "to",
                        image_dates.max(),
                    )

                    overlaps = (
                        valid_times.min().date()
                        <= image_dates.max().date()
                        and valid_times.max().date()
                        >= image_dates.min().date()
                    )

                    if overlaps:
                        print(
                            "Weather time range overlaps image dates. "
                            "Spatial and temporal matching still needs "
                            "to be implemented and verified."
                        )
                    else:
                        print(
                            "WARNING: Weather and image date ranges do not "
                            "overlap. No weather values have been merged."
                        )
    else:
        print(f"Weather file not found: {WEATHER}")

    images.to_csv(OUTPUT, index=False)

    missing_dates = images[
        pd.to_datetime(images["image_date"], errors="coerce").isna()
    ]
    missing_dates.to_csv(MISSING_DATES, index=False)

    print(f"\nImages processed: {len(images)}")
    print(
        "Valid image dates:",
        pd.to_datetime(images["image_date"], errors="coerce").notna().sum(),
    )
    print(f"Missing image dates: {len(missing_dates)}")
    print("Disease labels present:", images["class_label"].notna().sum())
    print("Weather rows merged: 0 (historical data not yet verified)")
    print(f"\nSaved merged dataset: {OUTPUT}")
    print(f"Saved missing dates: {MISSING_DATES}")


if __name__ == "__main__":
    main()
