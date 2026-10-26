
import cdsapi
import logging
import time
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OUTPUT_DIR = Path.home() / "Downloads" / "projects" / "nsef" / "era5_monthly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "reanalysis-era5-land-monthly-means"

# Start with completed months of 2026.
YEAR = "2026"
MONTHS = [
    "01", "02", "03",
    "04", "05", "06",
    "07", "08", "09",
]

# Five weather variables relevant to yellow rust research.
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

# Pakistan region: North, West, South, East.
AREA = [37.7, 68.0, 20.1, 77.0]

# This dataset represents monthly means by hour of day.
TIMES = [f"{hour:02d}:00" for hour in range(24)]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

def main():
    client = cdsapi.Client()

    total = len(MONTHS)

    for index, month in enumerate(MONTHS, start=1):
        filename = f"era5_land_{YEAR}_{month}.zip"
        output_file = OUTPUT_DIR / filename

        if output_file.exists() and output_file.stat().st_size > 0:
            logging.info(
                "[%s/%s] Already downloaded, skipping: %s",
                index, total, filename,
            )
            continue

        request = {
            "product_type": [
                "monthly_averaged_reanalysis_by_hour_of_day"
            ],
            "variable": VARIABLES,
            "year": [YEAR],
            "month": [month],
            "time": TIMES,
            "data_format": "netcdf",
            "download_format": "zip",
            "area": AREA,
        }

        logging.info(
            "[%s/%s] Requesting ERA5-Land for %s-%s",
            index, total, YEAR, month,
        )

        try:
            # Passing the target path explicitly saves the file there.
            client.retrieve(
                DATASET,
                request,
                str(output_file),
            )

            if output_file.exists() and output_file.stat().st_size > 0:
                logging.info(
                    "Saved: %s (%.2f MB)",
                    output_file,
                    output_file.stat().st_size / (1024 * 1024),
                )
            else:
                logging.warning(
                    "Request finished but no output file was found: %s",
                    output_file,
                )

        except Exception:
            logging.exception(
                "Download failed for %s-%s. Continuing to next month.",
                YEAR, month,
            )

        # Avoid immediately submitting the next request.
        if index < total:
            time.sleep(3)

    logging.info("Finished processing monthly requests.")
    logging.info("Output directory: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
