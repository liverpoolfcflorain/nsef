
#!/usr/bin/env python3
"""
CPS Data Formatter
Cleans CPS crop/disease data for the wheat yellow rust project.

Input:  CSV, XLSX, or XLS
Output: cleaned CSV, cleaning report, and validation issues

Install:
    python -m pip install pandas openpyxl
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


MISSING_VALUES = {
    "", "na", "n/a", "nan", "none", "null",
    "missing", "not available", "unknown", "-"
}

DATE_HINTS = ("date", "time", "year")
NUMERIC_HINTS = (
    "latitude", "longitude", "lat", "lon",
    "temperature", "temp", "precipitation", "rain",
    "humidity", "severity", "ndvi", "ndre",
    "elevation", "altitude", "wind", "pressure"
)


def standardise_column(name):
    """Convert column names to consistent snake_case."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed_column"


def load_data(path, sheet=None):
    """Read CSV or Excel data."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        # Try common CSV encodings.
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode the CSV file.")

    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet or 0)

    raise ValueError("Unsupported file. Use CSV, XLSX, or XLS.")


def clean_data(df):
    """Clean data without fabricating missing research observations."""
    report = {
        "input_rows": int(len(df)),
        "input_columns": int(len(df.columns)),
        "actions": [],
        "warnings": [],
    }

    # Remove entirely empty rows and columns.
    df = df.dropna(how="all").copy()
    empty_columns = df.columns[df.isna().all()].tolist()

    if empty_columns:
        df = df.drop(columns=empty_columns)
        report["actions"].append(
            f"Removed {len(empty_columns)} entirely empty columns."
        )

    # Standardise column names.
    original_names = list(df.columns)
    new_names = [standardise_column(c) for c in original_names]

    # Keep column names unique if standardisation creates collisions.
    seen = {}
    unique_names = []

    for name in new_names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique_names.append(name if count == 0 else f"{name}_{count + 1}")

    df.columns = unique_names

    renamed = {
        str(old): new
        for old, new in zip(original_names, unique_names)
        if str(old) != new
    }

    if renamed:
        report["column_renaming"] = renamed

    # Standardise text and missing-value representations.
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or \
                pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(
                lambda x: x.strip() if isinstance(x, str) else x
            )

            df[col] = df[col].map(
                lambda x: pd.NA
                if isinstance(x, str) and x.lower() in MISSING_VALUES
                else x
            )

    # Convert columns that clearly represent dates.
    for col in df.columns:
        if any(hint in col for hint in DATE_HINTS):
            if not pd.api.types.is_numeric_dtype(df[col]):
                original_nonmissing = df[col].notna().sum()
                parsed = pd.to_datetime(
                    df[col], errors="coerce", dayfirst=False
                )

                # Only convert when most nonmissing values parse as dates.
                if original_nonmissing and (
                    parsed.notna().sum() / original_nonmissing >= 0.8
                ):
                    df[col] = parsed.dt.strftime("%Y-%m-%d")
                    df.loc[parsed.isna(), col] = pd.NA
                else:
                    report["warnings"].append(
                        f"Could not confidently parse date column '{col}'. "
                        "It was preserved."
                    )

    # Convert likely numeric columns without forcing unrelated text fields.
    for col in df.columns:
        if any(hint in col for hint in NUMERIC_HINTS):
            if not pd.api.types.is_numeric_dtype(df[col]):
                original = df[col].copy()
                cleaned = (
                    original.astype("string")
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                )
                converted = pd.to_numeric(cleaned, errors="coerce")

                original_nonmissing = original.notna().sum()
                conversion_rate = (
                    converted.notna().sum() / original_nonmissing
                    if original_nonmissing else 1
                )

                if conversion_rate >= 0.8:
                    df[col] = converted
                else:
                    report["warnings"].append(
                        f"Column '{col}' was not converted to numeric "
                        "because too many values could not be parsed."
                    )

    # Remove exact duplicate rows only.
    duplicates = int(df.duplicated().sum())
    if duplicates:
        df = df.drop_duplicates().copy()
        report["actions"].append(
            f"Removed {duplicates} exact duplicate rows."
        )

    # Validate latitude and longitude, but do not delete invalid records.
    for col, low, high in (
        ("latitude", -90, 90),
        ("longitude", -180, 180),
        ("lat", -90, 90),
        ("lon", -180, 180),
        ("severity_pct", 0, 100),
    ):
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            invalid = numeric.notna() & (
                (numeric < low) | (numeric > high)
            )

            if invalid.any():
                report["warnings"].append(
                    f"Column '{col}' has {int(invalid.sum())} values "
                    f"outside the expected range [{low}, {high}]. "
                    "Review these records."
                )

    # Report missingness for every column.
    missing_report = {}

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count:
            missing_report[col] = {
                "missing_count": missing_count,
                "missing_percent": round(
                    missing_count / max(len(df), 1) * 100, 2
                ),
            }

    report["output_rows"] = int(len(df))
    report["output_columns"] = int(len(df.columns))
    report["missing_values"] = missing_report
    report["column_types"] = {
        col: str(dtype) for col, dtype in df.dtypes.items()
    }

    return df, report


def main():
    parser = argparse.ArgumentParser(
        description="Clean and standardise CPS research data."
    )
    parser.add_argument(
        "--input", required=True,
        help="Input CSV, XLSX, or XLS file."
    )
    parser.add_argument(
        "--output-dir", default="formatted_cps",
        help="Directory for formatted files."
    )
    parser.add_argument(
        "--sheet", default=None,
        help="Optional Excel sheet name."
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_path}")
    raw = load_data(input_path, args.sheet)

    if raw.empty:
        raise ValueError("The input file contains no data rows.")

    cleaned, report = clean_data(raw)

    output_csv = output_dir / "cps_cleaned.csv"
    report_json = output_dir / "cps_cleaning_report.json"
    missing_csv = output_dir / "cps_missing_values.csv"

    cleaned.to_csv(output_csv, index=False, encoding="utf-8-sig")

    report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    missing_df = pd.DataFrame([
        {
            "column": col,
            "missing_count": info["missing_count"],
            "missing_percent": info["missing_percent"],
        }
        for col, info in report["missing_values"].items()
    ])

    if missing_df.empty:
        missing_df = pd.DataFrame(
            columns=["column", "missing_count", "missing_percent"]
        )

    missing_df.to_csv(missing_csv, index=False)

    print("\nFormatting complete.")
    print(f"Cleaned data: {output_csv}")
    print(f"Cleaning report: {report_json}")
    print(f"Missing-value report: {missing_csv}")
    print(f"Rows: {report['input_rows']} -> {report['output_rows']}")

    if report["warnings"]:
        print("\nWarnings to review:")
        for warning in report["warnings"]:
            print(f" - {warning}")


if __name__ == "__main__":
    main()
