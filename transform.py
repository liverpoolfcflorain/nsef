
#!/usr/bin/env python3
"""
transform.py
------------
Build a clean image manifest for the NWRD wheat rust dataset.

Expected directory structure:
    NWRD/
        train/
            images/
            masks/
        test/
            images/
            masks/

Outputs:
    formatted_nwrd/
        nwrd_image_manifest.csv
        nwrd_summary.csv
        nwrd_missing_masks.csv
        nwrd_missing_dates.csv

Run:
    python3 transform.py --root NWRD --output formatted_nwrd

Dependencies:
    pip install pillow pandas
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ExifTags


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}

IMAGE_DIR_NAMES = {"images", "image", "imgs", "img"}
MASK_DIR_NAMES = {"masks", "mask", "annotations", "labels"}

# Standard EXIF date fields.
EXIF_DATE_TAG_NAMES = {
    "DateTimeOriginal",
    "DateTimeDigitized",
    "DateTime",
    "CreateDate",
    "CreationDate",
}

# Common date formats found in image metadata.
DATE_FORMATS = (
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%Y:%m:%d",
    "%Y%m%d_%H%M%S",
    "%Y%m%d",
)


# ------------------------------------------------------------
# DATE PARSING
# ------------------------------------------------------------

def parse_exif_date(value: Any) -> datetime | None:
    """
    Parse EXIF dates such as:
        2022:06:27 20:08:11

    Returns None for missing or invalid dates.
    Never substitutes datetime.min or year 0001.
    """
    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    value = str(value).strip()

    if not value:
        return None

    # Remove common EXIF suffixes, if present.
    value = value.replace("\x00", "").strip()

    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(value, fmt)

            # Reject clearly invalid or placeholder years.
            if parsed.year < 1900 or parsed.year > 2100:
                continue

            return parsed
        except (ValueError, TypeError):
            continue

    # Try ISO parsing for less common variants.
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if 1900 <= parsed.year <= 2100:
            return parsed.replace(tzinfo=None)
    except (ValueError, TypeError):
        pass

    return None


def extract_exif_capture_date(
    image: Image.Image,
) -> tuple[datetime | None, str | None]:
    """
    Extract capture date from EXIF.

    Priority:
        1. DateTimeOriginal
        2. DateTimeDigitized
        3. DateTime

    Returns:
        (datetime, EXIF field name)
    """
    try:
        exif = image.getexif()
    except Exception:
        return None, None

    if not exif:
        return None, None

    # Convert numeric EXIF tag IDs to readable names.
    readable_exif: dict[str, Any] = {}

    for tag_id, value in exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
        readable_exif[tag_name] = value

    priority = (
        "DateTimeOriginal",
        "DateTimeDigitized",
        "DateTime",
    )

    for field in priority:
        if field in readable_exif:
            parsed = parse_exif_date(readable_exif[field])
            if parsed is not None:
                return parsed, field

    # Some images store metadata in nested EXIF IFD data.
    try:
        exif_ifd_tag = next(
            (
                tag_id
                for tag_id, name in ExifTags.TAGS.items()
                if name == "ExifOffset"
            ),
            None,
        )

        if exif_ifd_tag is not None:
            nested = exif.get_ifd(exif_ifd_tag)

            for tag_id, value in nested.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))

                if tag_name in EXIF_DATE_TAG_NAMES:
                    parsed = parse_exif_date(value)
                    if parsed is not None:
                        return parsed, tag_name
    except Exception:
        pass

    return None, None


def extract_filename_date(filename: str) -> datetime | None:
    """
    Look for a plausible date in the filename.

    Supports patterns such as:
        2022-06-27_image.jpg
        20220627_image.jpg
        image_20220627_120311.jpg

    A filename date is a fallback only. It is not treated as
    a verified capture date.
    """
    stem = Path(filename).stem

    patterns = (
        r"(?<!\d)(20\d{2}[-_]\d{2}[-_]\d{2})(?!\d)",
        r"(?<!\d)(20\d{6})(?!\d)",
        r"(?<!\d)(20\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})(?!\d)",
    )

    for pattern in patterns:
        for match in re.finditer(pattern, stem):
            candidate = match.group(1).replace("_", "-")

            parsed = parse_exif_date(candidate)
            if parsed is not None:
                return parsed

            # Handle YYYY-MM-DD-HH-MM-SS.
            try:
                parsed = datetime.strptime(candidate, "%Y-%m-%d-%H-%M-%S")
                if 1900 <= parsed.year <= 2100:
                    return parsed
            except ValueError:
                pass

    return None


# ------------------------------------------------------------
# GPS EXTRACTION
# ------------------------------------------------------------

def _rational_to_float(value: Any) -> float:
    """Convert a Pillow EXIF rational value to float."""
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        denominator = float(value.denominator)
        if denominator == 0:
            raise ValueError("Invalid rational denominator")
        return float(value.numerator) / denominator

    if isinstance(value, tuple) and len(value) == 2:
        denominator = float(value[1])
        if denominator == 0:
            raise ValueError("Invalid rational denominator")
        return float(value[0]) / denominator

    return float(value)


def _gps_coordinate_to_decimal(
    coordinate: Any,
    reference: Any,
) -> float | None:
    """Convert EXIF degrees/minutes/seconds to decimal degrees."""
    try:
        degrees = _rational_to_float(coordinate[0])
        minutes = _rational_to_float(coordinate[1])
        seconds = _rational_to_float(coordinate[2])

        decimal = degrees + minutes / 60 + seconds / 3600

        if isinstance(reference, bytes):
            reference = reference.decode("ascii", errors="ignore")

        reference = str(reference).strip().upper()

        if reference in {"S", "W"}:
            decimal = -decimal
        elif reference not in {"N", "E"}:
            return None

        return decimal
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        return None


def extract_gps(image: Image.Image) -> tuple[float | None, float | None]:
    """
    Extract latitude and longitude from EXIF GPS metadata.

    Returns (latitude, longitude), or (None, None) if unavailable.
    """
    try:
        exif = image.getexif()
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:
        gps_ifd = {}

    if not gps_ifd:
        # Fallback for Pillow versions that expose GPSInfo directly.
        try:
            exif = image.getexif()
            raw_gps = exif.get(34853)
            if isinstance(raw_gps, dict):
                gps_ifd = raw_gps
        except Exception:
            gps_ifd = {}

    if not gps_ifd:
        return None, None

    gps: dict[str, Any] = {
        ExifTags.GPSTAGS.get(tag_id, str(tag_id)): value
        for tag_id, value in gps_ifd.items()
    }

    latitude = _gps_coordinate_to_decimal(
        gps.get("GPSLatitude"),
        gps.get("GPSLatitudeRef"),
    )
    longitude = _gps_coordinate_to_decimal(
        gps.get("GPSLongitude"),
        gps.get("GPSLongitudeRef"),
    )

    if latitude is None or longitude is None:
        return None, None

    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None, None

    return latitude, longitude


# ------------------------------------------------------------
# FILE DISCOVERY AND MATCHING
# ------------------------------------------------------------

def discover_images(root: Path) -> list[Path]:
    """Find image files under the dataset root."""
    images = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.parent.name.lower() in IMAGE_DIR_NAMES
    ]

    return sorted(images)


def discover_mask_directories(root: Path) -> list[Path]:
    """Find directories that appear to contain masks."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.lower() in MASK_DIR_NAMES
    )


def normalize_stem(path: Path) -> str:
    """Normalize filename stem for image/mask matching."""
    return path.stem.strip().lower()


def build_mask_index(root: Path) -> dict[str, list[Path]]:
    """
    Index masks by filename stem.

    Example:
        images/105.jpg
        masks/105.png

    These files match even if their extensions differ.
    """
    index: dict[str, list[Path]] = {}

    for mask_dir in discover_mask_directories(root):
        for path in mask_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(normalize_stem(path), []).append(path)

    # Remove duplicate paths and keep output deterministic.
    for key in index:
        index[key] = sorted(set(index[key]))

    return index


def find_mask_for_image(
    image_path: Path,
    mask_index: dict[str, list[Path]],
) -> tuple[Path | None, int]:
    """
    Find a mask matching the image stem.

    If multiple masks have the same stem, prefer a mask in a
    directory corresponding to the image's train/test split.
    """
    candidates = mask_index.get(normalize_stem(image_path), [])

    if not candidates:
        return None, 0

    split_name = None
    for part in image_path.parts:
        if part.lower() in {"train", "test", "validation", "val"}:
            split_name = part.lower()
            break

    if split_name:
        matching_split = [
            candidate
            for candidate in candidates
            if split_name in {part.lower() for part in candidate.parts}
        ]

        if matching_split:
            candidates = matching_split

    return candidates[0], len(candidates)


def infer_split(image_path: Path) -> str:
    """Infer train/test split from the image path."""
    for part in image_path.parts:
        normalized = part.lower()

        if normalized in {"train", "training"}:
            return "train"
        if normalized in {"test", "testing"}:
            return "test"
        if normalized in {"val", "validation"}:
            return "validation"

    return "unknown"


def infer_class_label(image_path: Path) -> str:
    """
    Infer a class from a class-specific parent folder if present.

    For the standard NWRD structure where every image is directly
    inside train/images or test/images, returns 'unknown' because
    the directory structure does not provide a class label.
    """
    parent = image_path.parent

    if parent.name.lower() not in IMAGE_DIR_NAMES:
        return parent.name

    # Check for a class directory between the split and images.
    parts = list(image_path.parts)

    for index, part in enumerate(parts):
        if part.lower() in IMAGE_DIR_NAMES and index > 0:
            candidate = parts[index - 1].lower()

            if candidate not in {
                "train",
                "training",
                "test",
                "testing",
                "val",
                "validation",
            }:
                return parts[index - 1]

    return "unknown"


# ------------------------------------------------------------
# IMAGE METADATA
# ------------------------------------------------------------

def read_image_metadata(
    image_path: Path,
) -> dict[str, Any]:
    """Read date, GPS, dimensions and basic file information."""
    result: dict[str, Any] = {
        "image_date": None,
        "capture_datetime": None,
        "date_source": "missing",
        "exif_date_field": None,
        "candidate_date_from_filename": None,
        "latitude": None,
        "longitude": None,
        "image_width": None,
        "image_height": None,
        "image_format": None,
        "metadata_error": None,
    }

    try:
        with Image.open(image_path) as image:
            result["image_width"] = image.width
            result["image_height"] = image.height
            result["image_format"] = image.format

            capture_date, exif_field = extract_exif_capture_date(image)
            latitude, longitude = extract_gps(image)

            result["latitude"] = latitude
            result["longitude"] = longitude
            result["exif_date_field"] = exif_field

            if capture_date is not None:
                result["capture_datetime"] = capture_date.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                result["image_date"] = capture_date.strftime("%Y-%m-%d")
                result["date_source"] = "EXIF"

            else:
                filename_date = extract_filename_date(image_path.name)

                if filename_date is not None:
                    result["candidate_date_from_filename"] = (
                        filename_date.strftime("%Y-%m-%d")
                    )

                    # Do not silently promote a filename-derived date
                    # to a verified capture date.
                    result["date_source"] = "filename_candidate"

    except Exception as exc:
        result["metadata_error"] = str(exc)

    return result


# ------------------------------------------------------------
# MANIFEST CREATION
# ------------------------------------------------------------

def create_manifest(root: Path) -> pd.DataFrame:
    """Create one manifest record for each discovered image."""
    images = discover_images(root)

    if not images:
        raise FileNotFoundError(
            f"No images found under {root.resolve()}.\n"
            "Expected folders such as NWRD/train/images and NWRD/test/images."
        )

    mask_index = build_mask_index(root)
    records: list[dict[str, Any]] = []

    project_root = root.resolve().parent

    print(f"Dataset root: {root.resolve()}")
    print(f"Images discovered: {len(images)}")
    print(f"Mask filename keys: {len(mask_index)}")
    print("Reading image metadata...")

    for number, image_path in enumerate(images, start=1):
        metadata = read_image_metadata(image_path)
        mask_path, mask_match_count = find_mask_for_image(
            image_path,
            mask_index,
        )

        try:
            relative_path = str(image_path.resolve().relative_to(project_root))
        except ValueError:
            relative_path = str(image_path.resolve())

        if mask_path is not None:
            try:
                relative_mask_path = str(
                    mask_path.resolve().relative_to(project_root)
                )
            except ValueError:
                relative_mask_path = str(mask_path.resolve())
        else:
            relative_mask_path = None

        record = {
            "image_id": image_path.stem,
            "image_path": str(image_path.resolve()),
            "relative_path": relative_path,
            "split": infer_split(image_path),
            "class_label": infer_class_label(image_path),
            "candidate_date_from_filename": metadata[
                "candidate_date_from_filename"
            ],
            "image_date": metadata["image_date"],
            "capture_datetime": metadata["capture_datetime"],
            "date_source": metadata["date_source"],
            "exif_date_field": metadata["exif_date_field"],
            "latitude": metadata["latitude"],
            "longitude": metadata["longitude"],
            "mask_path": relative_mask_path,
            "mask_match_count": mask_match_count,
            "mask_found": mask_path is not None,
            "file_extension": image_path.suffix.lower(),
            "image_format": metadata["image_format"],
            "image_width": metadata["image_width"],
            "image_height": metadata["image_height"],
            "metadata_verified": (
                metadata["date_source"] == "EXIF"
                or (
                    metadata["latitude"] is not None
                    and metadata["longitude"] is not None
                )
            ),
            "metadata_error": metadata["metadata_error"],
        }

        records.append(record)

        if number % 25 == 0 or number == len(images):
            print(f"  Processed {number}/{len(images)} images")

    df = pd.DataFrame(records)

    # Explicitly parse our known ISO output format.
    # Missing values stay missing; they never become year 0001.
    df["image_date"] = pd.to_datetime(
        df["image_date"],
        format="%Y-%m-%d",
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    df["candidate_date_from_filename"] = pd.to_datetime(
        df["candidate_date_from_filename"],
        format="%Y-%m-%d",
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    # Preserve missing dates as actual blank/NaN values.
    df = df.sort_values(
        by=["split", "image_id"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    return df


# ------------------------------------------------------------
# REPORTS
# ------------------------------------------------------------

def create_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact dataset-quality summary."""
    total = len(df)

    def count_true(column: str) -> int:
        return int(df[column].fillna(False).astype(bool).sum())

    exif_count = int((df["date_source"] == "EXIF").sum())
    filename_count = int(
        (df["date_source"] == "filename_candidate").sum()
    )
    missing_date_count = int(df["image_date"].isna().sum())

    summary_rows = [
        ("total_images", total),
        ("train_images", int((df["split"] == "train").sum())),
        ("test_images", int((df["split"] == "test").sum())),
        ("validation_images", int((df["split"] == "validation").sum())),
        ("unknown_split_images", int((df["split"] == "unknown").sum())),
        ("images_with_masks", count_true("mask_found")),
        ("images_without_masks", int((~df["mask_found"]).sum())),
        ("images_with_exif_dates", exif_count),
        ("images_with_filename_date_candidates", filename_count),
        ("images_without_capture_date", missing_date_count),
        (
            "images_with_gps",
            int(
                (
                    df["latitude"].notna()
                    & df["longitude"].notna()
                ).sum()
            ),
        ),
        (
            "images_with_metadata_errors",
            int(df["metadata_error"].notna().sum()),
        ),
        (
            "images_with_multiple_mask_matches",
            int((df["mask_match_count"] > 1).sum()),
        ),
    ]

    valid_dates = pd.to_datetime(df["image_date"], errors="coerce").dropna()

    summary_rows.extend([
        (
            "earliest_capture_date",
            valid_dates.min().strftime("%Y-%m-%d")
            if not valid_dates.empty else "",
        ),
        (
            "latest_capture_date",
            valid_dates.max().strftime("%Y-%m-%d")
            if not valid_dates.empty else "",
        ),
    ])

    return pd.DataFrame(summary_rows, columns=["metric", "value"])


def save_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    """Save the manifest and quality-control reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "nwrd_image_manifest.csv"
    summary_path = output_dir / "nwrd_summary.csv"
    missing_masks_path = output_dir / "nwrd_missing_masks.csv"
    missing_dates_path = output_dir / "nwrd_missing_dates.csv"

    df.to_csv(manifest_path, index=False)

    summary = create_summary(df)
    summary.to_csv(summary_path, index=False)

    df.loc[~df["mask_found"]].to_csv(
        missing_masks_path,
        index=False,
    )

    df.loc[df["image_date"].isna()].to_csv(
        missing_dates_path,
        index=False,
    )

    print("\nOutput files:")
    print(f"  Manifest:      {manifest_path.resolve()}")
    print(f"  Summary:       {summary_path.resolve()}")
    print(f"  Missing masks: {missing_masks_path.resolve()}")
    print(f"  Missing dates: {missing_dates_path.resolve()}")

    print("\nSummary:")
    for _, row in summary.iterrows():
        print(f"  {row['metric']}: {row['value']}")

    print("\nDate-source counts:")
    for source, count in Counter(df["date_source"]).items():
        print(f"  {source}: {count}")

    print("\nFirst 10 manifest rows:")
    columns_to_show = [
        "image_id",
        "split",
        "image_date",
        "date_source",
        "mask_found",
    ]
    print(df[columns_to_show].head(10).to_string(index=False))


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract NWRD image metadata and build a clean CSV manifest."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("NWRD"),
        help="Path to the NWRD dataset directory (default: NWRD)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("formatted_nwrd"),
        help="Output directory (default: formatted_nwrd)",
    )

    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not root.exists():
        print(f"ERROR: Dataset directory does not exist: {root}", file=sys.stderr)
        return 1

    if not root.is_dir():
        print(f"ERROR: Dataset root is not a directory: {root}", file=sys.stderr)
        return 1

    try:
        manifest = create_manifest(root)
        save_outputs(manifest, output_dir)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\nDone. The manifest has been generated.")
    print("Important: filename-derived dates are candidates, not verified EXIF dates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
