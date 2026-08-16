"""
Merge the sensor exports into a single consolidated file
=====================================================================
Reads every CSV given as input, whether clean or carrying the malformed
outer quoting, deduplicates on Date, sorts chronologically and writes a
single clean consolidated file.

Designed to be re-run whenever new exports arrive.

Usage:
    python consolidate_data.py
    (default: merges sensor_export_1.csv + sensor_export_2.csv -> EDB_consolidated.csv)

Or programmatically:
    from consolidate_data import consolidate_csvs
    df = consolidate_csvs(["file1.csv", "file2.csv"], output_path="merged.csv")
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import pandas as pd


# Expected schema: the downstream pipeline depends on these columns
EXPECTED_COLUMNS = [
    "Date", "heat_index", "humidity", "NH3_rate",
    "sensorId", "temperature", "temperaturext",
]


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a CSV, handling the malformed double-quoting case.

    Detection: if the first line starts with a quote followed by the
    first column name, with no separator inside the quote, the whole
    file is taken to be wrapped in an outer quote that must be
    stripped.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        raw = f.read()

    lines = raw.strip().splitlines()
    if not lines:
        return pd.DataFrame()

    # Double-quoting detection: first line starts with a quoted header
    first = lines[0]
    is_double_quoted = (
        first.startswith('"')
        and first.endswith('"')
        and '""' in first
    )

    if is_double_quoted:
        cleaned = []
        for line in lines:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1].replace('""', '"')
            cleaned.append(line)
        text = "\n".join(cleaned)
        df = pd.read_csv(io.StringIO(text))
        print(f"  [{path.name}] double-quoted format detected -> cleaned")
    else:
        df = pd.read_csv(path)

    # Schema validation
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} : missing columns {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Keep only the expected columns, drop any extras
    df = df[EXPECTED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"])

    return df


def consolidate_csvs(
    input_paths: Sequence[Path | str],
    output_path: Path | str | None = None,
    sort: bool = True,
    keep: str = "first",
) -> pd.DataFrame:
    """Merge several sensor CSVs, deduplicate, and write the result.

    Parameters
    ----------
    input_paths : CSV files to merge.
    output_path : output path (None to write nothing).
    sort : True to sort chronologically.
    keep : 'first' (default) or 'last', for duplicate timestamps.

    Returns
    -------
    The consolidated DataFrame.
    """
    frames = []
    for p in input_paths:
        p = Path(p)
        if not p.exists():
            print(f"  [{p.name}] not found, skipped")
            continue
        df = read_csv_robust(p)
        df["_source"] = p.name
        print(f"  [{p.name}] {len(df):,} valid rows | "
              f"{df['Date'].min()} -> {df['Date'].max()}")
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid file to consolidate.")

    merged = pd.concat(frames, ignore_index=True)

    # Counts before dedup
    n_before = len(merged)

    # Dedup key is the (Date, sensorId) pair; keep='first' retains the
    # first occurrence encountered.
    # To give one file priority, put it first in input_paths
    merged = merged.drop_duplicates(subset=["Date", "sensorId"], keep=keep)

    n_after = len(merged)
    n_dropped = n_before - n_after

    if sort:
        merged = merged.sort_values("Date").reset_index(drop=True)

    # The source-tracking column is dropped from the final output
    merged_clean = merged.drop(columns=["_source"])

    print()
    print(f"Total before dedup : {n_before:,}")
    print(f"Duplicates removed : {n_dropped:,}")
    print(f"Total after        : {n_after:,}")
    print(f"Range              : {merged_clean['Date'].min()} -> "
          f"{merged_clean['Date'].max()}")
    print(f"Sensors            : {merged_clean['sensorId'].unique().tolist()}")

    # Missing-value diagnostic
    n_missing_text = merged_clean["temperaturext"].isna().sum()
    print(f"T_ext missing      : {n_missing_text:,} "
          f"({100*n_missing_text/len(merged_clean):.1f}%)")

    if output_path:
        output_path = Path(output_path)
        merged_clean.to_csv(output_path, index=False)
        print(f"\nConsolidated file written: {output_path}")

    return merged_clean


def main():
    SCRIPT_DIR = Path(__file__).resolve().parent

    # Order sets priority: sensor_export_1.csv first, cleaner and more complete
    input_files = [
        SCRIPT_DIR / "sensor_export_1.csv",
        SCRIPT_DIR / "sensor_export_2.csv",
    ]

    output_file = SCRIPT_DIR / "EDB_consolidated.csv"

    print("=" * 65)
    print(" CONSOLIDATION OF THE EDB SENSOR EXPORTS")
    print("=" * 65)
    print()
    print("Input files:")

    df = consolidate_csvs(input_files, output_file, keep="first")

    print()
    print("Preview (first and last three rows):")
    print(pd.concat([df.head(3), df.tail(3)]).to_string())


if __name__ == "__main__":
    main()
