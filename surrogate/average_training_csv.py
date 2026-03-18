#!/usr/bin/env python3
"""
Average a run-level surrogate training CSV into one row per geometry/particle/energy point.
Example:
python3 surrogate/average_training_csv.py \
  --in surrogate/csv_data/training.csv \
  --out surrogate/csv_data/training_avg.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

FEATURE_COLUMNS = [
    "gun_particle",
    "nLayers",
    "seg1_layers",
    "seg2_layers",
    "seg3_layers",
    "t_absorber_seg1",
    "t_absorber_seg2",
    "t_absorber_seg3",
    "t_scin_seg1",
    "t_scin_seg2",
    "t_scin_seg3",
    "t_spacer",
    "gun_energy_GeV",
    "muon_threshold_GeV",
]

GROUP_BY_COLUMNS = [
    "geometry_id",
    "gun_particle",
    "gun_energy_GeV",
    "muon_threshold_GeV",
]

AVERAGE_COLUMNS = [
    "detection_efficiency",
    "eff_lo",
    "eff_hi",
    "energy_resolution",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average a run-level surrogate training CSV by geometry, particle, energy, and threshold."
    )
    parser.add_argument(
        "--in",
        dest="input_csv",
        required=True,
        help="Input run-level training CSV path.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output geometry-level averaged CSV path.",
    )
    return parser.parse_args()


def parse_float(value_text: str) -> float | None:
    value_text = value_text.strip()
    if not value_text:
        return None
    try:
        return float(value_text)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    average_value = sum(values) / len(values)
    variance = sum((value - average_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def main() -> int:
    # Read the run-level rows, group them by geometry/particle/energy/threshold, then write one averaged row per point.
    arguments = parse_arguments()
    input_path = Path(arguments.input_csv).expanduser().resolve()
    output_path = Path(arguments.out).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} does not contain a CSV header.")
        missing_group_columns = [column_name for column_name in GROUP_BY_COLUMNS if column_name not in reader.fieldnames]
        if missing_group_columns:
            raise ValueError(f"{input_path} is missing required columns: {missing_group_columns}")

        rows_by_group: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            group_key = tuple(row.get(column_name, "").strip() for column_name in GROUP_BY_COLUMNS)
            if not group_key[0]:
                raise ValueError("Encountered a row with an empty geometry_id.")
            rows_by_group[group_key].append(row)

    output_rows: list[dict[str, object]] = []
    # Collapse all runs for one geometry/particle/energy/threshold point into one averaged training row.
    for group_key in sorted(rows_by_group):
        geometry_rows = rows_by_group[group_key]
        first_row = geometry_rows[0]

        output_row: dict[str, object] = {
            "geometry_id": group_key[0],
            "n_runs": len(geometry_rows),
        }

        for column_name in FEATURE_COLUMNS:
            output_row[column_name] = first_row.get(column_name, "")

        for column_name in AVERAGE_COLUMNS:
            values = []
            for row in geometry_rows:
                parsed_value = parse_float(row.get(column_name, ""))
                if parsed_value is not None:
                    values.append(parsed_value)

            average_value = mean(values)
            std_value = sample_std(values)
            output_row[column_name] = "" if average_value is None else average_value
            output_row[f"{column_name}_std"] = "" if std_value is None else std_value

        output_rows.append(output_row)

    fieldnames = [
        "geometry_id",
        "n_runs",
        *FEATURE_COLUMNS,
        "detection_efficiency",
        "detection_efficiency_std",
        "eff_lo",
        "eff_lo_std",
        "eff_hi",
        "eff_hi_std",
        "energy_resolution",
        "energy_resolution_std",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"Wrote {len(output_rows)} geometry rows -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
