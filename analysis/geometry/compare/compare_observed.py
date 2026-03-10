#!/usr/bin/env python3
"""
Compare cumulative observed start probability across HCAL layers for two geometries.
Example:

python analysis/geometry/compare/compare_observed.py \
  --reference data/geometry_analysis/81c3da7d/rune3896ec0d8/start_layer_observed_layers.csv \
  --candidate data/geometry_analysis/e5333e82/run7bbd931745/start_layer_observed_layers.csv

python analysis/geometry/compare/compare_observed.py \
  --reference data/geometry_analysis/81c3da7d/rune3896ec0d8/start_layer_observed_layers.csv \
  --candidate data/geometry_analysis/3a2e74e3/runfc6edabae2/start_layer_observed_layers.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import ROOT

PROJECT_DIRECTORY = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "data" / "geometry_analysis" / "comparisons"


@dataclass
class ObservedLayerRow:
    layer_index: int
    cumulative_observed_start_fraction: float


@dataclass
class ObservedLayerDifferenceRow:
    layer_index: int
    difference_cumulative_observed_start_fraction: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cumulative observed start probability vs layer for two observed CSV files."
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference observed CSV path.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate observed CSV path.",
    )
    parser.add_argument(
        "--out-root",
        default="",
        help="Optional output ROOT path.",
    )
    return parser.parse_args()


def resolve_output_path(path_value: str, fallback_path: Path) -> Path:
    if path_value.strip():
        return Path(path_value).expanduser().resolve()
    return fallback_path


def comparison_label_from_csv_path(csv_path: Path) -> str:
    if csv_path.stem == "start_layer_observed_layers":
        return f"{csv_path.parent.parent.name}_{csv_path.parent.name}"
    return csv_path.stem


def load_observed_rows(csv_path: Path) -> list[ObservedLayerRow]:
    # Read the cumulative observed start probability for each layer.
    with csv_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"layer_index", "cumulative_observed_start_fraction"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path} is missing one or more required columns: "
                "layer_index, cumulative_observed_start_fraction"
            )

        observed_rows: list[ObservedLayerRow] = []
        for raw_row in reader:
            observed_rows.append(
                ObservedLayerRow(
                    layer_index=int(raw_row["layer_index"]),
                    cumulative_observed_start_fraction=float(
                        raw_row["cumulative_observed_start_fraction"]
                    ),
                )
            )

    if not observed_rows:
        raise ValueError(f"{csv_path} does not contain any layer rows.")
    return observed_rows


def build_difference_rows(
    reference_rows: list[ObservedLayerRow],
    candidate_rows: list[ObservedLayerRow],
) -> list[ObservedLayerDifferenceRow]:
    # Match the curves by layer index and compute candidate minus reference.
    if len(reference_rows) != len(candidate_rows):
        raise ValueError(
            "Reference and candidate files have different numbers of layers: "
            f"{len(reference_rows)} vs {len(candidate_rows)}."
        )

    difference_rows: list[ObservedLayerDifferenceRow] = []
    for reference_row, candidate_row in zip(reference_rows, candidate_rows):
        if reference_row.layer_index != candidate_row.layer_index:
            raise ValueError(
                "Reference and candidate files do not share the same layer_index sequence: "
                f"{reference_row.layer_index} vs {candidate_row.layer_index}."
            )

        difference_rows.append(
            ObservedLayerDifferenceRow(
                layer_index=reference_row.layer_index,
                difference_cumulative_observed_start_fraction=(
                    candidate_row.cumulative_observed_start_fraction
                    - reference_row.cumulative_observed_start_fraction
                ),
            )
        )

    return difference_rows


def write_root_plot(
    output_root_path: Path,
    difference_rows: list[ObservedLayerDifferenceRow],
) -> Path:
    # Write the layer-by-layer observed probability difference graph.
    output_root_path.parent.mkdir(parents=True, exist_ok=True)
    ROOT.gROOT.SetBatch(True)

    output_file = ROOT.TFile(str(output_root_path), "RECREATE")
    if not output_file or output_file.IsZombie():
        raise OSError(f"Failed to open {output_root_path} for writing.")

    layer_directory = output_file.mkdir("layer")
    layer_directory.cd()

    observed_diff_vs_layer = ROOT.TGraph(len(difference_rows))
    observed_diff_vs_layer.SetName("observed_diff_vs_layer")
    observed_diff_vs_layer.SetTitle(
        "Observed cumulative start probability difference;Layer index;#DeltaP_{start,observed}"
    )
    observed_diff_vs_layer.SetLineWidth(2)

    max_abs_difference = max(
        abs(row.difference_cumulative_observed_start_fraction) for row in difference_rows
    )
    y_axis_limit = max_abs_difference * 1.1  # Small padding so the extrema are still visible.
    if y_axis_limit == 0.0:
        y_axis_limit = 1e-6  # Arbitrarily chosen non-zero range for identical curves.
    observed_diff_vs_layer.SetMinimum(-y_axis_limit)
    observed_diff_vs_layer.SetMaximum(y_axis_limit)

    for row_index, row in enumerate(difference_rows):
        observed_diff_vs_layer.SetPoint(
            row_index,
            float(row.layer_index),
            row.difference_cumulative_observed_start_fraction,
        )

    observed_diff_vs_layer.Write()
    output_file.Close()
    return output_root_path


def main() -> int:
    # Resolve the inputs, build the difference curve, and write the ROOT output.
    arguments = parse_arguments()
    reference_csv_path = Path(arguments.reference).expanduser().resolve()
    candidate_csv_path = Path(arguments.candidate).expanduser().resolve()
    if not reference_csv_path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {reference_csv_path}")
    if not candidate_csv_path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {candidate_csv_path}")

    reference_label = comparison_label_from_csv_path(reference_csv_path)
    candidate_label = comparison_label_from_csv_path(candidate_csv_path)
    output_root_path = resolve_output_path(
        arguments.out_root,
        OUTPUT_DIRECTORY / f"{reference_label}_vs_{candidate_label}.root",
    )

    reference_rows = load_observed_rows(reference_csv_path)
    candidate_rows = load_observed_rows(candidate_csv_path)
    difference_rows = build_difference_rows(reference_rows, candidate_rows)
    write_root_plot(output_root_path, difference_rows)

    print(f"[compare_observed] wrote output to {output_root_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
