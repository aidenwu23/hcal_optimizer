#!/usr/bin/env python3
"""
Compare cumulative interaction probability across HCAL layers for two geometries.
Example:
python analysis/geometry/compare/compare_probability.py \
  --reference data/geometry_analysis/<ref_geometry>/layers.csv \
  --candidate data/geometry_analysis/<candidate_geometry>/layers.csv

python analysis/geometry/compare/compare_probability.py \
  --reference data/geometry_analysis/81c3da7d/layers.csv \
  --candidate data/geometry_analysis/e5333e82/layers.csv

python analysis/geometry/compare/compare_probability.py \
  --reference data/geometry_analysis/81c3da7d/layers.csv \
  --candidate data/geometry_analysis/3a2e74e3/layers.csv

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
class LayerProbabilityRow:
    layer_index: int
    cumulative_probability: float


@dataclass
class LayerProbabilityDifferenceRow:
    layer_index: int
    difference_cumulative_probability: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cumulative interaction probability vs layer for two layers.csv files."
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference layers.csv path.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate layers.csv path.",
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


def comparison_label_from_layers_path(layers_csv_path: Path) -> str:
    if layers_csv_path.stem != "layers":
        return layers_csv_path.stem
    return layers_csv_path.parent.name


def load_probability_rows(layers_csv_path: Path) -> list[LayerProbabilityRow]:
    # Read the cumulative interaction probability for each layer.
    with layers_csv_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"layer_index", "cumulative_probability"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{layers_csv_path} is missing one or more required columns: "
                "layer_index, cumulative_probability"
            )

        probability_rows: list[LayerProbabilityRow] = []
        for raw_row in reader:
            probability_rows.append(
                LayerProbabilityRow(
                    layer_index=int(raw_row["layer_index"]),
                    cumulative_probability=float(raw_row["cumulative_probability"]),
                )
            )

    if not probability_rows:
        raise ValueError(f"{layers_csv_path} does not contain any layer rows.")
    return probability_rows


def build_difference_rows(
    reference_rows: list[LayerProbabilityRow],
    candidate_rows: list[LayerProbabilityRow],
) -> list[LayerProbabilityDifferenceRow]:
    # Match the curves by layer index and compute candidate minus reference.
    if len(reference_rows) != len(candidate_rows):
        raise ValueError(
            "Reference and candidate files have different numbers of layers: "
            f"{len(reference_rows)} vs {len(candidate_rows)}."
        )

    difference_rows: list[LayerProbabilityDifferenceRow] = []
    for reference_row, candidate_row in zip(reference_rows, candidate_rows):
        if reference_row.layer_index != candidate_row.layer_index:
            raise ValueError(
                "Reference and candidate files do not share the same layer_index sequence: "
                f"{reference_row.layer_index} vs {candidate_row.layer_index}."
            )

        difference_rows.append(
            LayerProbabilityDifferenceRow(
                layer_index=reference_row.layer_index,
                difference_cumulative_probability=(
                    candidate_row.cumulative_probability - reference_row.cumulative_probability
                ),
            )
        )

    return difference_rows


def write_root_plot(
    output_root_path: Path,
    difference_rows: list[LayerProbabilityDifferenceRow],
) -> Path:
    # Write the layer-by-layer probability difference graph.
    output_root_path.parent.mkdir(parents=True, exist_ok=True)
    ROOT.gROOT.SetBatch(True)

    output_file = ROOT.TFile(str(output_root_path), "RECREATE")
    if not output_file or output_file.IsZombie():
        raise OSError(f"Failed to open {output_root_path} for writing.")

    layer_directory = output_file.mkdir("layer")
    layer_directory.cd()

    difference_vs_layer = ROOT.TGraph(len(difference_rows))
    difference_vs_layer.SetName("difference_cumulative_probability_vs_layer")
    difference_vs_layer.SetTitle(
        "Cumulative interaction probability difference;Layer index;#DeltaP_{interact}"
    )
    difference_vs_layer.SetLineWidth(2)

    max_abs_difference = max(
        abs(row.difference_cumulative_probability) for row in difference_rows
    )
    y_axis_limit = max_abs_difference * 1.1  # Small padding so the extrema are still visible.
    if y_axis_limit == 0.0:
        y_axis_limit = 1e-6  # Arbitrarily chosen non-zero range for identical curves.
    difference_vs_layer.SetMinimum(-y_axis_limit)
    difference_vs_layer.SetMaximum(y_axis_limit)

    for row_index, row in enumerate(difference_rows):
        difference_vs_layer.SetPoint(
            row_index,
            float(row.layer_index),
            row.difference_cumulative_probability,
        )

    difference_vs_layer.Write()
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

    reference_label = comparison_label_from_layers_path(reference_csv_path)
    candidate_label = comparison_label_from_layers_path(candidate_csv_path)
    output_root_path = resolve_output_path(
        arguments.out_root,
        OUTPUT_DIRECTORY / f"{reference_label}_vs_{candidate_label}.root",
    )

    # Load both curves, compare them, then write the ROOT graph.
    reference_rows = load_probability_rows(reference_csv_path)
    candidate_rows = load_probability_rows(candidate_csv_path)
    difference_rows = build_difference_rows(reference_rows, candidate_rows)
    write_root_plot(output_root_path, difference_rows)

    print(f"[compare_probability] wrote output to {output_root_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
