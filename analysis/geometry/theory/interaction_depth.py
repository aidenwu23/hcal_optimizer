#!/usr/bin/env python3
"""Geometry-only nuclear interaction depth analysis for generated HCAL stacks."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.geometry.theory.material_lambda import (
    MaterialLibrary,
    load_material_library,
    resolve_material_lambda_mm,
)
from simulation.helpers.geometry_index import (
    GeometryLayerRow,
    GeometryLayerSummary,
    GeometryVariant,
    build_layer_stack,
    summarize_layer_stack,
)

PROJECT_DIRECTORY = Path(__file__).resolve().parents[3]
ELEMENTS_XML_PATH = PROJECT_DIRECTORY / "geometries" / "definitions" / "elements.xml"
MATERIALS_XML_PATH = PROJECT_DIRECTORY / "geometries" / "definitions" / "materials.xml"
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "data" / "geometry_analysis"

# One analyzed HCAL layer with optical-depth values attached.
@dataclass
class LayerInteractionRow:
    layer_index: int
    absorber_thickness_mm: float
    scintillator_thickness_mm: float
    spacer_thickness_mm: float
    layer_total_thickness_mm: float
    absorber_lambda_I_mm: float
    scintillator_lambda_I_mm: float
    spacer_lambda_I_mm: float
    delta_tau_layer: float
    cumulative_tau: float
    cumulative_probability: float
    lambda_I_eff_layer_mm: float
    depth_front_mm: float
    depth_back_mm: float
    depth_mid_mm: float


# Geometry-level interaction-depth summary values.
@dataclass
class InteractionDepthSummary:
    geometry_id: str
    layer_count: int
    absorber_material: str
    active_material: str
    spacer_material: str
    total_depth_mm: float
    total_depth_lambda: float
    absorber_fraction_by_depth: float
    depth_90pct_interaction_mm: float
    depth_90pct_interaction_lambda: float
    depth_95pct_interaction_mm: float
    depth_95pct_interaction_lambda: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geometry-only interaction-depth analysis for generated HCAL geometries."
    )
    parser.add_argument(
        "--geometry-json",
        nargs="+",
        help="One or more generated geometry.json paths.",
    )
    return parser.parse_args()


def _load_geometry_variant_from_json_path(geometry_json_path: str) -> GeometryVariant:
    """Load one generated geometry.json into a GeometryVariant."""
    # Resolve the generated geometry file and load its parameter payload.
    params_path = Path(geometry_json_path).expanduser().resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Geometry JSON not found: {params_path}")

    with params_path.open("r", encoding="utf-8") as json_file:
        geometry_params = json.load(json_file)

    geometry_id = str(geometry_params.get("geometry_id", "")).strip()
    if not geometry_id:
        raise ValueError(f"geometry_id is missing in {params_path}")

    # Rebuild the GeometryVariant fields expected by the shared geometry helpers.
    geometry_directory = params_path.parent
    xml_path = geometry_directory / "geometry.xml"
    return GeometryVariant(
        geometry_id=geometry_id,
        tag=geometry_id,
        geometry_directory=geometry_directory,
        params_path=params_path,
        xml_path=xml_path,
        spec_path=params_path,
        params={str(key): value for key, value in geometry_params.items()},
    )


def load_material_library_for_interaction_depth() -> MaterialLibrary:
    """Load the repo material definitions and prepare lambda_I lookup records."""
    return load_material_library(
        elements_xml_path=ELEMENTS_XML_PATH,
        materials_xml_path=MATERIALS_XML_PATH,
    )


def build_layer_interaction_rows(
    geometry_variant: GeometryVariant,
    material_library: MaterialLibrary,
    layer_rows: List[GeometryLayerRow] | None = None,
) -> List[LayerInteractionRow]:
    """Attach material lambda_I to each physical layer and convert to optical depth."""
    if layer_rows is None:
        layer_rows = build_layer_stack(geometry_variant)

    # Read the three HCAL materials that define the layer interaction rate.
    absorber_material = str(geometry_variant.params.get("absorberMaterial", "")).strip()
    active_material = str(geometry_variant.params.get("activeMaterial", "")).strip()
    spacer_material = str(geometry_variant.params.get("spacerMaterial", "")).strip()
    if not absorber_material or not active_material or not spacer_material:
        raise ValueError(
            f"Geometry '{geometry_variant.geometry_id}' is missing one or more HCAL material names."
        )

    # Resolve the material interaction lengths once before walking the stack.
    absorber_lambda_I_mm = resolve_material_lambda_mm(absorber_material, material_library)
    scintillator_lambda_I_mm = resolve_material_lambda_mm(active_material, material_library)
    spacer_lambda_I_mm = resolve_material_lambda_mm(spacer_material, material_library)

    interaction_rows: List[LayerInteractionRow] = []
    cumulative_tau = 0.0
    # Convert each built layer into optical depth and cumulative interaction probability.
    for layer_row in layer_rows:
        absorber_tau = layer_row.absorber_thickness_mm / absorber_lambda_I_mm
        scintillator_tau = layer_row.scintillator_thickness_mm / scintillator_lambda_I_mm
        spacer_tau = 2.0 * layer_row.spacer_thickness_mm / spacer_lambda_I_mm
        delta_tau_layer = absorber_tau + scintillator_tau + spacer_tau
        if delta_tau_layer <= 0.0:
            raise ValueError(
                f"Layer {layer_row.layer_index} in geometry '{geometry_variant.geometry_id}' "
                "has non-positive optical depth."
            )

        cumulative_tau += delta_tau_layer
        cumulative_probability = 1.0 - math.exp(-cumulative_tau)
        interaction_rows.append(
            LayerInteractionRow(
                layer_index=layer_row.layer_index,
                absorber_thickness_mm=layer_row.absorber_thickness_mm,
                scintillator_thickness_mm=layer_row.scintillator_thickness_mm,
                spacer_thickness_mm=layer_row.spacer_thickness_mm,
                layer_total_thickness_mm=layer_row.layer_total_thickness_mm,
                absorber_lambda_I_mm=absorber_lambda_I_mm,
                scintillator_lambda_I_mm=scintillator_lambda_I_mm,
                spacer_lambda_I_mm=spacer_lambda_I_mm,
                delta_tau_layer=delta_tau_layer,
                cumulative_tau=cumulative_tau,
                cumulative_probability=cumulative_probability,
                lambda_I_eff_layer_mm=layer_row.layer_total_thickness_mm / delta_tau_layer,
                depth_front_mm=layer_row.depth_front_mm,
                depth_back_mm=layer_row.depth_back_mm,
                depth_mid_mm=layer_row.depth_mid_mm,
            )
        )
    return interaction_rows


def interpolate_depth_at_probability(
    layer_rows: List[LayerInteractionRow],
    target_probability: float,
) -> tuple[float, float]:
    """Find the depth where the cumulative interaction probability reaches a target value."""
    if not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must lie strictly between 0 and 1.")
    if not layer_rows:
        raise ValueError("Cannot interpolate an empty interaction-depth curve.")

    # Convert the requested probability into the matching cumulative optical depth.
    target_tau = -math.log(1.0 - target_probability)
    final_tau = layer_rows[-1].cumulative_tau
    if target_tau > final_tau:
        return math.nan, math.nan

    previous_tau = 0.0
    # Find the crossing layer and interpolate the matching physical depth.
    for layer_row in layer_rows:
        if layer_row.cumulative_tau < target_tau:
            previous_tau = layer_row.cumulative_tau
            continue

        tau_in_layer = layer_row.cumulative_tau - previous_tau
        if tau_in_layer <= 0.0:
            return layer_row.depth_front_mm, previous_tau
        tau_fraction = (target_tau - previous_tau) / tau_in_layer
        depth_target_mm = (
            layer_row.depth_front_mm
            + tau_fraction * (layer_row.depth_back_mm - layer_row.depth_front_mm)
        )
        return depth_target_mm, target_tau

    return math.nan, math.nan


def summarize_interaction_depth(
    geometry_variant: GeometryVariant,
    geometry_summary: GeometryLayerSummary,
    layer_rows: List[LayerInteractionRow],
) -> InteractionDepthSummary:
    """Reduce one geometry to the compact interaction-depth scalars."""
    if not layer_rows:
        raise ValueError(f"Geometry '{geometry_variant.geometry_id}' has no interaction rows.")

    # Measure the depths where the cumulative curve reaches the standard percentiles.
    depth_90pct_interaction_mm, depth_90pct_interaction_lambda = interpolate_depth_at_probability(
        layer_rows,
        0.90,
    )
    depth_95pct_interaction_mm, depth_95pct_interaction_lambda = interpolate_depth_at_probability(
        layer_rows,
        0.95,
    )

    return InteractionDepthSummary(
        geometry_id=geometry_variant.geometry_id,
        layer_count=len(layer_rows),
        absorber_material=str(geometry_variant.params.get("absorberMaterial", "")),
        active_material=str(geometry_variant.params.get("activeMaterial", "")),
        spacer_material=str(geometry_variant.params.get("spacerMaterial", "")),
        total_depth_mm=geometry_summary.total_depth_mm,
        total_depth_lambda=layer_rows[-1].cumulative_tau,
        absorber_fraction_by_depth=geometry_summary.absorber_fraction_by_depth,
        depth_90pct_interaction_mm=depth_90pct_interaction_mm,
        depth_90pct_interaction_lambda=depth_90pct_interaction_lambda,
        depth_95pct_interaction_mm=depth_95pct_interaction_mm,
        depth_95pct_interaction_lambda=depth_95pct_interaction_lambda,
    )


def analyze_geometry(
    geometry_variant: GeometryVariant,
    material_library: MaterialLibrary,
) -> tuple[InteractionDepthSummary, List[LayerInteractionRow]]:
    """Run the full geometry-only interaction-depth analysis for one geometry."""
    # Build the layer stack once, then derive both the summary and detailed curve from it.
    layer_rows = build_layer_stack(geometry_variant)
    geometry_summary = summarize_layer_stack(layer_rows)
    layer_interaction_rows = build_layer_interaction_rows(geometry_variant, material_library, layer_rows)
    interaction_summary = summarize_interaction_depth(
        geometry_variant,
        geometry_summary,
        layer_interaction_rows,
    )
    return interaction_summary, layer_interaction_rows


def _format_summary_value(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def _json_float_or_null(value: float) -> float | None:
    if math.isnan(value):
        return None
    return value


def build_summary_payload(
    interaction_summary: InteractionDepthSummary,
    final_interaction_probability: float,
) -> dict[str, object]:
    # Write percentile fields as null when the requested depth lies beyond the HCAL.
    return {
        "geometry_id": interaction_summary.geometry_id,
        "layer_count": interaction_summary.layer_count,
        "absorber_material": interaction_summary.absorber_material,
        "active_material": interaction_summary.active_material,
        "spacer_material": interaction_summary.spacer_material,
        "total_depth_mm": interaction_summary.total_depth_mm,
        "total_depth_lambda": interaction_summary.total_depth_lambda,
        "absorber_fraction_by_depth": interaction_summary.absorber_fraction_by_depth,
        "depth_90pct_interaction_mm": _json_float_or_null(
            interaction_summary.depth_90pct_interaction_mm
        ),
        "depth_90pct_interaction_lambda": _json_float_or_null(
            interaction_summary.depth_90pct_interaction_lambda
        ),
        "depth_95pct_interaction_mm": _json_float_or_null(
            interaction_summary.depth_95pct_interaction_mm
        ),
        "depth_95pct_interaction_lambda": _json_float_or_null(
            interaction_summary.depth_95pct_interaction_lambda
        ),
        "final_interaction_probability": final_interaction_probability,
    }


def write_summary_json(
    output_directory: Path,
    interaction_summary: InteractionDepthSummary,
    final_interaction_probability: float,
) -> Path:
    """Write the geometry-level interaction-depth summary."""
    # Build and write the compact JSON summary for this geometry.
    output_path = output_directory / "summary.json"
    summary_payload = build_summary_payload(
        interaction_summary,
        final_interaction_probability,
    )
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary_payload, output_file, indent=2)
        output_file.write("\n")
    return output_path


def write_layers_csv(
    output_directory: Path,
    layer_rows: List[LayerInteractionRow],
) -> Path:
    """Write the full layer-by-layer interaction curve."""
    # Write the full analyzed layer curve so later studies can reuse it directly.
    output_path = output_directory / "layers.csv"
    field_names = [
        "layer_index",
        "absorber_thickness_mm",
        "scintillator_thickness_mm",
        "spacer_thickness_mm",
        "layer_total_thickness_mm",
        "absorber_lambda_I_mm",
        "scintillator_lambda_I_mm",
        "spacer_lambda_I_mm",
        "delta_tau_layer",
        "cumulative_tau",
        "cumulative_probability",
        "lambda_I_eff_layer_mm",
        "depth_front_mm",
        "depth_mid_mm",
        "depth_back_mm",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        csv_writer = csv.DictWriter(output_file, fieldnames=field_names)
        csv_writer.writeheader()
        # Write one CSV row for each analyzed HCAL layer.
        for layer_row in layer_rows:
            csv_writer.writerow(
                {
                    "layer_index": layer_row.layer_index,
                    "absorber_thickness_mm": layer_row.absorber_thickness_mm,
                    "scintillator_thickness_mm": layer_row.scintillator_thickness_mm,
                    "spacer_thickness_mm": layer_row.spacer_thickness_mm,
                    "layer_total_thickness_mm": layer_row.layer_total_thickness_mm,
                    "absorber_lambda_I_mm": layer_row.absorber_lambda_I_mm,
                    "scintillator_lambda_I_mm": layer_row.scintillator_lambda_I_mm,
                    "spacer_lambda_I_mm": layer_row.spacer_lambda_I_mm,
                    "delta_tau_layer": layer_row.delta_tau_layer,
                    "cumulative_tau": layer_row.cumulative_tau,
                    "cumulative_probability": layer_row.cumulative_probability,
                    "lambda_I_eff_layer_mm": layer_row.lambda_I_eff_layer_mm,
                    "depth_front_mm": layer_row.depth_front_mm,
                    "depth_mid_mm": layer_row.depth_mid_mm,
                    "depth_back_mm": layer_row.depth_back_mm,
                }
            )
    return output_path


def main() -> int:
    """Analyze the requested geometries and write the output files."""
    arguments = parse_arguments()
    if not arguments.geometry_json:
        raise SystemExit("Provide at least one --geometry-json path.")

    # Load the shared material table once for all requested geometries.
    material_library = load_material_library_for_interaction_depth()
    # Analyze each geometry, then write both output formats.
    for geometry_json_path in arguments.geometry_json:
        geometry_variant = _load_geometry_variant_from_json_path(geometry_json_path)
        interaction_summary, interaction_rows = analyze_geometry(
            geometry_variant,
            material_library,
        )
        output_directory = OUTPUT_DIRECTORY / interaction_summary.geometry_id
        output_directory.mkdir(parents=True, exist_ok=True)
        final_interaction_probability = interaction_rows[-1].cumulative_probability
        summary_json_path = write_summary_json(
            output_directory,
            interaction_summary,
            final_interaction_probability,
        )
        layers_csv_path = write_layers_csv(output_directory, interaction_rows)
        print(
            f"{interaction_summary.geometry_id}: "
            f"layers={interaction_summary.layer_count} "
            f"total_depth_mm={interaction_summary.total_depth_mm:.6f} "
            f"total_depth_lambda={interaction_summary.total_depth_lambda:.6f} "
            f"p_interact_final={final_interaction_probability:.6f}"
        )
        print(
            f"  depth_90pct_interaction_mm="
            f"{_format_summary_value(interaction_summary.depth_90pct_interaction_mm)} "
            f"depth_90pct_interaction_lambda="
            f"{_format_summary_value(interaction_summary.depth_90pct_interaction_lambda)} "
            f"depth_95pct_interaction_mm="
            f"{_format_summary_value(interaction_summary.depth_95pct_interaction_mm)} "
            f"depth_95pct_interaction_lambda="
            f"{_format_summary_value(interaction_summary.depth_95pct_interaction_lambda)}"
        )
        print(f"  summary_json={summary_json_path}")
        print(f"  layers_csv={layers_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
