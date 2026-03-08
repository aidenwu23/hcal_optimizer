#!/usr/bin/env python3
"""Run the geometry-only interaction-depth analysis and ROOT plotting in one step."""
# python analysis/geometry/run_interaction_depth.py --geometry-json geometries/generated/<geom_id>/geometry.json

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.geometry.interaction_depth.interaction_depth import (
    OUTPUT_DIRECTORY,
    _load_geometry_variant_from_json_path,
    analyze_geometry,
    load_material_library_for_interaction_depth,
    write_layers_csv,
    write_summary_json,
)

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
ROOT_MACRO_PATH = (
    PROJECT_DIRECTORY / "analysis" / "geometry" / "interaction_depth" / "plot_interaction_depth.C"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run geometry-only interaction-depth analysis and plotting in one step."
    )
    parser.add_argument(
        "--geometry-json",
        nargs="+",
        required=True,
        help="One or more generated geometry.json paths.",
    )
    parser.add_argument(
        "--root-bin",
        default="root",
        help="ROOT executable to use for the plotting step.",
    )
    return parser.parse_args()


# The ROOT macro is still the plotting backend, so call it after the Python layer table exists.
def run_root_plot(geometry_id: str, root_bin: str) -> Path:
    output_root_path = OUTPUT_DIRECTORY / geometry_id / f"{geometry_id}_interaction_depth.root"
    macro_call = f'{ROOT_MACRO_PATH}("{geometry_id}","{output_root_path}")'
    subprocess.run(
        [root_bin, "-l", "-b", "-q", macro_call],
        cwd=PROJECT_DIRECTORY,
        check=True,
    )
    return output_root_path


def main() -> int:
    arguments = parse_arguments()

    # One material table can serve every requested geometry in the same wrapper call.
    material_library = load_material_library_for_interaction_depth()
    for geometry_json_path in arguments.geometry_json:
        # Run the geometry-only optical-depth model first and keep the machine-readable outputs.
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

        # Turn the layer table into ROOT graphs and histograms for the same geometry.
        output_root_path = run_root_plot(interaction_summary.geometry_id, arguments.root_bin)

        print(
            f"{interaction_summary.geometry_id}: "
            f"summary_json={summary_json_path} "
            f"layers_csv={layers_csv_path} "
            f"interaction_depth_root={output_root_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
