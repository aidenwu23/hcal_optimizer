#!/usr/bin/env python3
"""Geometry discovery and geometry-derived metadata."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
GEOMETRY_DIRECTORY = PROJECT_DIRECTORY / "geometries"

UNIT_MM: Dict[str, float] = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "um": 0.001,
    "nm": 1e-6,
}


def eval_length_mm(value: object, *, default: float = 0.0) -> float:
    """Evaluate a DD4hep-style length expression in millimeter."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        expression = value.strip()
        if not expression:
            return default
        try:
            return float(expression)
        except ValueError:
            safe_locals = {**UNIT_MM, "pi": math.pi}
            try:
                return float(eval(expression, {"__builtins__": {}}, safe_locals))
            except Exception:
                return default
    return default


@dataclass
class GeometryVariant:
    geometry_id: str
    tag: str
    geometry_directory: Path
    params_path: Path
    xml_path: Path
    spec_path: Path
    params: Dict[str, object]

    @property
    def n_layers(self) -> int:
        return int(self.params.get("nLayers", 0))

    @property
    def side(self) -> str:
        return str(self.params.get("side", "-z"))


def inspect_geometry_rows(python_bin: str, spec_paths: Iterable[Path]) -> List[Dict[str, object]]:
    """Ask the sweep script for the geometry rows implied by the requested specs."""
    sweep_script = GEOMETRY_DIRECTORY / "sweep_geometries.py"
    if not sweep_script.exists():
        raise FileNotFoundError(f"{sweep_script} not found.")
    command = [python_bin, str(sweep_script), "--dry-run"]
    for spec_path in spec_paths:
        command.extend(["--spec", str(spec_path)])
    result = subprocess.run(
        command,
        cwd=PROJECT_DIRECTORY,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        geometry_rows = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Failed to parse sweep_geometries.py dry-run output") from error
    if not isinstance(geometry_rows, list):
        raise RuntimeError("Expected sweep_geometries.py dry-run to return a JSON list")
    return geometry_rows


def load_geometry_variants(
    geometry_rows: Iterable[Dict[str, object]],
    *,
    require_geometry_files: bool,
) -> List[GeometryVariant]:
    """Compile geometry metadata from dry-run sweep rows."""
    geometry_variants: List[GeometryVariant] = []
    for geometry_row in geometry_rows:
        geometry_directory = _resolve_project_path(str(geometry_row["geometry_directory"]))
        params_path = _resolve_project_path(str(geometry_row["json_path"]))
        xml_path = _resolve_project_path(str(geometry_row["xml_path"]))
        parameters = geometry_row.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"Missing parameters in geometry row: {geometry_row}")
        geometry_id = str(geometry_row.get("geometry_id", "")).strip()
        if not geometry_id:
            raise ValueError(f"Missing geometry_id in geometry row: {geometry_row}")
        if geometry_directory.name != geometry_id:
            raise ValueError(f"Geometry directory does not match geometry_id: {geometry_directory}")
        if params_path.parent != geometry_directory:
            raise ValueError(f"Geometry JSON is outside geometry_directory: {params_path}")
        if xml_path.parent != geometry_directory:
            raise ValueError(f"Geometry XML is outside geometry_directory: {xml_path}")
        if require_geometry_files:
            _validate_geometry_files(geometry_id, params_path, xml_path)
        spec_path = _resolve_project_path(str(geometry_row["spec_path"]))
        geometry_variants.append(
            GeometryVariant(
                geometry_id=geometry_id,
                tag=str(geometry_row.get("tag", "")),
                geometry_directory=geometry_directory,
                params_path=params_path,
                xml_path=xml_path,
                spec_path=spec_path,
                params={str(key): value for key, value in parameters.items()},
            )
        )
    return geometry_variants


def derive_thickness_and_zrange(geometry_variant: GeometryVariant) -> Tuple[float, float, float]:
    """Return (thickness_mm, zmin_mm_world, zmax_mm_world) from geometry params."""
    geometry_parameters = geometry_variant.params
    spacer_thickness = eval_length_mm(geometry_parameters.get("t_spacer"), default=0.0)
    absorber_thickness = eval_length_mm(geometry_parameters.get("t_absorber"), default=0.0)
    scintillator_thickness = eval_length_mm(geometry_parameters.get("t_scin"), default=0.0)
    layer_count = geometry_variant.n_layers
    segment_layers = [
        int(geometry_parameters.get("seg1_layers", 0) or 0),
        int(geometry_parameters.get("seg2_layers", 0) or 0),
        int(geometry_parameters.get("seg3_layers", 0) or 0),
    ]

    total_thickness = 0.0
    if sum(segment_layers) > 0:
        for segment_index, segment_layer_count in enumerate(segment_layers, start=1):
            if segment_layer_count <= 0:
                continue
            segment_absorber_thickness = eval_length_mm(
                geometry_parameters.get(f"t_absorber_seg{segment_index}"),
                default=absorber_thickness,
            )
            segment_scintillator_thickness = eval_length_mm(
                geometry_parameters.get(f"t_scin_seg{segment_index}"),
                default=scintillator_thickness,
            )
            segment_spacer_thickness = eval_length_mm(
                geometry_parameters.get(f"t_spacer_seg{segment_index}"),
                default=spacer_thickness,
            )
            total_thickness += segment_layer_count * (
                segment_absorber_thickness + segment_scintillator_thickness + segment_spacer_thickness
            )
    else:
        total_thickness = layer_count * (absorber_thickness + scintillator_thickness + spacer_thickness)

    z_face_mm = eval_length_mm(geometry_parameters.get("zmin"), default=0.0)
    if geometry_variant.side == "-z":
        zmax_world = -z_face_mm
        zmin_world = -(z_face_mm + total_thickness)
    else:
        zmin_world = z_face_mm
        zmax_world = z_face_mm + total_thickness
    return total_thickness, zmin_world, zmax_world


def _resolve_project_path(path_text: str) -> Path:
    raw_path = Path(path_text).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (PROJECT_DIRECTORY / raw_path).resolve()


def _validate_geometry_files(geometry_id: str, params_path: Path, xml_path: Path) -> None:
    if not params_path.exists():
        raise FileNotFoundError(f"Params JSON missing: {params_path}")
    if not xml_path.exists():
        raise FileNotFoundError(f"Compact XML missing: {xml_path}")
    with params_path.open("r", encoding="utf-8") as parameter_file:
        parameter_payload = json.load(parameter_file)
    file_geometry_id = str(parameter_payload.get("geometry_id", "")).strip()
    if file_geometry_id != geometry_id:
        raise ValueError(f"Geometry ID mismatch in {params_path}")
