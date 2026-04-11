#!/usr/bin/env python3
"""
Load G4GPS beam specs and format macros for DDsim.

Example:
python3 conductor.py --spec geometries/sweeps/nhcal.yaml \
  --g4gps-spec simulation/g4gps/neutron_0.1-3_GeV_KE.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional

import yaml

VALID_X_AXES = {"kinetic_energy_GeV", "momentum_GeV"}

PARTICLE_MASS_GEV = {
    "neutron": 0.93956542052,
    "proton": 0.93827208816,
    "kaon0l": 0.497611,
    "pi-": 0.13957039,
    "pi+": 0.13957039,
    "pion-": 0.13957039,
    "pion+": 0.13957039,
    "pion0": 0.1349768,
    "pi0": 0.1349768,
    "mu-": 0.1056583755,
    "mu+": 0.1056583755,
    "electron": 0.00051099895,
    "e-": 0.00051099895,
    "positron": 0.00051099895,
    "e+": 0.00051099895,
    "photon": 0.0,
    "gamma": 0.0,
}


@dataclass(frozen=True)
class SpectrumPoint:
    x_value_GeV: float
    weight: float


@dataclass(frozen=True)
class G4GPSSpec:
    spec_id: str
    particle: str
    position: str
    direction: str
    x_axis: str
    interpolation: str
    points: List[SpectrumPoint]
    x_min_GeV: float
    x_max_GeV: float
    event_count: Optional[int]


def _parse_vector_text(vector_text: str, *, label: str) -> List[float]:
    values = [token for token in vector_text.split() if token]
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly 3 numbers.")
    return [float(value) for value in values]


def _parse_vector_value(raw_value: object, *, label: str) -> str:
    if isinstance(raw_value, str):
        _parse_vector_text(raw_value, label=label)
        return raw_value
    if isinstance(raw_value, list) and len(raw_value) == 3:
        return " ".join(str(float(value)) for value in raw_value)
    raise ValueError(f"{label} must be a 3-value string or list.")


def _particle_mass_gev(particle: str) -> float:
    particle_key = particle.strip().lower()
    if particle_key not in PARTICLE_MASS_GEV:
        raise ValueError(f"Unsupported particle for momentum_GeV spec: {particle}")
    return PARTICLE_MASS_GEV[particle_key]


# Convert a momentum-axis point into the energy-axis value GPS expects for stable interpolation.
def _x_value_to_energy_gev(particle: str, x_axis: str, x_value_gev: float) -> float:
    if x_axis == "kinetic_energy_GeV":
        return x_value_gev

    mass_gev = _particle_mass_gev(particle)
    if mass_gev == 0.0:
        return x_value_gev
    return math.hypot(x_value_gev, mass_gev) - mass_gev


# Parse the selected spectrum axis and enforce increasing x values.
def _load_spectrum_points(spec_path: Path, x_axis: str, raw_spectrum: dict[object, object]) -> List[SpectrumPoint]:
    raw_x_values = raw_spectrum.get(x_axis)
    raw_weights = raw_spectrum.get("weights")
    
    if not isinstance(raw_x_values, list) or not isinstance(raw_weights, list):
        raise ValueError(f"{spec_path} energy_spectrum must define list {x_axis} and weights.")
    if not raw_x_values:
        raise ValueError(f"{spec_path} energy_spectrum must not be empty.")
    if len(raw_x_values) != len(raw_weights):
        raise ValueError(f"{spec_path} energy_spectrum {x_axis} and weights must have the same length.")

    points: List[SpectrumPoint] = []
    previous_x_value_gev = None
    for raw_x_value, raw_weight in zip(raw_x_values, raw_weights):
        x_value_gev = float(raw_x_value)
        weight = float(raw_weight)
        if previous_x_value_gev is not None and x_value_gev <= previous_x_value_gev:
            raise ValueError(f"{spec_path} points must be in ascending x-axis order.")
        points.append(SpectrumPoint(x_value_GeV=x_value_gev, weight=weight))
        previous_x_value_gev = x_value_gev
    return points


def load_g4gps_spec(spec_path: Path) -> G4GPSSpec:
    # Read one G4GPS YAML spec and validate it before DDsim uses it.
    with spec_path.open("r", encoding="utf-8") as spec_file:
        payload = yaml.safe_load(spec_file) or {}

    # Guards and checks
    if not isinstance(payload, dict):
        raise ValueError(f"{spec_path} must contain a top-level mapping.")
    spec_id = str(payload.get("spec_id", "")).strip()
    if not spec_id:
        raise ValueError(f"{spec_path} is missing spec_id.")
    particle = str(payload.get("particle", "")).strip()
    if not particle:
        raise ValueError(f"{spec_path} is missing particle.")

    position = _parse_vector_value(payload.get("position", "0 0 0"), label="position")
    direction = _parse_vector_value(payload.get("direction", "0 0 -1"), label="direction")
    x_axis = str(payload.get("x_axis", "kinetic_energy_GeV")).strip()
    
    if x_axis not in VALID_X_AXES:
        raise ValueError(f"{spec_path} x_axis must be kinetic_energy_GeV or momentum_GeV.")

    raw_spectrum = payload.get("energy_spectrum")
    if not isinstance(raw_spectrum, dict):
        raise ValueError(f"{spec_path} must define energy_spectrum.")
    points = _load_spectrum_points(spec_path, x_axis, raw_spectrum)

    interpolation = str(payload.get("interpolation", "Lin")).strip() or "Lin"
    raw_event_count = payload.get("events")
    event_count = None if raw_event_count is None else int(raw_event_count)

    if event_count is not None and event_count <= 0:
        raise ValueError(f"{spec_path} events must be positive.")

    return G4GPSSpec(
        spec_id=spec_id,
        particle=particle,
        position=position,
        direction=direction,
        x_axis=x_axis,
        interpolation=interpolation,
        points=points,
        x_min_GeV=points[0].x_value_GeV,
        x_max_GeV=points[-1].x_value_GeV,
        event_count=event_count,
    )


def build_gps_macro_text(
    spec: G4GPSSpec,
    event_count: int,
) -> str:
    # Format one DDsim GPS macro from the loaded spec.
    position_values = _parse_vector_text(spec.position, label="position")
    direction_values = _parse_vector_text(spec.direction, label="direction")

    lines = [
        f"/gps/particle {spec.particle}",
        "/gps/number 1",
        "/gps/pos/type Point",
        f"/gps/pos/centre {position_values[0]:.12g} {position_values[1]:.12g} {position_values[2]:.12g} mm",
        f"/gps/direction {direction_values[0]:.12g} {direction_values[1]:.12g} {direction_values[2]:.12g}",
        "/gps/ene/type Arb",
        "/gps/ene/emspec true",
        "/gps/ene/diffspec true",
        "/gps/hist/type arb",
    ]

    # Always emit an energy-axis histogram because GPS momentum interpolation crashes here.
    for point in spec.points:
        energy_gev = _x_value_to_energy_gev(spec.particle, spec.x_axis, point.x_value_GeV)
        lines.append(f"/gps/hist/point {energy_gev * 1000.0:.12g} {point.weight:.12g}")

    lines.extend(
        [
            f"/gps/hist/inter {spec.interpolation}",
            f"/run/beamOn {event_count}",
            "",
        ]
    )
    return "\n".join(lines)
