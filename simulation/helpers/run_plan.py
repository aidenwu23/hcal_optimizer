#!/usr/bin/env python3
"""Run planning and deterministic run identity."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .geometry_index import GeometryVariant
from .spectrum import G4GPSSpec, load_g4gps_spec

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"

PARTICLE_PDG: Dict[str, int] = {
    "neutron": 2112,
    "proton": 2212,
    "kaon0L": 130,
    "pi-": -211,
    "pi+": 211,
    "pion-": -211,
    "pion+": 211,
    "pion0": 111,
    "pi0": 111,
    "mu-": -13,
    "mu+": 13,
    "electron": 11,
    "e-": 11,
    "positron": -11,
    "e+": -11,
    "photon": 22,
    "gamma": 22,
}

def lookup_pdg(particle: str) -> Optional[int]:
    """Translate a human-readable particle name into a PDG code when possible."""
    particle_name = particle.strip()
    if particle_name in PARTICLE_PDG:
        return PARTICLE_PDG[particle_name]
    return PARTICLE_PDG.get(particle_name.lower())


# One fully expanded simulation run, including all file paths that later execution steps need.
@dataclass
class RunPlan:
    geometry_variant: GeometryVariant
    gun_particle: str
    beam_mode: str
    beam_label: str
    momentum_GeV: Optional[float]
    g4gps_spec_path: Optional[Path]
    gun_direction: Optional[str]
    gun_position: Optional[str]
    seed: Optional[int]
    n_events: int
    run_id: str
    run_id_int: int
    raw_path: Path
    events_path: Path
    meta_path: Path
    calibration_path: Path
    performance_path: Path
    macro_path: Optional[Path]
    expected_pdg: Optional[int]


# Per-run bookkeeping.
@dataclass
class RunRecord:
    plan: RunPlan
    status: str
    error: Optional[str] = None
    ddsim_seconds: Optional[float] = None
    process_seconds: Optional[float] = None
    performance_seconds: Optional[float] = None
    meta_seconds: Optional[float] = None


# Build a deterministic run identity from the geometry, beam settings, and processing extras.
def compute_run_id(
    geometry_id: str,
    particle: str,
    beam_label: str,
    seed: Optional[int],
    event_count: int,
    extra_tokens: Sequence[str],
) -> Tuple[str, int]:
    seed_token = "noseed" if seed is None else str(seed)
    payload = "|".join(
        [
            geometry_id,
            particle,
            beam_label,
            seed_token,
            str(event_count),
            *extra_tokens,
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    run_id = f"run{digest}"
    run_id_int = int(digest[:8], 16) & 0x7FFFFFFF
    return run_id, run_id_int


def _build_run_paths(geometry_id: str, run_id: str) -> tuple[Path, Path]:
    raw_directory = DATA_DIRECTORY / "raw" / geometry_id
    processed_directory = DATA_DIRECTORY / "processed" / geometry_id / run_id
    return raw_directory, processed_directory


# Build one run plan that uses the external G4GPS spec as the beam source of truth.
def _build_g4gps_run_plan(
    geometry_variant: GeometryVariant,
    particle: str,
    expected_pdg: Optional[int],
    spec: G4GPSSpec,
    spec_path: Path,
    seed: Optional[int],
    event_count: int,
    run_id_tokens: Sequence[str],
) -> RunPlan:
    run_id, run_id_int = compute_run_id(
        geometry_variant.geometry_id,
        particle,
        f"g4gps:{spec.spec_id}",
        seed,
        event_count,
        run_id_tokens,
    )
    raw_directory, processed_directory = _build_run_paths(geometry_variant.geometry_id, run_id)
    return RunPlan(
        geometry_variant=geometry_variant,
        gun_particle=particle,
        beam_mode="g4gps_spec",
        beam_label=spec.spec_id,
        momentum_GeV=None,
        g4gps_spec_path=spec_path,
        gun_direction=None,
        gun_position=None,
        seed=seed,
        n_events=event_count,
        run_id=run_id,
        run_id_int=run_id_int,
        raw_path=raw_directory / f"{run_id}.edm4hep.root",
        events_path=processed_directory / "events.root",
        meta_path=processed_directory / "meta.json",
        calibration_path=processed_directory / "calibration.json",
        performance_path=processed_directory / "performance.json",
        macro_path=processed_directory / "gps.mac",
        expected_pdg=expected_pdg,
    )


# Build one fixed-gun run plan (i.e. one singular energy)
def _build_fixed_gun_run_plan(
    geometry_variant: GeometryVariant,
    particle: str,
    expected_pdg: Optional[int],
    momentum_gev: float,
    gun_direction: str,
    gun_position: str,
    seed: Optional[int],
    event_count: int,
    run_id_tokens: Sequence[str],
) -> RunPlan:
    run_id, run_id_int = compute_run_id(
        geometry_variant.geometry_id,
        particle,
        f"momentum:{momentum_gev:.6f}",
        seed,
        event_count,
        run_id_tokens,
    )
    raw_directory, processed_directory = _build_run_paths(geometry_variant.geometry_id, run_id)
    return RunPlan(
        geometry_variant=geometry_variant,
        gun_particle=particle,
        beam_mode="fixed_gun",
        beam_label=f"{momentum_gev:.6f}GeV",
        momentum_GeV=momentum_gev,
        g4gps_spec_path=None,
        gun_direction=gun_direction,
        gun_position=gun_position,
        seed=seed,
        n_events=event_count,
        run_id=run_id,
        run_id_int=run_id_int,
        raw_path=raw_directory / f"{run_id}.edm4hep.root",
        events_path=processed_directory / "events.root",
        meta_path=processed_directory / "meta.json",
        calibration_path=processed_directory / "calibration.json",
        performance_path=processed_directory / "performance.json",
        macro_path=None,
        expected_pdg=expected_pdg,
    )


# Expand the geometry list, energies, and seeds into the concrete run list that conductor executes.
def build_run_plans(
    args: argparse.Namespace,
    geometry_variants: List[GeometryVariant],
    extra_process_flags: Sequence[str],
) -> List[RunPlan]:
    run_plans: List[RunPlan] = []
    seed_values: List[Optional[int]] = list(args.seeds) if args.seeds is not None else [None]
    requested_particles = [str(particle).strip() for particle in args.gun_particle if str(particle).strip()]
    momentum_values = list(args.gun_momentum)
    g4gps_spec_path = Path(args.g4gps_spec) if getattr(args, "g4gps_spec", None) else None
    g4gps_spec = load_g4gps_spec(g4gps_spec_path) if g4gps_spec_path else None

    for geometry_variant in geometry_variants:
        # Include the processor extras, geometry tag, and detector side in the run identity.
        run_id_tokens = list(extra_process_flags) + [geometry_variant.tag or "", geometry_variant.side]
        gun_direction = geometry_variant.params.get("gun.direction", args.gun_direction)
        gun_position = geometry_variant.params.get("gun.position", args.gun_position)

        plan_particles = requested_particles
        if g4gps_spec is not None:
            plan_particles = [g4gps_spec.particle]

        # Sweep over every requested particle
        for particle in plan_particles:
            expected_pdg = lookup_pdg(particle)
            use_g4gps_spec = g4gps_spec is not None and particle.lower() == g4gps_spec.particle.lower()

            # Build a with the g4gps spec for a continuous energy spectrum.
            if use_g4gps_spec:
                event_count = g4gps_spec.event_count or args.events
                for seed in seed_values:
                    run_plans.append(
                        _build_g4gps_run_plan(
                            geometry_variant=geometry_variant,
                            particle=particle,
                            expected_pdg=expected_pdg,
                            spec=g4gps_spec,
                            spec_path=g4gps_spec_path,
                            seed=seed,
                            event_count=event_count,
                            run_id_tokens=run_id_tokens,
                        )
                    )
                continue

            # For some reason ddsim labels gun energy as total energy but it's actually all converted into momentum.
            for momentum_gev in momentum_values:
                for seed in seed_values:
                    run_plans.append(
                        _build_fixed_gun_run_plan(
                            geometry_variant=geometry_variant,
                            particle=particle,
                            expected_pdg=expected_pdg,
                            momentum_gev=momentum_gev,
                            gun_direction=gun_direction,
                            gun_position=gun_position,
                            seed=seed,
                            event_count=args.events,
                            run_id_tokens=run_id_tokens,
                        )
                    )
    return run_plans
