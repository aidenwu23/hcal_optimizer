#!/usr/bin/env python3
"""Run a muon control simulation and derive a detection threshold."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CALIBRATION_MACRO_PATH = PROJECT_DIRECTORY / "simulation" / "analysis" / "calibrate_fpr.C"
DEFAULT_GUN_DIRECTION = "0 0 -1"
DEFAULT_GUN_POSITION = "0 0 0"


def run_command(command: List[str], label: str) -> None:
    quoted = " ".join(shlex.quote(token) for token in command)
    print(f"[{label}] {quoted}")
    subprocess.run(command, check=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate muon false-positive threshold")
    parser.add_argument("--compact-xml", required=True, help="Path to compact XML geometry")
    parser.add_argument("--raw-out", required=True, help="Output raw EDM4hep file")
    parser.add_argument("--events-out", required=True, help="Output processed events file")
    parser.add_argument("--json-out", required=True, help="Output calibration JSON file")

    parser.add_argument("--ddsim", default="ddsim", help="ddsim executable")
    parser.add_argument("--process-bin", default="./build/bin/process", help="Processor executable")
    parser.add_argument("--root-bin", default="root", help="ROOT executable")

    parser.add_argument("--gun-energy", default="10*GeV", help="Gun energy expression")
    parser.add_argument("--gun-particle", default="mu-", help="Control particle")
    parser.add_argument("--n-events", type=int, default=2000, help="Control event count")

    parser.add_argument("--metric", default="sim_E", help="Metric expression in events tree")
    parser.add_argument("--target-fpr", type=float, default=0.01, help="Target false-positive rate")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    compact_xml_path = Path(arguments.compact_xml).expanduser().resolve()
    raw_output_path = Path(arguments.raw_out).expanduser().resolve()
    events_output_path = Path(arguments.events_out).expanduser().resolve()
    json_output_path = Path(arguments.json_out).expanduser().resolve()

    if not compact_xml_path.exists():
        print(f"ERROR: missing compact XML: {compact_xml_path}", file=sys.stderr)
        return 2
    if not CALIBRATION_MACRO_PATH.exists():
        print(f"ERROR: missing macro: {CALIBRATION_MACRO_PATH}", file=sys.stderr)
        return 2

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    events_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

    ddsim_command: List[str] = [
        arguments.ddsim,
        "--compactFile",
        str(compact_xml_path),
        "--outputFile",
        str(raw_output_path),
        "--numberOfEvents",
        str(arguments.n_events),
        "--enableGun",
        "--gun.particle",
        arguments.gun_particle,
        "--gun.energy",
        arguments.gun_energy,
        "--gun.direction",
        DEFAULT_GUN_DIRECTION,
        "--gun.position",
        DEFAULT_GUN_POSITION,
        "--part.keepAllParticles",
        "true",
        "--part.minimalKineticEnergy",
        "0*MeV",
        "--part.minDistToParentVertex",
        "0*mm",
    ]

    process_command: List[str] = [
        arguments.process_bin,
        str(raw_output_path),
        "--out",
        str(events_output_path),
    ]

    macro_call = (
        f'{CALIBRATION_MACRO_PATH}('
        f'"{events_output_path}","{json_output_path}","{arguments.metric}",{arguments.target_fpr})'
    )
    macro_command: List[str] = [arguments.root_bin, "-l", "-b", "-q", macro_call]

    run_command(ddsim_command, "ddsim")
    run_command(process_command, "process")
    run_command(macro_command, "macro")

    if not json_output_path.exists():
        print(f"ERROR: missing calibration output: {json_output_path}", file=sys.stderr)
        return 3

    with json_output_path.open("r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    if payload.get("threshold_GeV") is None:
        print("ERROR: calibration JSON missing threshold_GeV", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
