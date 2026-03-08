#!/usr/bin/env python3
"""Clean generated artifacts under the standalone new/ tree."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

def remove_path(path: Path, dry_run: bool) -> None:
    """Delete a file or directory tree, respecting dry-run."""
    if dry_run:
        print(f"[dry-run] would remove {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
        print(f"removed dir {path}")
    else:
        path.unlink()
        print(f"removed file {path}")


def clear_directory(root: Path, *, dry_run: bool) -> None:
    """Remove all generated children from ``root`` while keeping Git sentinels."""
    if not root.exists():
        return
    for child in root.iterdir():
        name = child.name
        if name.startswith(".git"):
            continue
        remove_path(child, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove generated geometry/data artifacts.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files/directories that would be removed without deleting anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    targets: List[Path] = [
        project_root / "geometries" / "generated",
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "data" / "manifests",
    ]

    for target in targets:
        clear_directory(target, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
