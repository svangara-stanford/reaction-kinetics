#!/usr/bin/env python3
"""Run Level 1 descriptive kinetics pipeline."""

import argparse
import sys
from pathlib import Path

# Run from repo root so that reaction_kinetics is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reaction_kinetics.config import (
    DEFAULT_ERODE_BOUNDARY_PX,
    DEFAULT_SMOOTH_WINDOW,
    get_data_root,
    get_output_root,
)
from reaction_kinetics.pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run descriptive reaction-kinetics pipeline")
    parser.add_argument("--data-root", type=str, default=None, help="Data directory (default: data)")
    parser.add_argument("--output-root", type=str, default=None, help="Output directory (default: outputs)")
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW, help="Savitzky-Golay window (odd)")
    parser.add_argument("--erode-boundary-px", type=int, default=DEFAULT_ERODE_BOUNDARY_PX, help="Erode boundary pixels")
    parser.add_argument("--particle-ids", type=int, nargs="*", default=None, metavar="ID", help="Restrict to these particle IDs (e.g. 1 2 3 4 5 6 7 8)")
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    output_root = get_output_root(args.output_root)
    if not data_root.exists():
        print(f"Data root not found: {data_root}", file=sys.stderr)
        return 1

    summary = run(
        data_root=data_root,
        output_root=output_root,
        smooth_window=args.smooth_window,
        erode_boundary_px=args.erode_boundary_px,
        particle_ids_include=args.particle_ids if args.particle_ids else None,
    )

    print("Run summary:")
    print(f"  Particles: {len(summary.per_particle)}")
    print(f"  Figures: {len(summary.figure_paths)}")
    print(f"  Tables: {len(summary.table_paths)}")
    for p in summary.figure_paths:
        print(f"    {p}")
    for p in summary.table_paths:
        print(f"    {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
