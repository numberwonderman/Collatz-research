#!/usr/bin/env python3
"""
Exports a Swiss Cheese parameter-cube scan as structured JSON, for the
collatz-box-universes 3D visualizer (box-universe-viewer.html) to consume.

Both repos share the same generalized Collatz model, parameterized by a
cube of (a, b, c) values (n -> n/a if n = 0 mod a, else n -> b*n + c).
This script runs collatz-research's Benford's-Law statistical scan over
that cube and writes one JSON point per (a, b, c), keyed the same way the
visualizer keys its rendered cubes, so results can be loaded directly.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from src.pipeline import SwissCheeseParameterScanner

SCHEMA_VERSION = "1.0"


def build_bridge_payload(
    results: List[Dict],
    cube_center: Tuple[int, int, int],
    side_length: int,
    initial_range: Tuple[int, int],
    max_iterations: int,
) -> Dict:
    """Reshape SwissCheeseParameterScanner results into the bridge format."""
    points = []
    for result in results:
        a, b, c = result["parameters"]
        points.append({
            "a": a,
            "b": b,
            "c": c,
            "total_samples": result.get("total_samples"),
            "mad": result.get("mad"),
            "chi_squared_statistic": result.get("chi_squared_statistic"),
            "p_value": result.get("p_value"),
            "ks_d_max": result.get("ks_d_max"),
            "dmix_variance": result.get("dmix_variance"),
            "digital_mixing_speed": result.get("digital_mixing_speed"),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "collatz-research",
        "cube_center": list(cube_center),
        "side_length": side_length,
        "initial_range": list(initial_range),
        "max_iterations": max_iterations,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "points": points,
    }


def run_export(
    cube_center: Tuple[int, int, int],
    side_length: int,
    initial_range: Tuple[int, int],
    max_iterations: int,
    output_path: Path,
) -> Dict:
    scanner = SwissCheeseParameterScanner(cube_center, side_length)
    scanner.add_trivial_patterns()
    results = scanner.scan_cube(initial_range=initial_range, max_iterations=max_iterations)

    payload = build_bridge_payload(results, cube_center, side_length, initial_range, max_iterations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Export a Swiss Cheese parameter-cube scan as JSON for the "
                     "collatz-box-universes 3D visualizer."
    )
    parser.add_argument("--center", type=int, nargs=3, metavar=("A", "B", "C"), default=(21, 14, 12),
                         help="Cube center (a, b, c). Default: 21 14 12")
    parser.add_argument("--side", type=int, default=5, help="Cube side length (odd). Default: 5")
    parser.add_argument("--range", type=int, nargs=2, metavar=("START", "END"), default=(1, 10000),
                         help="Initial N range to sample. Default: 1 10000")
    parser.add_argument("--max-iterations", type=int, default=2000,
                         help="Max iterations per sequence. Default: 2000")
    parser.add_argument("--output", type=str, default="bridge/cube_export.json",
                         help="Output JSON path. Default: bridge/cube_export.json")
    args = parser.parse_args()

    payload = run_export(
        cube_center=tuple(args.center),
        side_length=args.side,
        initial_range=tuple(args.range),
        max_iterations=args.max_iterations,
        output_path=Path(args.output),
    )

    print(f"Exported {len(payload['points'])} cube points to {args.output}")


if __name__ == "__main__":
    main()
