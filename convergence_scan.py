#!/usr/bin/env python3
"""
Scans a region of the generalized Collatz (a, b, c) parameter space and
reports, for each point, both Benford's-Law digit-conformity stats and the
fraction of sampled starting numbers that actually converge to 1.

bridge_export.py's cube scans found some (a, b, c) with extreme Benford
conformity, but it turned out to be driven almost entirely by divergent
trajectories: a sequence that grows by a roughly constant ratio each step
obeys Benford's Law for the classical reason (log10(ratio) is irrational,
so leading digits equidistribute mod 1), independent of anything
Collatz-specific. This scanner exists to separate that explained case from
points that are both non-trivial (see is_degenerate) and actually
Collatz-like (mostly return to 1), which the simple geometric-growth
argument does not already account for. See RESEARCH_NOTES.md for findings.
"""
import argparse
import itertools
import json
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from src.collatz_generator import generalized_collatz
from src.benford_analyzer import get_leading_digit
from src.statistical_tests import analyze_conformity


def analyze_point(a: int, b: int, c: int, initial_range: Tuple[int, int], max_iterations: int) -> Dict:
    """Runs both the Benford digit analysis and convergence-rate check for one (a, b, c)."""
    start, end = initial_range
    all_digits: List[int] = []
    converged = 0
    total = 0

    for n in range(start, end + 1):
        if n == 1:
            continue
        seq = generalized_collatz(n, a, b, c, max_iterations)
        if not seq:
            continue
        total += 1
        if seq[-1] == 1:
            converged += 1
        for term in seq:
            if term > 1:
                all_digits.append(get_leading_digit(term))

    counts = Counter(all_digits)
    observed = {d: counts.get(d, 0) for d in range(1, 10)}
    stats = analyze_conformity(observed)

    return {
        "a": a, "b": b, "c": c,
        "convergence_rate": converged / total if total else 0.0,
        "sampled_n": total,
        "mad": stats.get("mad"),
        "chi_squared_statistic": stats.get("chi_squared_statistic"),
        "p_value": stats.get("p_value"),
        "ks_d_max": stats.get("ks_d_max"),
        "total_digit_samples": stats.get("total_samples"),
    }


def is_degenerate(point: Dict) -> Optional[str]:
    """
    Flags (a, b, c) where the odd-branch shortcut (maximal division by a of
    b*n + c) collapses the map to a much simpler one, which explains away any
    strong Benford conformity or convergence without it being Collatz-specific:
      - c == a and b == 2*a: collapses to n -> 2n+1 (the classical doubling
        map, the textbook Benford-conforming sequence).
      - b % a == 0 or c % a == 0: the odd-branch result's residue mod a does
        not depend on n, which similarly removes the "mixing" the chaotic-map
        framing is meant to be about.
    Returns a short reason string, or None if the point looks generic.
    """
    a = point["a"]
    if point["c"] == a and point["b"] == 2 * a:
        return "c=a, b=2a (collapses to n -> 2n+1)"
    if point["b"] % a == 0:
        return "b % a == 0 (odd-branch residue mod a is constant)"
    if point["c"] % a == 0:
        return "c % a == 0 (odd-branch residue mod a is constant)"
    return None


def parse_range(spec: str) -> range:
    """Parses 'start:end' (inclusive) into a range()."""
    start_str, end_str = spec.split(":")
    return range(int(start_str), int(end_str) + 1)


def scan_region(a_range: Iterable[int], b_range: Iterable[int], c_range: Iterable[int],
                 initial_range: Tuple[int, int], max_iterations: int, verbose: bool = True) -> List[Dict]:
    results = []
    combos = [
        (a, b, c) for a, b, c in itertools.product(a_range, b_range, c_range)
        if a != 1 and not (b == 0 and c == 0) and not (b < 0 and c < 0)
    ]
    for i, (a, b, c) in enumerate(combos, 1):
        try:
            point = analyze_point(a, b, c, initial_range, max_iterations)
            point["degenerate_reason"] = is_degenerate(point)
            results.append(point)
        except Exception as e:
            if verbose:
                print(f"  Error at ({a},{b},{c}): {e}")
        if verbose and (i % 50 == 0 or i == len(combos)):
            print(f"[{i}/{len(combos)}] scanned...")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Scan a generalized-Collatz (a, b, c) region for both Benford's-Law "
                    "digit conformity and convergence rate, flagging points whose "
                    "odd-branch shortcut trivially collapses the dynamics."
    )
    parser.add_argument("--a-range", type=parse_range, default=range(2, 11), metavar="START:END")
    parser.add_argument("--b-range", type=parse_range, default=range(2, 11), metavar="START:END")
    parser.add_argument("--c-range", type=parse_range, default=range(1, 10), metavar="START:END")
    parser.add_argument("--range", type=int, nargs=2, default=(2, 3000), metavar=("START", "END"),
                         help="Initial N range to sample. Default: 2 3000")
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--min-convergence", type=float, default=0.9,
                         help="Convergence-rate threshold (0-1) for the summary table. Default: 0.9")
    parser.add_argument("--output", type=str, default="convergence_scan_results.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results = scan_region(
        args.a_range, args.b_range, args.c_range,
        tuple(args.range), args.max_iterations, verbose=not args.quiet,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nScanned {len(results)} points. Results saved to {args.output}")

    non_degenerate = [r for r in results if r["degenerate_reason"] is None]
    high_conv = [r for r in non_degenerate if r["convergence_rate"] >= args.min_convergence]
    high_conv.sort(key=lambda r: r["mad"])

    print(f"{len(high_conv)} non-degenerate points with convergence_rate >= {args.min_convergence*100:.0f}%:")
    print(f"{'(a,b,c)':<15}{'Conv%':<10}{'MAD':<10}{'ChiSq':<12}{'samples'}")
    for r in high_conv[:20]:
        print(f"({r['a']},{r['b']},{r['c']})".ljust(15) +
              f"{r['convergence_rate']*100:<10.1f}{r['mad']:<10.5f}{r['chi_squared_statistic']:<12.2f}{r['sampled_n']}")


if __name__ == "__main__":
    main()
