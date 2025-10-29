import sys
import json
import time
import math
from collections import Counter
from typing import Dict, Set, Tuple, List
import itertools

# --- Imports (Assuming these files are correctly updated/exist in 'src/') ---
from src.collatz_generator import generalized_collatz
from src.benford_analyzer import get_leading_digit
from src.statistical_tests import analyze_conformity


def benford_distribution():
    """Returns the Benford expected distribution for digits 1 through 9."""
    return {d: math.log10(1 + 1/d) for d in range(1, 10)}


def calculate_dmix(observed_counts: Dict[int, int]) -> float:
    """
    Calculates Dmix (total variation distance) between observed digit distribution and Benford distribution.
    """
    total = sum(observed_counts.values())
    if total == 0:
        return float('nan')
    
    benford_dist = benford_distribution()
    dmix = 0.0
    
    for d in range(1, 10):
        p_obs = observed_counts.get(d, 0) / total
        p_benford = benford_dist[d]
        dmix += abs(p_obs - p_benford)
        
    dmix /= 2
    return dmix


def digital_mixing_speed(mad: float, sample_size: int) -> float:
    """
    Calculates the digital mixing speed metric as (1/MAD) * log10(sample_size).
    """
    if mad == 0 or sample_size <= 0:
        return float('inf')
    return (1.0 / mad) * math.log10(sample_size)


class SwissCheeseParameterScanner:
    """
    Scans through a parameter cube with 'holes' (trivial states) excluded,
    integrated with Benford's Law analysis pipeline.
    """
    
    def __init__(self, cube_center: Tuple[int, int, int], cube_side_length: int):
        """
        Initialize the scanner with a cube centered at (a0, b0, c0).
        
        Args:
            cube_center: (a, b, c) center point
            cube_side_length: length of cube side (must be odd for symmetric scanning)
        """
        self.center = cube_center
        self.side_length = cube_side_length
        self.holes: Set[Tuple[int, int, int]] = set()
        
        # Calculate bounds
        half = cube_side_length // 2
        self.a_range = range(cube_center[0] - half, cube_center[0] + half + 1)
        self.b_range = range(cube_center[1] - half, cube_center[1] + half + 1)
        self.c_range = range(cube_center[2] - half, cube_center[2] + half + 1)
    
    def add_hole(self, a: int, b: int, c: int):
        """Add a single trivial state (hole) to skip."""
        self.holes.add((a, b, c))
    
    def add_holes_from_list(self, hole_list: List[Tuple[int, int, int]]):
        """Add multiple holes at once."""
        self.holes.update(hole_list)
    
    def add_trivial_patterns(self):
        """Add common trivial/degenerate cases as holes."""
        trivial = []
        
        for a in self.a_range:
            for b in self.b_range:
                for c in self.c_range:
                    # Trivial cases to skip:
                    # 1. b = 0, c = 0 (collapses to 0)
                    if b == 0 and c == 0:
                        trivial.append((a, b, c))
                    
                    # 2. b = 1, c = 0 (identity-like, no transformation)
                    elif b == 1 and c == 0:
                        trivial.append((a, b, c))
                    
                    # 3. c = 0 and b even (immediate division, less interesting)
                    elif c == 0 and b % 2 == 0:
                        trivial.append((a, b, c))
                    
                    # 4. Both b and c negative (typically diverges immediately)
                    elif b < 0 and c < 0:
                        trivial.append((a, b, c))
        
        self.add_holes_from_list(trivial)
        return len(trivial)
    
    def analyze_single_parameter_set(
        self, 
        a: int, 
        b: int, 
        c: int, 
        initial_range: Tuple[int, int],
        max_iterations: int = 1000000
    ) -> Dict:
        """
        Run full Benford analysis for a single (a, b, c) parameter set.
        """
        start, end = initial_range
        all_digits = []
        
        start_time = time.time()
        
        for n in range(start, end + 1):
            if n == 1:
                continue
            sequence = generalized_collatz(n, a, b, c, max_iterations)
            for term in sequence:
                try:
                    if term > 1:
                        digit = get_leading_digit(term)
                        all_digits.append(digit)
                except ValueError:
                    continue
        
        end_time = time.time()
        
        final_counts = Counter(all_digits)
        observed_counts = {d: final_counts.get(d, 0) for d in range(1, 10)}
        
        # Run statistical tests
        results = analyze_conformity(observed_counts)
        
        total_samples = sum(observed_counts.values())
        
        # Add custom metrics
        mad_value = results.get('mad', None)
        if mad_value is not None and mad_value > 0:
            results['digital_mixing_speed'] = digital_mixing_speed(mad_value, total_samples)
        else:
            results['digital_mixing_speed'] = None
        
        results['dmix_variance'] = calculate_dmix(observed_counts)
        
        return {
            "parameters": (a, b, c),
            "initial_range": initial_range,
            "digit_frequencies": observed_counts,
            "total_samples": total_samples,
            "computation_time": end_time - start_time,
            **results
        }
    
    def scan_cube(
        self,
        initial_range: Tuple[int, int] = (1, 1000),
        max_iterations: int = 1000000,
        output_file: str = None
    ) -> List[Dict]:
        """
        Scan through the entire parameter cube, skipping holes.
        
        Args:
            initial_range: Range of initial values to test (start, end)
            max_iterations: Max iterations per sequence
            output_file: Optional file to save results as JSON
            
        Returns:
            List of analysis results for each parameter combination
        """
        total_points = len(self.a_range) * len(self.b_range) * len(self.c_range)
        scanned = 0
        skipped = 0
        results = []
        
        print(f"=== SWISS CHEESE CUBE SCAN ===")
        print(f"Cube center: {self.center}")
        print(f"Cube side length: {self.side_length}")
        print(f"Total parameter points: {total_points}")
        print(f"Holes (trivial states): {len(self.holes)}")
        print(f"Points to scan: {total_points - len(self.holes)}")
        print(f"Initial range: N={initial_range[0]} to {initial_range[1]}")
        print("=" * 50)
        
        for a, b, c in itertools.product(self.a_range, self.b_range, self.c_range):
            # Skip holes (trivial states)
            if (a, b, c) in self.holes:
                skipped += 1
                continue
            
            scanned += 1
            print(f"\n[{scanned}/{total_points - len(self.holes)}] Analyzing (a={a}, b={b}, c={c})...")
            
            try:
                analysis = self.analyze_single_parameter_set(
                    a, b, c, initial_range, max_iterations
                )
                results.append(analysis)
                
                # Print brief summary
                print(f"  ✓ Samples: {analysis['total_samples']:,} | "
      f"Chi-Sq Stat: {analysis.get('chi_squared_statistic', 'N/A'):.4f} | "
      f"Chi-Sq p-value: {analysis.get('p_value', 'N/A')} | "
      f"MAD: {analysis.get('mad', 'N/A'):.5f} | "
      f"KS D-Max: {analysis.get('ks_d_max', 'N/A'):.5f} | "
      f"Dmix: {analysis.get('dmix_variance', 'N/A'):.6f} | "
      f"Digital Mix Speed: {analysis.get('digital_mixing_speed', 'N/A'):.3f} | "
      f"Time: {analysis['computation_time']:.2f}s")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        print(f"\n{'='*50}")
        print(f"Scan complete! Scanned: {scanned}, Skipped: {skipped}")
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results
    
    def find_best_benford_conformity(self, results: List[Dict], top_n: int = 10) -> List[Dict]:
        """
        Find parameter sets with best Benford's Law conformity (lowest MAD).
        """
        # Filter out results without MAD values
        valid_results = [r for r in results if r.get('mad') is not None]
        
        # Sort by MAD (lower is better)
        sorted_results = sorted(valid_results, key=lambda x: x['mad'])
        
        return sorted_results[:top_n]
    
    def find_fastest_mixing(self, results: List[Dict], top_n: int = 10) -> List[Dict]:
        """
        Find parameter sets with fastest digital mixing speed.
        """
        # Filter out results without digital_mixing_speed values
        valid_results = [r for r in results if r.get('digital_mixing_speed') is not None]
        
        # Sort by digital mixing speed (higher is better)
        sorted_results = sorted(valid_results, key=lambda x: x['digital_mixing_speed'], reverse=True)
        
        return sorted_results[:top_n]


def print_summary_table(results: List[Dict], title: str):
    """Print a formatted summary table of results."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'(a, b, c)':<20}{'MAD':<12}{'Dmix':<12}{'Mix Speed':<15}{'Samples':<10}")
    print(f"{'-'*80}")
    
    for i, result in enumerate(results, 1):
        params = result['parameters']
        mad = result.get('mad', float('nan'))
        dmix = result.get('dmix_variance', float('nan'))
        mix_speed = result.get('digital_mixing_speed', float('nan'))
        samples = result.get('total_samples', 0)
        
        print(f"{i:<6}{str(params):<20}{mad:<12.6f}{dmix:<12.6f}{mix_speed:<15.2f}{samples:<10,}")


# --- Example Usage ---
if __name__ == "__main__":
    # Create a scanner centered at (12, 16, -4) - your test config
    scanner = SwissCheeseParameterScanner(
        cube_center=(4, 5, 1), 
        cube_side_length=5
          # Scans from (10, 14, -6) to (14, 18, -2)
    )
    
    # Add trivial patterns as holes
    num_holes = scanner.add_trivial_patterns()
    print(f"Added {num_holes} trivial states as holes\n")
    
    # You can manually add specific known trivial cases
    # scanner.add_hole(12, 14, -4)
    
    # Scan the cube
    results = scanner.scan_cube(
        initial_range=(1, 100000),  # Smaller range for testing
        max_iterations=100000,
        output_file="swiss_cheese_results.json"
    )
    
    # Find best Benford conformity
    best_benford = scanner.find_best_benford_conformity(results, top_n=5)
    print_summary_table(best_benford, "TOP 5: BEST BENFORD'S LAW CONFORMITY (Lowest MAD)")
    
    # Find fastest mixing
    fastest_mix = scanner.find_fastest_mixing(results, top_n=5)
    print_summary_table(fastest_mix, "TOP 5: FASTEST DIGITAL MIXING SPEED")