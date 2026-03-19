import sys
import json
import time
import math
from collections import Counter
from typing import Dict, Set, Tuple, List
import itertools

# --- Corrected Mod-Based Generalized Collatz ---
def generalized_collatz(n: int, a: int, b: int, c: int, max_iterations: int = 1000000) -> list[int]:
    """
    MOD-BASED generalized Collatz sequence.
    Strictly stays within Natural Numbers (N).
    - If n ≡ 0 (mod a): n → n/a
    - If n ≢ 0 (mod a): n → b*n + c, then apply maximal division by a (shortcut)
    - Fallback: If (b*n + c) is not divisible by a, use standard Collatz logic (// 2)
    """
    if n <= 0 or a <= 0:
        return []
    
    sequence = [n]
    current = n
    
    for _ in range(max_iterations):
        # 1. Termination Check
        if current == 1:
            break
            
        # 2. Magnitude Safety Check (Prevent overflow in deep searches)
        if current > 10**60: 
            break 
            
        if current % a == 0:
            # DIVISION STEP: divide by parameter 'a'
            current = current // a
        else:
            # TRANSFORMATION STEP: apply b*n + c
            numerator = b * current + c
            
            if a == 2 and b == 3 and c == 1:
                # Standard Collatz (3n+1): No shortcut, just the result
                current = numerator
            else:
                # Generalized Collatz: Check divisibility by 'a'
                if numerator % a == 0:
                    # Apply the "Syracuse" shortcut only if divisible
                    while numerator % a == 0 and numerator > 0:
                        numerator = numerator // a
                    current = numerator
                else:
                    # RATIONAL FALLBACK: 
                    # If b*n+c isn't divisible by 'a', use // 2 to stay in N
                    # and ensure the sequence has a descent mechanism.
                    current = numerator // 2 if numerator % 2 == 0 else numerator + 1

        # 3. Cycle Detection
        if current in sequence:
            if current != sequence[-1]:
                sequence.append(current)
            break 
            
        sequence.append(current)
        
    return sequence


# --- Benford's Law Helper Functions ---
def get_leading_digit(n: int) -> int:
    """Extract the leading digit of a positive integer."""
    if n <= 0:
        return 0
    return int(str(abs(n))[0])


def benford_distribution():
    """Returns the Benford expected distribution for digits 1 through 9."""
    return {d: math.log10(1 + 1/d) for d in range(1, 10)}


def calculate_dmix(observed_counts: Dict[int, int]) -> float:
    """Calculates Dmix (total variation distance)."""
    total = sum(observed_counts.values())
    if total == 0:
        return float('nan')
    
    benford_dist = benford_distribution()
    dmix = 0.0
    for d in range(1, 10):
        p_obs = observed_counts.get(d, 0) / total
        p_benford = benford_dist[d]
        dmix += abs(p_obs - p_benford)
        
    return dmix / 2


def calculate_mad(observed_counts: Dict[int, int]) -> float:
    """Calculate Mean Absolute Deviation (MAD) from Benford's Law."""
    total = sum(observed_counts.values())
    if total == 0:
        return float('nan')
    
    benford_dist = benford_distribution()
    mad = 0.0
    for d in range(1, 10):
        p_obs = observed_counts.get(d, 0) / total
        p_benford = benford_dist[d]
        mad += abs(p_obs - p_benford)
    
    return mad / 9


def digital_mixing_speed(mad: float, sample_size: int) -> float:
    """Calculates the digital mixing speed metric."""
    if mad == 0 or sample_size <= 0:
        return float('nan')
    return (1.0 / mad) * math.log10(sample_size)


class SwissCheeseParameterScanner:
    def __init__(self, cube_center: Tuple[int, int, int], cube_side_length: int):
        self.center = cube_center
        self.side_length = cube_side_length
        self.holes: Set[Tuple[int, int, int]] = set()
        
        half = cube_side_length // 2
        self.a_range = range(max(2, cube_center[0] - half), cube_center[0] + half + 1)
        self.b_range = range(cube_center[1] - half, cube_center[1] + half + 1)
        self.c_range = range(cube_center[2] - half, cube_center[2] + half + 1)
    
    def add_trivial_patterns(self):
        trivial = []
        for a in self.a_range:
            for b in self.b_range:
                for c in self.c_range:
                    if a <= 1 or (b == 1 and c == 0) or (b < 0 and c < 0):
                        trivial.append((a, b, c))
        self.holes.update(trivial)
        return len(trivial)
    
    def analyze_single_parameter_set(self, a, b, c, initial_range, max_iterations):
        start, end = initial_range
        all_digits = []
        start_time = time.time()
        
        for n in range(start, end + 1):
            if n == 1: continue
            seq = generalized_collatz(n, a, b, c, max_iterations)
            for term in seq:
                if term > 0:
                    all_digits.append(get_leading_digit(term))
        
        counts = Counter(all_digits)
        observed = {d: counts.get(d, 0) for d in range(1, 10)}
        total = sum(observed.values())
        mad_val = calculate_mad(observed)
        
        return {
            "parameters": (a, b, c),
            "total_samples": total,
            "mad": mad_val,
            "dmix": calculate_dmix(observed),
            "mixing_speed": digital_mixing_speed(mad_val, total) if total > 0 else 0,
            "time": time.time() - start_time
        }

    def scan_cube(self, initial_range=(1, 1000), max_iterations=100000):
        results = []
        points = list(itertools.product(self.a_range, self.b_range, self.c_range))
        
        print(f"Starting Scan: {len(points) - len(self.holes)} active points.")
        for a, b, c in points:
            if (a, b, c) in self.holes: continue
            res = self.analyze_single_parameter_set(a, b, c, initial_range, max_iterations)
            results.append(res)
            print(f"  (a={a}, b={b}, c={c}) | MAD: {res['mad']:.6f}")
        return results

def print_summary(results):
    sorted_res = sorted([r for r in results if not math.isnan(r['mad'])], key=lambda x: x['mad'])
    print("\n" + "="*60)
    print(f"{'Rank':<6}{'Params':<15}{'MAD':<12}{'Mix Speed':<15}")
    print("-" * 60)
    for i, r in enumerate(sorted_res[:10], 1):
        print(f"{i:<6}{str(r['parameters']):<15}{r['mad']:<12.6f}{r['mixing_speed']:<15.2f}")

if __name__ == "__main__":
    # Test on the elite set vicinity
    scanner = SwissCheeseParameterScanner(cube_center=(21, 14, 12), cube_side_length=3)
    scanner.add_trivial_patterns()
    results = scanner.scan_cube(initial_range=(1, 500))
    print_summary(results)
