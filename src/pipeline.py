import math
from collections import Counter
from typing import Dict
import time

# Assuming these imports remain unchanged
from src.collatz_generator import generalized_collatz
from src.benford_analyzer import get_leading_digit
from src.statistical_tests import analyze_conformity 

def benford_distribution():
    """Returns the Benford expected distribution for digits 1 through 9."""
    return {d: math.log10(1 + 1/d) for d in range(1, 10)}

def calculate_dmix(observed_counts: Dict[int, int]) -> float:
    """
    Calculates Dmix (total variation distance) between observed digit distribution and Benford distribution.
    This is a statistical measure of distributional divergence.
    """
    total = sum(observed_counts.values())
    if total == 0:
        return float('nan')  # No samples to compare
    
    benford_dist = benford_distribution()
    dmix = 0.0
    
    for d in range(1, 10):
        p_obs = observed_counts.get(d, 0) / total
        p_benford = benford_dist[d]
        dmix += abs(p_obs - p_benford)
        
    dmix /= 2  # total variation distance
    return dmix

def digital_mixing_speed(mad: float, sample_size: int) -> float:
    """
    Calculates the digital mixing speed metric as (1/MAD) * log10(sample_size).
    Reflects conformity quality weighted by sample size.
    """
    if mad == 0 or sample_size <= 0:
        return float('inf')  # Avoid division by zero, infinite conformity
    return (1.0 / mad) * math.log10(sample_size)

def run_analysis_pipeline(config: Dict) -> Dict:
    a = config["param_a"]
    b = config["param_b"]
    c = config["param_c"]
    start, end = config["initial_range"]
    max_iter = config["max_iterations"]
    
    all_digits = []
    
    print(f"--- Running Generalized Collatz (a={a}, b={b}, c={c}) on N={start} to {end} ---")
    start_time = time.time()
    
    for n in range(start, end + 1):
        if n == 1:
            continue
        sequence = generalized_collatz(n, a, b, c, max_iter)
        for term in sequence:
            try:
                if term > 1:
                    digit = get_leading_digit(term)
                    all_digits.append(digit)
            except ValueError:
                continue
    end_time = time.time()
    print(f"Collatz Generation Time: {end_time - start_time:.2f} seconds")
    
    final_counts = Counter(all_digits)
    observed_counts = {d: final_counts.get(d, 0) for d in range(1, 10)}
    
    print("
--- Running Statistical Tests ---")
    results = analyze_conformity(observed_counts)
    
    total_samples = sum(observed_counts.values())
    
    # Calculate Digital Mixing Speed (custom metric)
    mad_value = results.get('mad', None)
    if mad_value is not None:
        results['digital_mixing_speed'] = digital_mixing_speed(mad_value, total_samples)
    
    # Calculate Dmix (total variation distance)
    results['dmix_variance'] = calculate_dmix(observed_counts)
    
    final_output = {
        "parameter_set": f"a={a}, b={b}, c={c}",
        "initial_range": config["initial_range"],
        "digit_frequencies": observed_counts,
        "total_samples": total_samples,
        **results
    }
    
    return final_output
