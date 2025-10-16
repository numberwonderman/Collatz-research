import sys
import json
import time
import math
from collections import Counter
from typing import Dict

# --- Imports (Assuming these files are correctly updated/exist in 'src/') ---
from src.collatz_generator import generalized_collatz
from src.benford_analyzer import get_leading_digit
from src.statistical_tests import analyze_conformity 

# --- Configuration ---
TEST_CONFIG = {
    "param_a": 12,
    "param_b": 16,
    "param_c": -4,
    "initial_range": (1, 10000),
    "max_iterations": 1000000
}
# python -m src.pipeline >> raw_test_data.txt

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
    
    print("\n--- Running Statistical Tests ---")
    results = analyze_conformity(observed_counts)
    
    total_samples = sum(observed_counts.values())
    
    mad_value = results.get('mad', None)
    if mad_value is not None:
        results['digital_mixing_speed'] = digital_mixing_speed(mad_value, total_samples)
    
    results['dmix_variance'] = calculate_dmix(observed_counts)
    
    final_output = {
        "parameter_set": f"a={a}, b={b}, c={c}",
        "initial_range": config["initial_range"],
        "digit_frequencies": observed_counts,
        "total_samples": total_samples,
        **results
    }
    
    return final_output

# --- Main Execution Block for Console Testing ---
if __name__ == "__main__":
    try:
        # Run the Collatz analysis with the current TEST_CONFIG (Classical Collatz)
        analysis_output = run_analysis_pipeline(TEST_CONFIG)
        
        # Print a clean, readable output for console verification
        print("\n--- ANALYSIS RESULTS (CONCATENATED SAMPLES) ---")
        print(f"Parameters: {analysis_output['parameter_set']}")
        range_start, range_end = analysis_output['initial_range']
        print(f"Initial Range: N={range_start} to {range_end}")
        print(f"Total Samples: {analysis_output['total_samples']:,}")  # Comma formatting
        
        formatted_digits = json.dumps(analysis_output['digit_frequencies'], indent=2)
        print(f"Digit Counts: {formatted_digits}")
        
        # Print key statistical metrics with updated names
        print(f"Chi-Squared p-value: {analysis_output.get('p_value', 'N/A')}")
        print(f"MAD Score: {analysis_output.get('mad', 'N/A')}")
        print(f"KS D-Max: {analysis_output.get('ks_d_max', 'N/A')}")
        print(f"Digital Mixing Speed (1/MAD * log10(m)): {analysis_output.get('digital_mixing_speed', 'N/A')}")
        print(f"Dmix Variance (Total Variation Distance): {analysis_output.get('dmix_variance', 'N/A')}")
        
    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}", file=sys.stderr)
