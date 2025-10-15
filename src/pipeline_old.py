# src/pipeline.py

import sys
import json
from typing import Dict
from collections import Counter

# --- NEW IMPORT: Add the visualization function ---
from src.visualization import plot_benford_histogram 

# --- Imports (Corrected to use defined function names) ---
from src.collatz_generator import generalized_collatz
from src.benford_analyzer import get_leading_digit, analyze_leading_digits
from src.statistical_tests import get_expected_counts, run_chi_squared_test, calculate_mad, analyze_conformity 

# --- Configuration (CRITICAL FIX: Consistency and Scope) ---

# Classical Collatz Test Configuration (a=2, b=3, c=1)
TEST_CONFIG = {
    "param_a": 2,
    "param_b": 3,
    "param_c": 1,
    # CRITICAL CORRECTION: Set initial range to 10,000 as stated in the paper
    "initial_range": (1, 10000), 
    # CRITICAL CORRECTION: Set max iterations to 100,000 for consistency
    "max_iterations": 100000 
}

# --- Core Pipeline Function ---

def run_analysis_pipeline(config: Dict) -> Dict:
    """
    Runs the Collatz generation, digit extraction, statistical analysis, AND visualization.
    """
    a = config["param_a"]
    b = config["param_b"]
    c = config["param_c"]
    start, end = config["initial_range"]
    max_iter = config["max_iterations"]
    
    all_digits = []
    
    # 1. Sequence Generation (P1)
    print(f"--- Running Generalized Collatz (a={a}, b={b}, c={c}) on N={start} to {end} ---")
    
    # Run the Collatz process for the specified range of starting numbers
    for n in range(start, end + 1):
        # We start analysis from 2 since 1 is the convergence point
        if n == 1:
            continue

        sequence = generalized_collatz(n, a, b, c, max_iter)
        
        # 2. Leading Digit Extraction & Aggregation (P2)
        for term in sequence:
            try:
                # We skip the number 1 (the final convergence point) from the analysis.
                if term > 1:
                    digit = get_leading_digit(term)
                    all_digits.append(digit)
            except ValueError:
                # Skip non-positive values
                continue
                
    # 3. Digit Frequency Counting (P2/P3 boundary)
    final_counts = Counter(all_digits)
    observed_counts = {d: final_counts.get(d, 0) for d in range(1, 10)}
    
    # 4. Statistical Analysis (P3)
    print("\n--- Running Statistical Tests ---")
    results = analyze_conformity(observed_counts)
    
    # 5. NEW STEP: Visualization
    # Calculate observed frequencies (proportions) for the plot
    total_samples = sum(observed_counts.values())
    observed_frequencies = [observed_counts[d] / total_samples for d in range(1, 10)]
    
    # Generate the plot
    plot_title = f"First-Digit Distribution for C_{a},{b},{c}"
    # NOTE: The plotting function is expected to save the figure to the 'results' directory
    plot_benford_histogram(observed_frequencies, results['mad'], plot_title)

    # Combine everything for the final output structure
    final_output = {
        "parameter_set": f"a={a}, b={b}, c={c}",
        "initial_range": config["initial_range"],
        "digit_frequencies": observed_counts,
        **results
    }
    
    return final_output

# --- Main Execution Block for Console Testing ---
if __name__ == "__main__":
    try:
        # Run the standard Collatz analysis (This should conform to Benford's Law)
        analysis_output = run_analysis_pipeline(TEST_CONFIG)
        
        # Print a clean, readable output for console verification
        print("\n--- ANALYSIS RESULTS (CONCATENATED SAMPLES) ---")
        print(f"Parameters: {analysis_output['parameter_set']}")
        print(f"Total Samples: {analysis_output['total_samples']}")
        
        formatted_digits = json.dumps(analysis_output['digit_frequencies'], indent=2)
        print(f"Digit Counts: {formatted_digits}")
        print(f"Chi-Squared p-value: {analysis_output['p_value']}")
        
        # CRITICAL FIX: Correct the misleading comment on the MAD threshold!
        print(f"MAD Score: {analysis_output['mad']}")
        print(f"Conformity Level (Rigorous): {analysis_output['conformity_label']}")
        
    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}", file=sys.stderr)