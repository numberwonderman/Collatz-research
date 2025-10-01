# src/pipeline.py

import sys
import json
from typing import Dict
from collections import Counter

# --- Imports (Corrected to use defined function names) ---
from src.collatz_generator import generalized_collatz
from src.benford_analyzer import get_leading_digit, analyze_leading_digits
from src.statistical_tests import get_expected_counts, run_chi_squared_test, calculate_mad, analyze_conformity 

# --- Configuration ---

# Classical Collatz Test Configuration (a=2, b=3, c=1)
TEST_CONFIG = {
    "param_a": 2,
    "param_b": 3,
    "param_c": 1,
    "initial_range": (1, 10000),   # Test range of 1 to 100
    "max_iterations": 2000
}

# --- Core Pipeline Function ---

def run_analysis_pipeline(config: Dict) -> Dict:
    """
    Runs the Collatz generation, digit extraction, and statistical analysis.
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
        # CRITICAL FIX: Extract the digit and append the digit itself.
        for term in sequence:
            try:
                # We skip the number 1 (the final convergence point) from the analysis.
                if term > 1:
                    digit = get_leading_digit(term)
                    # We append the digit (an integer 1-9) to the master list
                    all_digits.append(digit)
            except ValueError:
                # Skip non-positive values
                continue
                
    # 3. Digit Frequency Counting (P2/P3 boundary)
    # The 'all_digits' list now correctly contains only the leading digits (1-9).
    final_counts = Counter(all_digits)
    
    # Pad the counts dictionary to ensure keys 1-9 exist for the statistical functions
    observed_counts = {d: final_counts.get(d, 0) for d in range(1, 10)}
    
    # 4. Statistical Analysis (P3)
    print("\n--- Running Statistical Tests ---")
    results = analyze_conformity(observed_counts)
    
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
        # Ensure the dictionary is printed nicely
        formatted_digits = json.dumps(analysis_output['digit_frequencies'], indent=2)
        print(f"Digit Counts: {formatted_digits}")
        print(f"Chi-Squared p-value: {analysis_output['p_value']}")
        print(f"MAD Score: {analysis_output['mad']} (Conforms if < 0.015)")
        print(f"Overall Conformity: {analysis_output['conforms_benford']}")
        
    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}", file=sys.stderr)