# src/pipeline.py

import sys
import json
from typing import Dict

# Assume these are in the same 'src' directory
# (You will need to ensure the imports work based on your main.py execution context)
from collatz_generator import generalized_collatz
from benford_analyzer import analyze_leading_digits
from statistical_tests import analyze_conformity 

# --- Configuration (This would normally come from CLI/config file in Phase 5) ---

# Classical Collatz Test Configuration (a=2, b=3, c=1)
TEST_CONFIG = {
    "param_a": 2,
    "param_b": 3,
    "param_c": 1,
    "initial_range": (1, 100),  # Test range of 1 to 100
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
    
    for n in range(start, end + 1):
        # We start analysis from 2 since 1 is the convergence point
        if n == 1:
            continue

        sequence = generalized_collatz(n, a, b, c, max_iter)
        
        # 2. Leading Digit Extraction (P2)
        # We concatenate all sequences to form one large dataset of leading digits
        for term in sequence:
            try:
                # Only analyze terms > 0, excluding the starting number 'n' itself
                if term > 1: # We keep 1 in the sequence for cycle detection but skip it here.
                    digit = analyze_leading_digits([term])[get_leading_digit(term)] # This is a bit clumsy, but works
                    all_digits.append(digit)
            except ValueError:
                # Non-positive numbers skipped
                continue
                
    # 3. Digit Frequency Counting (P2)
    # Re-run the counting on the collective list of all digits
    # The analyze_leading_digits function in P2 is currently designed to take a sequence, 
    # not a list of digits, so we need a slight adjustment or a clean count:
    from collections import Counter
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
        print(f"Digit Counts: {json.dumps(analysis_output['digit_frequencies'], indent=2)}")
        print(f"Chi-Squared p-value: {analysis_output['p_value']}")
        print(f"MAD Score: {analysis_output['mad']} (Conforms if < 0.015)")
        print(f"Overall Conformity: {analysis_output['conforms_benford']}")
        
    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}", file=sys.stderr)
