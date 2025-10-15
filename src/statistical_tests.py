import math
from typing import Dict, List
from scipy.stats import chisquare
import numpy as np
from collections import Counter

# --- Benford's Law Expected Distribution ---
BENFORD_EXPECTED_PROBABILITIES: Dict[int, float] = {
    d: math.log10(1 + 1/d) for d in range(1, 10)
}

# --- Utility Functions ---

def get_expected_counts(total_samples: int) -> List[float]:
    """
    Calculates the expected count for each digit (1-9) based on Benford's Law.
    """
    expected_counts = [
        total_samples * BENFORD_EXPECTED_PROBABILITIES[d]
        for d in range(1, 10)
    ]
    return expected_counts

def run_chi_squared_test(observed_counts: Dict[int, int], total_samples: int) -> Dict[str, float]:
    """
    Performs the Chi-squared goodness of fit test against the Benford distribution.
    Returns the statistic and the p-value.
    """
    observed = [observed_counts[d] for d in range(1, 10)]
    expected = get_expected_counts(total_samples)
    
    # chisquare returns the statistic and the p-value
    # ddof=0 for a test against a completely specified distribution (Benford's Law)
    try:
        chi_squared_stat, p_value = chisquare(f_obs=observed, f_exp=expected, ddof=0) 
    except ValueError:
        # Handles cases where expected counts are too low (unlikely with large samples)
        return {
            "chi_squared_statistic": float('nan'),
            "p_value": float('nan'),
        }

    return {
        "chi_squared_statistic": round(chi_squared_stat, 4),
        "p_value": round(p_value, 5), # Increased precision for reporting 0.0
        "degrees_of_freedom": 8
    }

def calculate_mad(observed_counts: Dict[int, int], total_samples: int) -> Dict[str, float]:
    """
    Calculates the Mean Absolute Deviation (MAD).
    """
    observed_probs = np.array([observed_counts.get(d, 0) / total_samples for d in range(1, 10)])
    expected_probs = np.array([BENFORD_EXPECTED_PROBABILITIES[d] for d in range(1, 10)])
    
    # Mean Absolute Deviation (MAD) = sum(|Observed_Freq - Expected_Freq|) / 9
    absolute_deviations = np.abs(observed_probs - expected_probs)
    mad_value = np.sum(absolute_deviations) / 9
    
    return {
        "mad": round(mad_value, 5) # Essential for manual conformity check (0.0078)
    }

def run_kolmogorov_smirnov_statistic(observed_counts: Dict[int, int], total_samples: int) -> Dict[str, float]:
    """
    Calculates the Kolmogorov-Smirnov D-statistic (max absolute difference between CDFs).
    We skip the complex p-value for the discrete case and only report the D-max stat.
    """
    observed_freq = np.array([observed_counts.get(d, 0) / total_samples for d in range(1, 10)])
    expected_freq = np.array([BENFORD_EXPECTED_PROBABILITIES[d] for d in range(1, 10)])
    
    observed_cdf = np.cumsum(observed_freq)
    expected_cdf = np.cumsum(expected_freq)
    
    ks_statistic = np.max(np.abs(observed_cdf - expected_cdf))
    
    return {
        "ks_d_max": round(ks_statistic, 5) # The D-statistic for KS test
    }


# --- Primary Analysis Function (Cleaned) ---

def analyze_conformity(observed_counts: Dict[int, int]) -> Dict:
    """
    Runs all required statistical tests (Chi-Squared, MAD, KS) and aggregates the results.
    DOES NOT perform any automatic conformity labeling or use fixed thresholds.
    """
    total_samples = sum(observed_counts.values())
    
    if total_samples == 0:
        return {
            "total_samples": 0,
            "p_value": 1.0, 
            "mad": 1.0, 
        }

    # Run all metrics
    chi_results = run_chi_squared_test(observed_counts, total_samples)
    ks_results = run_kolmogorov_smirnov_statistic(observed_counts, total_samples)
    mad_results = calculate_mad(observed_counts, total_samples)
    
    results = {
        "total_samples": total_samples,
        **chi_results,
        **ks_results,
        **mad_results,
        # IMPORTANT: No 'conforms_benford' or 'conformity_label' is returned.
    }
    
    return results
