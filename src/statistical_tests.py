import math
from typing import Dict, List
from scipy.stats import chisquare, kstest
import numpy as np

# --- Benford's Law Expected Distribution ---
BENFORD_EXPECTED_PROBABILITIES: Dict[int, float] = {
    d: math.log10(1 + 1/d) for d in range(1, 10)
}

# Values for easy reference (from your spec table)
# {1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079, 
#  6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046}


def get_expected_counts(total_samples: int) -> List[float]:
    """
    Calculates the expected count for each digit (1-9) based on Benford's Law.

    Args:
        total_samples: The total number of leading digits observed.

    Returns:
        A list of 9 expected counts, indexed 0-8 for digits 1-9.
    """
    expected_counts = [
        total_samples * BENFORD_EXPECTED_PROBABILITIES[d]
        for d in range(1, 10)
    ]
    return expected_counts

def run_chi_squared_test(observed_counts: Dict[int, int], total_samples: int, alpha: float = 0.05) -> Dict[str, float]:
    """
    Performs the Chi-squared goodness of fit test against the Benford distribution.

    Args:
        observed_counts: Dictionary of observed digit counts (1-9).
        total_samples: Total number of observations.
        alpha: The significance level (default 0.05).

    Returns:
        A dictionary with chi_squared statistic, p_value, and boolean result.
    """
    observed = [observed_counts[d] for d in range(1, 10)]
    expected = get_expected_counts(total_samples)

    # Chi-squared requires expected counts to be > 5 for reliable results.
    # While Collatz sequences are typically large, this is a standard caveat.
    
    # We use ddof=8 because we are testing against a specific, fixed distribution
    # (Benford's Law) and not estimating any parameters from the data.
    # Degrees of Freedom (df) = k - 1, where k=9 categories. df = 8.
    
    # chisquare returns the statistic and the p-value
    chi_squared_stat, p_value = chisquare(f_obs=observed, f_exp=expected, ddof=0) 
    
    # A high p-value (p > alpha) means we DO NOT reject the null hypothesis (conformity).
    conforms = p_value > alpha
    
    return {
        "chi_squared_statistic": round(chi_squared_stat, 4),
        "p_value": round(p_value, 4),
        "conforms_chi_squared": conforms,
        "degrees_of_freedom": 8
    }

def run_kolmogorov_smirnov_test(observed_counts: Dict[int, int], total_samples: int) -> Dict[str, float]:
    """
    Performs the Kolmogorov-Smirnov test on the Cumulative Distribution Function (CDF).
    
    Args:
        observed_counts: Dictionary of observed digit counts (1-9).
        total_samples: Total number of observations.

    Returns:
        A dictionary with the KS statistic and the p-value.
    """
    # 1. Prepare Observed Frequencies
    observed_freq = np.array([observed_counts[d] / total_samples for d in range(1, 10)])
    
    # 2. Prepare Expected Frequencies (Benford's Law)
    expected_freq = np.array([BENFORD_EXPECTED_PROBABILITIES[d] for d in range(1, 10)])
    
    # 3. Calculate CDFs
    observed_cdf = np.cumsum(observed_freq)
    expected_cdf = np.cumsum(expected_freq)
    
    # 4. The KS test in SciPy typically requires raw data or a known CDF function.
    # Since we have the cumulative counts, we use a manual approach or simpler test 
    # to find the D statistic (max absolute difference between CDFs).
    ks_statistic = np.max(np.abs(observed_cdf - expected_cdf))
    
    # NOTE: The p-value calculation for D from discrete data is complex. 
    # For a simplified tool, many only report the D-statistic and compare to a critical 
    # value table, or use a simplified approximation. We will report the D-statistic (D_max).
    
    # For a full implementation, you would need to use a package like 'statsmodels'
    # or implement the p-value calculation for the discrete KS test.
    # For this specification, we will proceed with the primary statistic (D-max).
    
    return {
        "ks_statistic": round(ks_statistic, 4),
        "ks_d_max": round(ks_statistic, 4) # Renamed for clarity in spec output
    }

def calculate_mad(observed_counts: Dict[int, int], total_samples: int) -> Dict[str, float]:
    """
    Calculates the Mean Absolute Deviation (MAD) and checks for conformity.
    
    MAD < 0.015 = conformity (based on a common Benford's Law standard).

    Args:
        observed_counts: Dictionary of observed digit counts (1-9).
        total_samples: Total number of observations.

    Returns:
        A dictionary with the MAD value and a boolean conformity check.
    """
    observed_probs = np.array([observed_counts[d] / total_samples for d in range(1, 10)])
    expected_probs = np.array([BENFORD_EXPECTED_PROBABILITIES[d] for d in range(1, 10)])
    
    # Mean Absolute Deviation (MAD) = sum(|Observed_Freq - Expected_Freq|) / 9
    absolute_deviations = np.abs(observed_probs - expected_probs)
    mad_value = np.sum(absolute_deviations) / 9
    
    # Conformity check based on your spec: MAD < 0.015
    CONFORMITY_THRESHOLD = 0.015
    conforms = mad_value < CONFORMITY_THRESHOLD
    
    return {
        "mad": round(mad_value, 5),
        "conforms_mad": conforms
    }


def analyze_conformity(observed_counts: Dict[int, int]) -> Dict:
    """
    Runs all statistical tests and aggregates the results.

    Args:
        observed_counts: Dictionary of observed leading digit counts (1-9).

    Returns:
        A dictionary containing all statistical results.
    """
    total_samples = sum(observed_counts.values())
    
    if total_samples == 0:
        return {
            "total_samples": 0,
            "chi_squared_statistic": 0.0,
            "p_value": 1.0, # Neutral p-value
            "conforms_chi_squared": False,
            "ks_statistic": 0.0,
            "mad": 1.0, # High MAD if no data
            "conforms_mad": False,
            "conforms_benford": False
        }

    chi_results = run_chi_squared_test(observed_counts, total_samples)
    ks_results = run_kolmogorov_smirnov_test(observed_counts, total_samples)
    mad_results = calculate_mad(observed_counts, total_samples)
    
    # Determine overall conformity (e.g., must pass MAD and Chi-squared)
    # The spec's output structure only requires a single "conforms_benford" bool.
    # We will define conformity as passing the Chi-squared test OR the MAD test
    # (The MAD test is generally considered the simpler, standard check in Benford's literature).
    
    # Sticking strictly to the spec's requirements for MAD:
    overall_conformity = mad_results["conforms_mad"]
    
    results = {
        "total_samples": total_samples,
        **chi_results,
        **ks_results,
        **mad_results,
        "conforms_benford": overall_conformity
    }
    
    return results
