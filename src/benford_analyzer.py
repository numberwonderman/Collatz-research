import math
from collections import Counter
from typing import List, Dict

def get_leading_digit(n: int) -> int:
    """
    Extracts the first significant digit (1-9) of a positive integer.
    Uses string conversion for robust handling of large integers, avoiding 
    floating-point precision issues common in Collatz sequences.
    
    Args:
        n: The number from the Collatz sequence.

    Returns:
        The first digit (1-9).
    """
    if n <= 0:
        raise ValueError("Input number must be a positive integer.")
    
    # --- The robust fix: Convert to string, take the first character, convert to int ---
    leading_digit_str = str(n)[0] 
    return int(leading_digit_str)

def analyze_leading_digits(sequence: List[int]) -> Dict[int, int]:
    """
    Processes a Collatz sequence to count the frequency of leading digits (1-9).

    Args:
        sequence: A list of integers (the Collatz orbit).

    Returns:
        A dictionary mapping the digit (1-9) to its observed count.
    """
    digit_counts = Counter()
    
    # Only analyze numbers >= 1 (as Collatz is defined on positive integers)
    for number in sequence:
        try:
            # We skip the number '1' itself when testing for Benford's Law 
            # in sequences, as its leading digit is always 1, which can skew results.
            # However, the spec doesn't explicitly mention skipping, so we include it
            # and let the sample size smooth it out, unless you want to add an explicit skip.
            if number > 0:
                 digit = get_leading_digit(number)
                 # Ensure we only count digits 1 through 9
                 if 1 <= digit <= 9:
                    digit_counts[digit] += 1
        except ValueError:
            # Skip non-positive values (already handled in get_leading_digit)
            continue
            
    # Fill in counts for any missing digits (0 for digits 1-9 not found)
    for d in range(1, 10):
        if d not in digit_counts:
            digit_counts[d] = 0

    return dict(digit_counts)
