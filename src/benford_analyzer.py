import math
from collections import Counter
from typing import List, Dict

def get_leading_digit(n: int) -> int:
    """
    Extracts the first significant digit (1-9) of a positive integer.
    Handles large numbers by using base-10 logarithm properties.
    
    Args:
        n: The number from the Collatz sequence.

    Returns:
        The first digit (1-9).
    """
    if n <= 0:
        # Collatz sequences are typically defined for positive integers.
        # This handles a potential edge case if the sequence hits zero or negative.
        raise ValueError("Input number must be a positive integer.")
    
    # Use the mantissa/exponent property: 
    # The leading digit is floor(n / 10^k) where k is floor(log10(n)).
    # A cleaner mathematical way: 10^(log10(n) - floor(log10(n)))
    
    # log10(n) gives the power of 10. floor(log10(n)) gives the exponent 'k'.
    exponent = math.floor(math.log10(n))
    
    # Divides n by 10^exponent to get the number between 1 and 10.
    leading_float = n / (10**exponent)
    
    # The integer part is the leading digit (1-9).
    return math.floor(leading_float)

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
