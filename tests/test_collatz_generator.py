import pytest
# Assuming your function is imported from the correct source file
from src.collatz_generator import generalized_collatz 

def test_standard_collatz_sequence():
    """
    Tests the classic (3n+1) sequence starting at 6.
    Parameters used: a=2 (ignored by new function), b=3, c=1.
    """
    # The expected sequence for N=6: 6, 3, 10, 5, 16, 8, 4, 2, 1
    expected = [6, 3, 10, 5, 16, 8, 4, 2, 1]
    
    # NOTE: The 'a' parameter (2) is still passed, but its value is ignored 
    # by the parity-based logic in the fixed function.
    result = generalized_collatz(6, a=2, b=3, c=1)
    
    assert result == expected, "Standard Collatz sequence failed for N=6"


def test_generalized_collatz_termination():
    """
    Tests the generalized (5n+1)/2^k sequence, which should terminate at 1.
    Parameters used: a=5 (ignored), b=5, c=1.
    
    Trace for N=3:
    3 (odd) -> (5*3 + 1) = 16. Max division by 2 is 4 steps: 16 -> 1.
    """
    # The expected sequence for N=3: 3, 1
    # Assuming your function is designed to go straight to 1 after the maximal division by 2.
    expected = [3, 1] 
    result = generalized_collatz(3, a=5, b=5, c=1)
    
    assert result == expected, "Generalized Collatz (5n+1) sequence failed for N=3"