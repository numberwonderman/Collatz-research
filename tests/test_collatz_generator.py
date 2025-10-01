import pytest
from src.collatz_generator import generalized_collatz

def test_standard_collatz_sequence_at_six():
    """
    Test for the standard Collatz sequence (3n+1) starting at 6.
    Parameters: a=2, b=3, c=1.
    """
    # The expected sequence from your saved information: [6, 3, 10, 5, 16, 8, 4, 2, 1]
    expected_sequence = [6, 3, 10, 5, 16, 8, 4, 2, 1]
    
    # Run the function for a standard Collatz (a=2, b=3, c=1)
    result_sequence = generalized_collatz(n=6, a=2, b=3, c=1)
    
    # Assert that the generated sequence matches the expected sequence
    assert result_sequence == expected_sequence, \
        f"Test Failed: Standard Collatz sequence is incorrect.\n" \
        f"Expected: {expected_sequence}\n" \
        f"Received: {result_sequence}"

def test_sequence_stopping_at_one():
    """Test a sequence that converges to 1."""
    expected_sequence = [19, 58, 29, 88, 44, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
    result_sequence = generalized_collatz(n=19, a=2, b=3, c=1)
    assert result_sequence == expected_sequence

def test_sequence_with_max_iterations():
    """Test the max_iterations safeguard."""
    # This sequence is much longer, the limit should cut it off before 1 is reached.
    result_sequence = generalized_collatz(n=27, a=2, b=3, c=1, max_iterations=5)
    expected_sequence = [27, 82, 41, 124, 62, 31] # The last step is the 5th iteration result (31)
    assert len(result_sequence) == 6
    assert result_sequence[0] == 27
    assert result_sequence[-1] == 31


# Add more tests here for other variants and edge cases as per your plan!
