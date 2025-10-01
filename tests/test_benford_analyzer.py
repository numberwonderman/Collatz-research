import pytest
from src.benford_analyzer import get_leading_digit, analyze_leading_digits

def test_get_leading_digit_single_digit():
    """Test standard single-digit numbers."""
    assert get_leading_digit(5) == 5
    assert get_leading_digit(9) == 9

def test_get_leading_digit_powers_of_ten():
    """Test numbers that are powers of 10."""
    assert get_leading_digit(10) == 1
    assert get_leading_digit(100) == 1
    assert get_leading_digit(1000) == 1

def test_get_leading_digit_various_numbers():
    """Test numbers starting with different digits across magnitudes."""
    assert get_leading_digit(1234) == 1
    assert get_leading_digit(200) == 2
    assert get_leading_digit(314159) == 3
    assert get_leading_digit(999999) == 9
    assert get_leading_digit(45678) == 4
    assert get_leading_digit(87654) == 8

def test_get_leading_digit_non_positive_raises_error():
    """Test that non-positive numbers are correctly handled."""
    with pytest.raises(ValueError):
        get_leading_digit(0)
    with pytest.raises(ValueError):
        get_leading_digit(-10)

def test_analyze_leading_digits_simple_sequence():
    """Test a small, predictable sequence."""
    sequence = [1, 2, 3, 10, 25, 400, 55, 6, 7, 8, 9]
    expected_counts = {
        1: 2,  # 1, 10
      2: 2,  # 2, 25q
        3: 1,  # 3
        4: 1,  # 400
        5: 1,  # 25, 55
        6: 1,  # 6
        7: 1,  # 7
        8: 1,  # 8
        9: 1   # 9
    }
    result = analyze_leading_digits(sequence)
    assert result == expected_counts

def test_analyze_leading_digits_empty_sequence():
    """Test an empty input list."""
    expected_counts = {
        1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0
    }
    result = analyze_leading_digits([])
    assert result == expected_counts
