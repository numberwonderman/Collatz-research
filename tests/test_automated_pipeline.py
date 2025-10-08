import pytest
import tempfile
import json
import os
from unittest import mock
from pathlib import Path

from automated_pipeline import (
    load_processed_cases,
    save_processed_cases,
    process_batch,
    RuntimeConfig,
)

# Mock Collatz functions
def mock_generalized_collatz(n, a, b, c, max_iter):
    # Simulated sequence (simple for testing)
    return [n, n*2, 1]

def mock_get_leading_digit(n):
    # Leading digit extraction mock
    return int(str(n)[0])

@mock.patch("automated_pipeline.generalized_collatz", side_effect=mock_generalized_collatz)
@mock.patch("automated_pipeline.get_leading_digit", side_effect=mock_get_leading_digit)
def test_processed_cases_skip_and_update(mock_collatz, mock_digit):
    with tempfile.TemporaryDirectory() as tmpdir:
        processed_file = Path(tmpdir) / "processed_cases.json"
        # Initially mark '2' as processed
        save_processed_cases(processed_file, {2})

        config = RuntimeConfig(
            param_a=2,
            param_b=3,
            param_c=1,
            initial_range_start=1,
            initial_range_end=5,
            max_iterations=10,
            batch_size=5,
            output_dir=tmpdir,
            save_intermediate=False,
        )

        batch_info = (1, 5, config)
        processed_cases = load_processed_cases(processed_file)

        result = process_batch(batch_info, processed_file, processed_cases)

        # '1' is trivial and skipped, '2' is already processed and skipped
        expected_processed_numbers = {2, 3, 4, 5}
        assert processed_cases == expected_processed_numbers or expected_processed_numbers.issubset(processed_cases)
        # Check batch result keys and values
        assert "batch_range" in result
        assert "digit_counts" in result
        assert result["processed_count"] == 4  # skipped 1 only

        # Check the processed file was updated on disk
        with open(processed_file, "r") as f:
            disk_processed = set(json.load(f))
        assert expected_processed_numbers == disk_processed
