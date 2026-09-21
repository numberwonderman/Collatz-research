from convergence_scan import analyze_point, is_degenerate, parse_range


def test_is_degenerate_ridge_family():
    reason = is_degenerate({"a": 7, "b": 14, "c": 7})
    assert reason is not None
    assert "2n+1" in reason


def test_is_degenerate_b_multiple_of_a():
    assert is_degenerate({"a": 6, "b": 12, "c": 5}) is not None


def test_is_degenerate_c_multiple_of_a():
    assert is_degenerate({"a": 4, "b": 5, "c": 8}) is not None


def test_is_degenerate_generic_point():
    assert is_degenerate({"a": 2, "b": 3, "c": 1}) is None


def test_analyze_point_classic_collatz_converges():
    result = analyze_point(2, 3, 1, (2, 200), max_iterations=1000)
    assert result["convergence_rate"] == 1.0
    assert result["sampled_n"] == 199
    assert result["mad"] is not None


def test_analyze_point_immediate_neighbor_diverges():
    # (2, 4, 1) is a distance-1 neighbor of classic Collatz that was found
    # to collapse to near-zero convergence.
    result = analyze_point(2, 4, 1, (2, 200), max_iterations=1000)
    assert result["convergence_rate"] < 0.05


def test_parse_range():
    assert list(parse_range("3:7")) == [3, 4, 5, 6, 7]
