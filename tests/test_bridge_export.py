from bridge_export import SCHEMA_VERSION, build_bridge_payload


def test_build_bridge_payload_shape():
    fake_results = [
        {
            "parameters": (2, 3, 1),
            "total_samples": 100,
            "mad": 0.01,
            "chi_squared_statistic": 5.0,
            "p_value": 0.5,
            "ks_d_max": 0.02,
            "dmix_variance": 0.015,
            "digital_mixing_speed": 200.0,
        },
        {
            "parameters": (2, 3, 3),
            "total_samples": 80,
            "mad": 0.02,
            "chi_squared_statistic": 6.0,
            "p_value": 0.4,
            "ks_d_max": 0.03,
            "dmix_variance": 0.025,
            "digital_mixing_speed": 150.0,
        },
    ]

    payload = build_bridge_payload(
        fake_results,
        cube_center=(2, 3, 2),
        side_length=3,
        initial_range=(1, 1000),
        max_iterations=2000,
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["source"] == "collatz-research"
    assert payload["cube_center"] == [2, 3, 2]
    assert payload["side_length"] == 3
    assert payload["initial_range"] == [1, 1000]
    assert payload["max_iterations"] == 2000
    assert "generated_at" in payload
    assert len(payload["points"]) == 2

    point = payload["points"][0]
    assert point["a"] == 2
    assert point["b"] == 3
    assert point["c"] == 1
    assert point["mad"] == 0.01
    assert point["digital_mixing_speed"] == 200.0


def test_build_bridge_payload_empty_results():
    payload = build_bridge_payload(
        [],
        cube_center=(2, 3, 1),
        side_length=1,
        initial_range=(1, 100),
        max_iterations=1000,
    )

    assert payload["points"] == []
