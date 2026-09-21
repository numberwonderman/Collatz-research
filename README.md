# Collatz-research
software to support my Collatz research

## Bridge to collatz-box-universes

This repo and [collatz-box-universes](https://github.com/numberwonderman/Collatz-box-universes)
scan the same generalized-Collatz parameter cube `(a, b, c)`: this repo runs
the Benford's-Law statistical analysis (`SwissCheeseParameterScanner`), and
collatz-box-universes renders that same cube in 3D. `bridge_export.py`
exports a scan as structured JSON that `box-universe-viewer.html` can load
directly to color cubes by MAD, digital mixing speed, chi-squared statistic,
etc., instead of only convergence/divergence/cycle behavior.

```bash
python3 bridge_export.py --center 21 14 12 --side 5 --range 1 10000 --output bridge/cube_export.json
```

Then, in `box-universe-viewer.html`'s "Research Data Bridge" panel, load
`bridge/cube_export.json` and pick a metric from the dropdown.

Output schema (`schema_version: "1.0"`):

```json
{
  "schema_version": "1.0",
  "source": "collatz-research",
  "cube_center": [21, 14, 12],
  "side_length": 5,
  "initial_range": [1, 10000],
  "max_iterations": 2000,
  "generated_at": "2026-...",
  "points": [
    {
      "a": 19, "b": 12, "c": 10,
      "total_samples": 123456,
      "mad": 0.00408,
      "chi_squared_statistic": 8500.55,
      "p_value": 0.0,
      "ks_d_max": 0.0135,
      "dmix_variance": 0.018347,
      "digital_mixing_speed": 1627.47
    }
  ]
}
```
