# Tabular Baselines Lab

Transparent nearest-centroid classification and least-squares regression on small numeric tables.

## Run locally

Use a currently supported Python 3 interpreter. No third-party packages, credentials, downloads, or network access are required.

```sh
python -m unittest discover -s tests -v
python demo.py
```

Run from the repository directory. The demo reads `examples/input.json`; inputs are reproducible, authored examples.

## Scope and behavior

The classifier stores one arithmetic centroid per label and resolves distance ties lexically. The regression is univariate ordinary least squares. Fit preprocessing only on training rows; the separate scaler and split helpers make leakage visible. Inputs are hand-written numeric fixtures, not an external dataset. These baselines are not calibrated probabilistic models.

## Layout

- `core.py`: primary algorithm or storage contract.
- `features.py`: pipeline or integration layer.
- `tests/`: independent behavioral assertions, failure cases, and regressions.
- `demo.py` and `examples/input.json`: repeatable end-to-end example.

## Example output

```json
{
  "class": "a",
  "line": {
    "intercept": 3.0,
    "slope": 2.0
  },
  "training_mse": 0.0
}
```

## Compatibility and license

The implementation uses language features and standard-library APIs available in [Python 3.4](https://docs.python.org/3.4/library/index.html), released in [March 2014](https://www.python.org/downloads/release/python-340/). Use a supported runtime; an old interpreter is not required. Compatibility with a historical runtime is not certified by current-runtime tests.

MIT licensed. No third-party source code or datasets are bundled.
