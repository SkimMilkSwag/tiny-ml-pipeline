# tiny-ml-pipeline

A deliberately small but *complete* machine-learning pipeline: synthetic data
generation, feature standardization, train/test split, model fitting
(LogisticRegression + RandomForestClassifier), and evaluation. The point is the shape
of a real pipeline — not the models — so it's easy to extend with your own
data and learner.

## Install

```bash
pip install -e .
```

## Run

```bash
python -m pipeline.train
# stderr: the test-split confusion matrix for the logistic-regression baseline
# stdout: {"train_acc": 1.0, "test_acc": 1.0, ..., "confusion_matrix": [[a, b], [c, d]]}

# train on your own data instead of the synthetic blobs:
python -m pipeline.train --data path/to/data.csv
```

`--data` expects a CSV with numeric feature columns plus a `label` column of
`0`/`1` values. The feature names flow through into the importance report, so
you get a ranked list for your own dataset.

## Structure

- `pipeline/features.py` — standardization + one-hot helpers
- `pipeline/load.py` — CSV loader (`load_csv`) for real datasets
- `pipeline/train.py` — data generation, splitting, fit/evaluate, and the `run()` entry point (accepts `--data path/to.csv`)

## Extending it

Swap `make_synthetic()` for your own loader, replace the model in `run()`, and
the split/standardize/evaluate scaffolding keeps working.

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
