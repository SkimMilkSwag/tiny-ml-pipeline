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
# -> {"train_acc": 1.0, "test_acc": 1.0,
#     "random_forest_train_acc": 1.0, "random_forest_test_acc": 1.0}
```

## Structure

- `pipeline/features.py` — standardization + one-hot helpers
- `pipeline/train.py` — data generation, splitting, fit/evaluate, and the `run()` entry point

## Extending it

Swap `make_synthetic()` for your own loader, replace the model in `run()`, and
the split/standardize/evaluate scaffolding keeps working.

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
