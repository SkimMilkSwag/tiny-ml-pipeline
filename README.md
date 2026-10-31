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

# report F1 or ROC-AUC for every model instead of accuracy:
python -m pipeline.train --metric f1      # hard-prediction metric
python -m pipeline.train --metric roc_auc # threshold-free ranking score
```

`--data` expects a CSV with numeric feature columns plus a `label` column of
`0`/`1` values. The feature names flow through into the importance report, so
you get a ranked list for your own dataset.

`--metric` (default `accuracy`) switches the per-model evaluation to `f1` or
`roc_auc`. Accuracy and F1 score hard 0/1 predictions; ROC-AUC scores the model's
continuous output (probability of the positive class), so it does not depend on
the 0.5 decision threshold. Each non-default metric adds a
`<model>_train_<metric>` / `<model>_test_<metric>` pair to the JSON report while
the legacy `*_acc` keys stay, and the chosen metric is echoed as `"metric"` in
the output.

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
