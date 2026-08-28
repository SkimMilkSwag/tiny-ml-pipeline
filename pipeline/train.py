"""A tiny but complete train/evaluate pipeline."""
import numpy as np


def make_synthetic(n=500, seed=7):
    """Generate a synthetic binary-classification dataset (two Gaussian blobs)."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal([0, 0], 1.0, size=(n // 2, 2))
    x1 = rng.normal([4, 4], 1.2, size=(n - n // 2, 2))
    X = np.vstack([x0, x1])
    y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    idx = rng.permutation(n)
    return X[idx], y[idx]


def split(X, y, test_frac=0.3, seed=7):
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def run():
    """End-to-end: make data -> standardize -> train LR + RF -> evaluate.

    Returns a dict with train/test accuracy for both LogisticRegression and
    RandomForestClassifier so the two baselines can be compared directly.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from pipeline.features import standardize

    X, y = make_synthetic()
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr, mean, std = standardize(Xtr)
    Xte = (Xte - mean) / std

    results: dict[str, float] = {}

    clf_lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    results["train_acc"] = round(accuracy(ytr, clf_lr.predict(Xtr)), 4)
    results["test_acc"] = round(accuracy(yte, clf_lr.predict(Xte)), 4)

    clf_rf = RandomForestClassifier(n_estimators=100, random_state=7).fit(Xtr, ytr)
    results["random_forest_train_acc"] = round(accuracy(ytr, clf_rf.predict(Xtr)), 4)
    results["random_forest_test_acc"] = round(accuracy(yte, clf_rf.predict(Xte)), 4)

    return results


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
