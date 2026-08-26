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
    """End-to-end: make data -> standardize -> train LogisticRegression -> evaluate."""
    from sklearn.linear_model import LogisticRegression
    from pipeline.features import standardize

    X, y = make_synthetic()
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr, mean, std = standardize(Xtr)
    Xte = (Xte - mean) / std
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    tr_acc = accuracy(ytr, clf.predict(Xtr))
    te_acc = accuracy(yte, clf.predict(Xte))
    return {"train_acc": round(tr_acc, 4), "test_acc": round(te_acc, 4)}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
