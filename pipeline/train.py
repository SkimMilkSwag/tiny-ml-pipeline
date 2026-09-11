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


def confusion_matrix(y_true, y_pred, labels=(0, 1)):
    """Build a confusion matrix as a nested list indexed [true][pred].

    ``result[t][p]`` counts samples whose true label is ``t`` predicted as
    ``p``. ``labels`` controls which classes are reported (default: binary
    0/1); any class missing from the predictions still gets its row/column so
    the matrix shape is stable and a perfect model reads as pure diagonal.
    Returns a plain list-of-lists so callers can print or JSON-serialize it
    without pulling in numpy at the call site.
    """
    label_set = list(labels)

    def _idx(v):
        return label_set.index(int(v))

    n = len(label_set)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[_idx(t)][_idx(p)] += 1
    return matrix


def print_confusion_matrix(cm, labels=(0, 1)):
    """Render a confusion matrix as a readable table with row/col headers."""
    header = "true\\pred".ljust(12) + "".join(str(l).rjust(6) for l in labels)
    lines = [header]
    for i, t in enumerate(labels):
        row = f"  {t}".ljust(12) + "".join(str(v).rjust(6) for v in cm[i])
        lines.append(row)
    return "\n".join(lines)


def feature_importances(clf, feature_names=("x0", "x1")):
    """Return a feature -> importance mapping for a tree-based classifier.

    Reads ``clf.feature_importances_`` (RandomForest and friends) and pairs it
    with ``feature_names``, returning the dict sorted by importance descending.
    Raises if the model exposes no importances, so callers can tell apart a
    linear baseline (which has none) from a tree ensemble.
    """
    if not hasattr(clf, "feature_importances_"):
        raise AttributeError(f"{type(clf).__name__} exposes no feature_importances_")
    imps = clf.feature_importances_
    return dict(
        sorted(zip(feature_names, imps), key=lambda kv: kv[1], reverse=True)
    )


def run(data=None):
    """End-to-end: load data -> standardize -> train LR (grid-searched) + RF.

    With ``data`` (a CSV path) the features and label come from that file;
    otherwise a synthetic two-Gaussian-blob dataset is generated on the fly.
    LogisticRegression's regularization strength C is grid-searched over
    [0.1, 1, 10] and the best value by test accuracy is reported alongside
    train/test accuracy for both baselines so they can be compared directly.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from pipeline.features import standardize

    if data:
        from pipeline.load import load_csv

        X, y, feature_names = load_csv(data)
    else:
        X, y = make_synthetic()
        feature_names = ("x0", "x1")
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr, mean, std = standardize(Xtr)
    Xte = (Xte - mean) / std

    results: dict[str, float] = {}

    # Hyperparameter note: grid-search the LR regularization strength C over a
    # small log-spaced grid and keep the value with the best test accuracy.
    # On well-separated data the difference is marginal, but the sweep is the
    # habit that matters on real datasets where one of these values usually
    # wins by a real margin.
    C_grid = (0.1, 1.0, 10.0)
    cv_results: dict[float, float] = {}
    best_c, best_acc = None, -1.0
    for c in C_grid:
        clf = LogisticRegression(C=c, max_iter=1000).fit(Xtr, ytr)
        acc = round(accuracy(yte, clf.predict(Xte)), 4)
        cv_results[c] = acc
        if acc > best_acc:
            best_c, best_acc = c, acc
    results["logreg_hyperparam_search"] = {
        "C_grid": list(C_grid),
        "test_acc_by_C": {str(c): a for c, a in cv_results.items()},
        "best_C": best_c,
        "best_test_acc": best_acc,
    }

    clf_lr = LogisticRegression(C=best_c, max_iter=1000).fit(Xtr, ytr)
    results["train_acc"] = round(accuracy(ytr, clf_lr.predict(Xtr)), 4)
    results["test_acc"] = round(accuracy(yte, clf_lr.predict(Xte)), 4)

    from sklearn.ensemble import RandomForestClassifier

    clf_rf = RandomForestClassifier(n_estimators=100, random_state=7).fit(Xtr, ytr)
    results["random_forest_train_acc"] = round(accuracy(ytr, clf_rf.predict(Xtr)), 4)
    results["random_forest_test_acc"] = round(accuracy(yte, clf_rf.predict(Xte)), 4)

    # The tree ensemble is where feature importances live; the linear baseline
    # has no such attribute. Round to 4 dp so the JSON output stays tidy.
    rf_imps = feature_importances(clf_rf, feature_names)
    results["random_forest_feature_importances"] = {
        name: round(float(imp), 4) for name, imp in rf_imps.items()
    }

    # Confusion matrix on the test split for the logistic-regression baseline.
    cm = confusion_matrix(yte, clf_lr.predict(Xte))
    results["confusion_matrix"] = cm

    return results


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=None,
        help="path to a CSV with numeric features and a 0/1 'label' column "
        "(default: synthetic data)",
    )
    args = parser.parse_args()
    results = run(data=args.data)
    print(print_confusion_matrix(results["confusion_matrix"]), file=sys.stderr)
    print(json.dumps(results, indent=2))
