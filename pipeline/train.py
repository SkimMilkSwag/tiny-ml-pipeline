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


def f1_score(y_true, y_pred, pos_label=1):
    """Binary F1 for ``pos_label``: precision and recall from the confusion counts.

    ``F1 = 2 * P * R / (P + R)``, with the standard 0.0 convention when the model
    never predicts the positive class (precision undefined, no true positives).
    Kept numpy-free like ``accuracy`` so callers don't need an array input — it
    works on any iterable of labels.
    """
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if int(p) == pos_label and int(t) == pos_label:
            tp += 1
        elif int(p) == pos_label:
            fp += 1
        elif int(t) == pos_label:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def roc_auc_score(y_true, y_prob):
    """Binary ROC-AUC as the rank statistic (Mann-Whitney U) of the scores.

    Counts, over every positive/negative pair, how often a random positive is
    scored above a random negative (ties count half). This is the definition of
    AUC for a binary classifier's ranking quality, computed without any
    threshold, so it works on raw decision values as well as probabilities.
    Runs in O(P*N) — fine for the test splits this pipeline evaluates; a
    production system would sort the scores and walk the ROC curve instead.
    """
    pos_scores = [float(s) for t, s in zip(y_true, y_prob) if int(t) == 1]
    neg_scores = [float(s) for t, s in zip(y_true, y_prob) if int(t) == 0]
    if not pos_scores or not neg_scores:
        raise ValueError(
            "roc_auc_score needs at least one sample of each class "
            f"(got {len(pos_scores)} positives, {len(neg_scores)} negatives)"
        )
    concordant = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                concordant += 1.0
            elif p == n:
                concordant += 0.5
    return float(concordant / (len(pos_scores) * len(neg_scores)))


METRICS = {
    "accuracy": accuracy,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
}
"""Registry of evaluation metrics by name; ``run(metric=...)`` looks them up here.

``accuracy`` and ``f1`` score hard predictions (0/1); ``roc_auc`` scores the
model's continuous output (``decision_function`` / ``predict_proba``) so it does
not depend on the 0.5 decision threshold.
"""


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


def _model_scores(model, X):
    """Continuous scores a metric can rank against: probabilities if the model
    exposes them, else raw decision values. Used for ``roc_auc``, which needs an
    ordered score rather than a hard 0/1 prediction."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def run(data=None, metric="accuracy"):
    """End-to-end: load data -> standardize -> train LR (grid-searched), RF, HGB.

    With ``data`` (a CSV path) the features and label come from that file;
    otherwise a synthetic two-Gaussian-blob dataset is generated on the fly.
    LogisticRegression's regularization strength C is grid-searched over
    [0.1, 1, 10] and the best value by test accuracy is reported alongside
    train/test scores for all three models (LR baseline, RandomForest, and
    HistGradientBoosting) so they can be compared directly on the same split.

    ``metric`` names the evaluation metric used to score every model: one of
    ``METRICS`` — ``accuracy``, ``f1``, or ``roc_auc``. accuracy and f1 score
    hard predictions; roc_auc scores the model's continuous output so it is
    threshold-free. Each model gets a ``<model>_train_<metric>`` /
    ``<model>_test_<metric>`` pair in the results; the legacy ``*_acc`` keys
    always stay (they are accuracy) so existing consumers keep working.
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

    if metric not in METRICS:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of {sorted(METRICS)}"
        )

    results: dict[str, float] = {}
    results["metric"] = metric

    def _metric_scores(model, Xtr_, ytr_, Xte_, yte_):
        """Score one model on the train/test splits with ``metric``.

        accuracy and f1 score hard predictions; roc_auc scores continuous model
        output so it does not depend on the 0.5 decision threshold.
        """
        if metric == "roc_auc":
            return (
                round(METRICS[metric](ytr_, _model_scores(model, Xtr_)), 4),
                round(METRICS[metric](yte_, _model_scores(model, Xte_)), 4),
            )
        return (
            round(METRICS[metric](ytr_, model.predict(Xtr_)), 4),
            round(METRICS[metric](yte_, model.predict(Xte_)), 4),
        )

    # Hyperparameter note: grid-search the LR regularization strength C over a
    # small log-spaced grid and keep the value with the best test accuracy.
    # On well-separated data the difference is marginal, but the sweep is the
    # habit that matters on real datasets where one of these values usually
    # wins by a real margin. The winning estimator itself is kept (not just
    # its C) so CV below reuses it instead of refitting.
    C_grid = (0.1, 1.0, 10.0)
    cv_results: dict[float, float] = {}
    best_c, best_acc, best_clf = None, -1.0, None
    for c in C_grid:
        clf = LogisticRegression(C=c, max_iter=1000).fit(Xtr, ytr)
        acc = round(accuracy(yte, clf.predict(Xte)), 4)
        cv_results[c] = acc
        if acc > best_acc:
            best_c, best_acc, best_clf = c, acc, clf
    results["logreg_hyperparam_search"] = {
        "C_grid": list(C_grid),
        "test_acc_by_C": {str(c): a for c, a in cv_results.items()},
        "best_C": best_c,
        "best_test_acc": best_acc,
    }

    clf_lr = best_clf
    lr_train, lr_test = _metric_scores(clf_lr, Xtr, ytr, Xte, yte)
    results["train_acc"] = round(accuracy(ytr, clf_lr.predict(Xtr)), 4)
    results["test_acc"] = round(accuracy(yte, clf_lr.predict(Xte)), 4)
    if metric != "accuracy":
        results[f"logreg_train_{metric}"] = lr_train
        results[f"logreg_test_{metric}"] = lr_test

    from sklearn.ensemble import RandomForestClassifier

    clf_rf = RandomForestClassifier(n_estimators=100, random_state=7).fit(Xtr, ytr)
    results["random_forest_train_acc"] = round(accuracy(ytr, clf_rf.predict(Xtr)), 4)
    results["random_forest_test_acc"] = round(accuracy(yte, clf_rf.predict(Xte)), 4)
    if metric != "accuracy":
        rf_train, rf_test = _metric_scores(clf_rf, Xtr, ytr, Xte, yte)
        results[f"random_forest_train_{metric}"] = rf_train
        results[f"random_forest_test_{metric}"] = rf_test

    # The tree ensemble is where feature importances live; the linear baseline
    # has no such attribute. Round to 4 dp so the JSON output stays tidy.
    rf_imps = feature_importances(clf_rf, feature_names)
    results["random_forest_feature_importances"] = {
        name: round(float(imp), 4) for name, imp in rf_imps.items()
    }

    # Gradient-boosted trees as the third comparison point. HistGradientBoosting
    # is a histogram-based learner that handles mixed feature scales without
    # standardization, so it's a fair head-to-head against both baselines on
    # the same split.
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf_hgb = HistGradientBoostingClassifier(max_iter=200, random_state=7).fit(
        Xtr, ytr
    )
    results["hist_gradient_boosting_train_acc"] = round(
        accuracy(ytr, clf_hgb.predict(Xtr)), 4
    )
    results["hist_gradient_boosting_test_acc"] = round(
        accuracy(yte, clf_hgb.predict(Xte)), 4
    )
    if metric != "accuracy":
        hgb_train, hgb_test = _metric_scores(clf_hgb, Xtr, ytr, Xte, yte)
        results[f"hist_gradient_boosting_train_{metric}"] = hgb_train
        results[f"hist_gradient_boosting_test_{metric}"] = hgb_test

    # Confusion matrix on the test split for the logistic-regression baseline.
    cm = confusion_matrix(yte, clf_lr.predict(Xte))
    results["confusion_matrix"] = cm

    # Cross-validation sanity check: holdout accuracy can flatter a model if
    # the split happens to be easy or hard, so the winning LR is also scored
    # with k-fold CV (stratified, folds are disjoint and cover the training
    # set exactly) before we trust it as "the best model". The per-fold
    # accuracies go in too — a spread there is the early warning that a single
    # holdout number is noise.
    n_folds = 3
    cv_scores: list[float] = []
    for tr_idx, va_idx in _stratified_folds(ytr, n_folds):
        fold_clf = LogisticRegression(C=best_c, max_iter=1000).fit(
            Xtr[tr_idx], ytr[tr_idx]
        )
        cv_scores.append(round(accuracy(ytr[va_idx], fold_clf.predict(Xtr[va_idx])), 4))
    results["logreg_cv"] = {
        "n_folds": n_folds,
        "fold_accs": cv_scores,
        "mean_acc": round(sum(cv_scores) / len(cv_scores), 4),
    }

    return results


def _stratified_folds(y, k):
    """Yield (train_idx, val_idx) arrays for k stratified folds over ``y``.

    Each class's samples are shuffled independently and round-robin assigned to
    the k folds, so every validation fold sees roughly the same class mix as
    the full dataset — the failure mode of a random split on imbalanced data is
    exactly what this prevents. Folds are disjoint and together cover every
    index exactly once; train for fold i is everything except fold i's rows.
    """
    rng = np.random.default_rng(7)
    order: dict[int, list[int]] = {}
    for cls in sorted(set(int(v) for v in y)):
        idx = [i for i, v in enumerate(y) if int(v) == cls]
        shuffled = list(rng.permutation(idx))
        order[cls] = shuffled
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in sorted(order):
        # deal each class round-robin over the fold count, independent of any
        # other class's remainder — a shared running counter would let the
        # second class's phase drift and starve later folds of small classes.
        for j, i in enumerate(order[cls]):
            folds[j % k].append(i)
    n = len(y)
    for val in folds:
        val_set = set(val)
        tr_idx = np.array([i for i in range(n) if i not in val_set], dtype=int)
        yield tr_idx, np.array(sorted(val), dtype=int)


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
    parser.add_argument(
        "--metric",
        choices=sorted(METRICS),
        default="accuracy",
        help="evaluation metric reported for every model (default: accuracy); "
        "f1 scores hard predictions, roc_auc scores continuous model output",
    )
    args = parser.parse_args()
    results = run(data=args.data, metric=args.metric)
    print(print_confusion_matrix(results["confusion_matrix"]), file=sys.stderr)
    print(json.dumps(results, indent=2))
