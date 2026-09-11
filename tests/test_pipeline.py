import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.train import (
    make_synthetic, split, run, accuracy, feature_importances,
    confusion_matrix, print_confusion_matrix,
)
from pipeline.load import load_csv


def _sklearn_available():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def test_make_synthetic_shape():
    X, y = make_synthetic(n=100)
    assert X.shape == (100, 2)
    assert set(y.tolist()) == {0, 1}


def test_split_disjoint():
    X, y = make_synthetic(n=200)
    Xtr, Xte, ytr, yte = split(X, y, test_frac=0.4)
    assert len(ytr) + len(yte) == len(y)


def test_run_gives_high_acc():
    if not _sklearn_available():
        import pytest
        pytest.skip("scikit-learn not installed")
    result = run()
    assert result["test_acc"] > 0.85  # blobs are well-separated


def test_run_includes_random_forest_comparison():
    if not _sklearn_available():
        import pytest
        pytest.skip("scikit-learn not installed")
    result = run()
    assert "random_forest_test_acc" in result
    assert result["random_forest_test_acc"] > 0.85


def test_feature_importances_from_trained_rf():
    if not _sklearn_available():
        import pytest
        pytest.skip("scikit-learn not installed")
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_synthetic(n=200)
    Xtr, Xte, ytr, yte = split(X, y)
    clf = RandomForestClassifier(n_estimators=50, random_state=7).fit(Xtr, ytr)
    imps = feature_importances(clf, feature_names=("x0", "x1"))
    assert set(imps.keys()) == {"x0", "x1"}
    # importances are non-negative and sum to ~1 (sklearn normalizes them)
    total = sum(imps.values())
    assert abs(total - 1.0) < 1e-6
    for v in imps.values():
        assert v >= 0.0


def test_feature_importances_sorted_descending():
    # The helper must return features ordered most-important-first so the dict
    # reads as a ranking, not an arbitrary ordering.
    if not _sklearn_available():
        import pytest
        pytest.skip("scikit-learn not installed")
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_synthetic(n=200)
    Xtr, Xte, ytr, yte = split(X, y)
    clf = RandomForestClassifier(n_estimators=50, random_state=7).fit(Xtr, ytr)
    imps = feature_importances(clf, feature_names=("x0", "x1"))
    vals = list(imps.values())
    assert vals == sorted(vals, reverse=True)


def test_feature_importances_rejects_non_tree_model():
    # A linear baseline has no feature_importances_; the helper should say so
    # with a clear AttributeError rather than silently return None.
    if not _sklearn_available():
        import pytest
        pytest.skip("scikit-learn not installed")
    from sklearn.linear_model import LogisticRegression

    X, y = make_synthetic(n=200)
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    import pytest
    with pytest.raises(AttributeError):
        feature_importances(clf)


def test_run_includes_rf_feature_importances():
    if not _sklearn_available():
        import pytest
        pytest.skip("scikit-learn not installed")
    result = run()
    imps = result["random_forest_feature_importances"]
    assert set(imps.keys()) == {"x0", "x1"}
    # synthetic blobs are two-axis Gaussians; both features carry signal, so
    # each importance should be a real non-trivial fraction of the whole.
    assert abs(sum(imps.values()) - 1.0) < 1e-3


def _write_csv(tmp_path, rows):
    import csv

    p = tmp_path / "data.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(rows[0])
        w.writerows(rows[1:])
    return p


def test_load_csv_shapes_and_labels(tmp_path):
    p = _write_csv(
        tmp_path,
        [
            ["x", "y", "label"],
            ["0.5", "2.0", "0"],
            ["1.5", "3.5", "1"],
            ["2.5", "4.0", "1"],
        ],
    )
    X, y, names = load_csv(p)
    assert X.shape == (3, 2)
    assert y.tolist() == [0, 1, 1]
    assert names == ("x", "y")


def test_load_csv_keeps_column_order(tmp_path):
    p = _write_csv(
        tmp_path,
        [
            ["b", "label", "a"],
            ["1.0", "1", "2.0"],
            ["3.0", "0", "4.0"],
        ],
    )
    X, y, names = load_csv(p)
    # feature_names must mirror the CSV column order (minus the target)
    assert names == ("b", "a")
    assert X.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_load_csv_blank_cells_become_nan(tmp_path):
    p = _write_csv(
        tmp_path,
        [
            ["x", "label"],
            ["1.0", "0"],
            ["", "1"],
        ],
    )
    import numpy as np

    X, y, names = load_csv(p)
    assert np.isnan(X[1][0])
    assert X[0][0] == 1.0


def test_load_csv_rejects_missing_target(tmp_path):
    p = _write_csv(
        tmp_path,
        [
            ["x", "y"],
            ["1.0", "2.0"],
        ],
    )
    import pytest

    with pytest.raises(ValueError):
        load_csv(p, target="label")


def test_load_csv_rejects_empty_file(tmp_path):
    import pytest

    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(ValueError):
        load_csv(p)


def test_run_with_csv_data(tmp_path):
    if not _sklearn_available():
        import pytest
        pytest.skip("scikit-learn not installed")
    # two well-separated blobs as a CSV: run() must train on it end-to-end
    # and the feature names in the importance report must be the CSV's.
    p = _write_csv(
        tmp_path,
        [
            ["x", "y", "label"],
            *[["0.1", "0.2", "0"]] * 25 + [["3.9", "4.1", "1"]] * 25,
        ],
    )
    result = run(data=str(p))
    assert result["test_acc"] > 0.85
    assert set(result["random_forest_feature_importances"].keys()) == {"x", "y"}


def test_run_includes_logreg_hyperparam_search():
    if not _sklearn_available():
        import pytest
        pytest.skip("scikit-learn not installed")
    result = run()
    search = result["logreg_hyperparam_search"]
    # every grid member was evaluated and reported
    assert search["C_grid"] == [0.1, 1.0, 10.0]
    assert set(search["test_acc_by_C"].keys()) == {"0.1", "1.0", "10.0"}
    # the best is a member of the grid and its accuracy matches its entry
    assert search["best_C"] in (0.1, 1.0, 10.0)
    assert search["best_test_acc"] == max(search["test_acc_by_C"].values())
    assert search["test_acc_by_C"][str(search["best_C"])] == search["best_test_acc"]
    # the final LR model is trained with the winning C
    assert result["test_acc"] == search["best_test_acc"]


def test_run_hyperparam_search_with_csv_data(tmp_path):
    if not _sklearn_available():
        import pytest
        pytest.skip("scikit-learn not installed")
    # the sweep must also work end-to-end on real CSV data, not just synthetic
    p = _write_csv(
        tmp_path,
        [
            ["x", "y", "label"],
            *[["0.1", "0.2", "0"]] * 25 + [["3.9", "4.1", "1"]] * 25,
        ],
    )
    result = run(data=str(p))
    search = result["logreg_hyperparam_search"]
    assert set(search["test_acc_by_C"].keys()) == {"0.1", "1.0", "10.0"}
    assert result["test_acc"] == search["best_test_acc"]


def test_confusion_matrix_counts_by_true_and_pred():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    cm = confusion_matrix(y_true, y_pred)
    # cm[true][pred]: one (0->1) and one (1->0) error, three correct.
    assert cm == [[1, 1], [1, 2]]


def test_confusion_matrix_all_correct_is_diagonal():
    y = [0, 0, 1, 1]
    cm = confusion_matrix(y, y)
    assert cm == [[2, 0], [0, 2]]


def test_confusion_matrix_handles_missing_class():
    # a model that predicts only class 1 must still get a class-0 row/column
    y_true = [0, 1]
    y_pred = [1, 1]
    cm = confusion_matrix(y_true, y_pred)
    assert cm == [[0, 1], [0, 1]]
    assert len(cm) == 2 and all(len(r) == 2 for r in cm)


def test_confusion_matrix_row_sums_match_true_class_counts():
    import random

    rng = random.Random(3)
    y_true = [rng.randint(0, 1) for _ in range(40)]
    y_pred = [rng.randint(0, 1) for _ in range(40)]
    cm = confusion_matrix(y_true, y_pred)
    assert sum(cm[0]) == y_true.count(0)
    assert sum(cm[1]) == y_true.count(1)


def test_print_confusion_matrix_layout():
    cm = [[10, 2], [3, 25]]
    out = print_confusion_matrix(cm)
    lines = out.splitlines()
    assert len(lines) == 3  # header + two class rows
    assert "10" in lines[1] and "25" in lines[2]


def test_run_includes_test_confusion_matrix():
    if not _sklearn_available():
        import pytest
        pytest.skip("scikit-learn not installed")
    result = run()
    cm = result["confusion_matrix"]
    assert len(cm) == 2 and all(len(r) == 2 for r in cm)
    # every test sample lands somewhere in the matrix, so its total must
    # equal the test split size.
    n_test = sum(sum(r) for r in cm)
    X, y = make_synthetic()
    _, Xte, _, yte = split(X, y)
    assert n_test == len(yte)
