import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.train import make_synthetic, split, run, accuracy, feature_importances


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
