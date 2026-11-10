"""Tests for the configurable CV fold count and _model_scores error path."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pytest
from pipeline.train import METRICS, run


def _sklearn_available():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def test_run_default_cv_uses_three_folds():
    if not _sklearn_available():
        import pytest
        pytest.skip("scikit-learn not installed")
    result = run()
    cv = result["logreg_cv"]
    assert cv["n_folds"] == 3
    assert len(cv["fold_accs"]) == 3


def test_run_custom_n_folds_two():
    if not _sklearn_available():
        import pytest
        pytest.skip("scikit-learn not installed")
    result = run(n_folds=2)
    cv = result["logreg_cv"]
    assert cv["n_folds"] == 2
    # exactly two folds: every sample appears in exactly one validation fold,
    # so the two fold scores are computed on disjoint holdouts
    assert len(cv["fold_accs"]) == 2
    for a in cv["fold_accs"]:
        assert 0.0 <= a <= 1.0
    # well-separated blobs: both folds should still score high
    assert cv["mean_acc"] > 0.85


def test_run_custom_n_folds_five():
    if not _sklearn_available():
        import pytest
        pytest.skip("scikit-learn not installed")
    result = run(n_folds=5)
    cv = result["logreg_cv"]
    assert cv["n_folds"] == 5
    assert len(cv["fold_accs"]) == 5
    # mean is the arithmetic mean of the five fold scores (4 dp rounding)
    assert cv["mean_acc"] == round(sum(cv["fold_accs"]) / 5, 4)


def test_run_n_folds_one_rejected():
    if not _sklearn_available():
        pytest.skip("scikit-learn not installed")
    # one fold has no holdout: the "validation" score would be a training
    # score, so run() must reject it before fitting anything
    with pytest.raises(ValueError, match="at least 2"):
        run(n_folds=1)


def test_run_n_folds_zero_rejected():
    if not _sklearn_available():
        pytest.skip("scikit-learn not installed")
    with pytest.raises(ValueError, match="at least 2"):
        run(n_folds=0)


def test_model_scores_raises_for_model_without_proba_or_decision():
    from pipeline.train import _model_scores

    class Bare:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    X = np.random.default_rng(1).normal(size=(5, 2))
    import pytest
    with pytest.raises(AttributeError, match="neither predict_proba"):
        _model_scores(Bare(), X)


def test_cli_n_folds_flag_changes_fold_count(tmp_path):
    """--n-folds 4 end-to-end: the CV report reflects four folds."""
    import json
    from tests.test_pipeline import _run_cli

    proc = _run_cli(["--n-folds", "4"], cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    cv = report["logreg_cv"]
    assert cv["n_folds"] == 4
    assert len(cv["fold_accs"]) == 4


def test_cli_n_folds_one_fails_loudly(tmp_path):
    """--n-folds 1 exits nonzero with the validation error on stderr."""
    from tests.test_pipeline import _run_cli

    proc = _run_cli(["--n-folds", "1"], cwd=tmp_path)
    assert proc.returncode != 0
    assert "at least 2" in proc.stderr
