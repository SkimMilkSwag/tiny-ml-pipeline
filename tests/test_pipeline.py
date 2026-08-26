import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.train import make_synthetic, split, run, accuracy


def test_make_synthetic_shape():
    X, y = make_synthetic(n=100)
    assert X.shape == (100, 2)
    assert set(y.tolist()) == {0, 1}


def test_split_disjoint():
    X, y = make_synthetic(n=200)
    Xtr, Xte, ytr, yte = split(X, y, test_frac=0.4)
    assert len(ytr) + len(yte) == len(y)


def test_run_gives_high_acc():
    result = run()
    assert result["test_acc"] > 0.85  # blobs are well-separated
