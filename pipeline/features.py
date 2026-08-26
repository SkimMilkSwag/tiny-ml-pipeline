"""Feature engineering helpers."""
import numpy as np


def standardize(X):
    """Z-score standardize features (column-wise). Returns X, mean, std."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def one_hot_encode(df, col):
    """One-hot encode a categorical column into new boolean columns."""
    import pandas as pd
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
    return pd.concat([df.drop(columns=[col]), dummies], axis=1)
