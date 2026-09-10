"""Data loading: bring a real CSV into the pipeline as (X, y, feature_names)."""
import csv
from pathlib import Path

import numpy as np


def load_csv(path, target="label"):
    """Load a CSV file as a binary-classification dataset.

    The ``target`` column becomes ``y``; every other column is treated as a
    numeric feature. Values are coerced to floats (empty cells become NaN so
    a caller can decide what to do with them) and the result is returned as
    ``(X, y, feature_names)`` where ``feature_names`` preserves the CSV's
    original column order.

    Raises ValueError if the file has no usable rows or if ``target`` is not
    one of its columns, so a typo in ``--target`` fails loudly rather than
    silently training on the wrong label.
    """
    p = Path(path)
    with p.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            raise ValueError(f"CSV file {p} has no header row")
        if target not in header:
            raise ValueError(
                f"target column {target!r} not in header {header}"
            )
        target_idx = header.index(target)
        feat_idx = [i for i in range(len(header)) if i != target_idx]
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        if not rows:
            raise ValueError(f"CSV file {p} has no data rows")

    def _cells(row):
        return [
            float(row[i]) if row[i].strip() != "" else np.nan for i in feat_idx
        ]

    X = np.array([_cells(row) for row in rows], dtype=float)
    y = np.array(
        [1.0 if row[target_idx].strip() == "1" else 0.0 for row in rows]
    )
    feature_names = tuple(header[i] for i in feat_idx)
    return X, y.astype(int), feature_names
