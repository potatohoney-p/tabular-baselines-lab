import math
from fractions import Fraction

def _finite_mean(values):
    total = sum(values)
    if math.isfinite(total):
        return total / len(values)
    # Finite inputs can overflow their sum even when their mean is finite.
    # Use exact arithmetic only on this exceptional path, before division.
    return float(sum(Fraction(value) for value in values) / len(values))

def matrix(rows):
    rows = [list(map(float, row)) for row in rows]
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError('Expected a nonempty rectangular matrix')
    if any(not math.isfinite(v) for r in rows for v in r):
        raise ValueError('Features must be finite')
    return rows

def fit_centroids(rows, labels):
    rows = matrix(rows)
    if len(rows) != len(labels):
        raise ValueError('One label is required per row')
    groups = {}
    for row, label in zip(rows, labels):
        groups.setdefault(str(label), []).append(row)
    return {label: [_finite_mean(col) for col in zip(*group)]
            for label, group in groups.items()}

def _squared_distances(model, row):
    try:
        distances = {label: sum((x-y)**2 for x,y in zip(row,centroid))
                     for label, centroid in model.items()}
        if all(math.isfinite(distance) for distance in distances.values()):
            return distances
    except OverflowError:
        pass
    # Finite coordinates can overflow subtraction, squaring, or summation.
    # Recompute every candidate from the original values so comparisons and
    # ties use the same exact arithmetic, without rounding back to floats.
    exact_row = [Fraction(value) for value in row]
    return {label: sum((x-Fraction(y))**2 for x,y in zip(exact_row,centroid))
            for label, centroid in model.items()}

def classify(model, row):
    if not model:
        raise ValueError('Empty model')
    row = matrix([row])[0]
    if any(len(c) != len(row) for c in model.values()):
        raise ValueError('Feature dimension mismatch')
    if any(not math.isfinite(value) for centroid in model.values() for value in centroid):
        raise ValueError('Model centroids must be finite')
    distances = _squared_distances(model, row)
    return min(model, key=lambda label: (distances[label],label))

def fit_line(x, y):
    rows = matrix(list(zip(x,y)))
    if len(x) != len(y) or len(rows) < 2:
        raise ValueError('At least two paired observations required')
    mx = sum(r[0] for r in rows)/len(rows)
    my = sum(r[1] for r in rows)/len(rows)
    denominator = sum((r[0]-mx)**2 for r in rows)
    if denominator == 0:
        raise ValueError('Constant predictor has no identifiable slope')
    slope = sum((a-mx)*(b-my) for a,b in rows)/denominator
    return {'slope':slope,'intercept':my-slope*mx}
