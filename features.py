import random
from core import matrix
def split_indices(size, fraction=0.25, seed=7):
    if size < 2 or not 0 < fraction < 1:
        raise ValueError('Need two rows and a fraction between zero and one')
    indices=list(range(size)); random.Random(seed).shuffle(indices)
    cut=max(1,min(size-1,int(size*fraction)))
    return indices[cut:],indices[:cut]
def fit_scaler(rows):
    rows=matrix(rows)
    means=[sum(c)/len(rows) for c in zip(*rows)]
    scales=[(sum((v-m)**2 for v in c)/len(rows))**0.5 or 1.0
            for c,m in zip(zip(*rows),means)]
    return means,scales
def transform(rows, scaler):
    rows=matrix(rows); means,scales=scaler
    if len(rows[0]) != len(means) or len(rows[0]) != len(scales):
        raise ValueError('Dimension mismatch')
    return [[(v-m)/s for v,m,s in zip(r,means,scales)] for r in rows]
def mse(actual,predicted):
    if not actual or len(actual)!=len(predicted): raise ValueError('Paired values required')
    return sum((a-b)**2 for a,b in zip(actual,predicted))/len(actual)
