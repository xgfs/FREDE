import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def covariance_error(A, sketch):
    squared_frob_A = (A * A).sum()
    diff = A.T @ A - sketch.T @ sketch  # ATA - BTB
    cov_err = np.linalg.norm(diff, 2) / squared_frob_A
    return cov_err


def projection_error(A, sketch, k):
    U, s, Vh = svds(A, k)
    A_k = U @ np.diag(s) @ Vh
    _, _, Vh = svds(sketch, k)
    proj = A @ Vh.T @ Vh  # A @ VVT
    proj_err = np.linalg.norm(A - proj) ** 2
    rank_k_err = np.linalg.norm(A - A_k) ** 2
    proj_err = proj_err / rank_k_err
    return proj_err


def pip_loss(A, sketch, normed=True):
    if normed:
        As = cosine_similarity(A.T)
        Ss = cosine_similarity(sketch.T)
    else:
        As = A.T @ A
        Ss = sketch.T @ sketch
    pip_err = np.linalg.norm(As - Ss)

    return pip_err


def rotate(embs):
    [_, s, Vt] = np.linalg.svd(embs, full_matrices=False)
    return np.diag(np.sqrt(s)) @ Vt


def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


class LazyClass(object):
    def __getattr__(self, attr):
        try:
            return object.__getattr__(attr)
        except AttributeError:
            return self.do_nothing

    @staticmethod
    def do_nothing(*args, **kwargs):
        pass
