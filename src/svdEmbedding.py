import numpy as np
from scipy.sparse.linalg import svds as scipy_svds
from sklearn.decomposition import TruncatedSVD
from fullMatrixEmbedding import FullMatrixEmbedding


class SVDEmbedding(FullMatrixEmbedding):
    # FIXME n is irrelevant but is added for the call to sketcher to be
    # compatible with other sketchers
    def __init__(self, n, d, left_vectors=True, algo='full'):
        super().__init__(d)
        if algo not in ['full', 'halko', 'arpack']:
            raise ValueError('Unknown SVD algorithm specified!')
        if algo == 'halko' and not left_vectors:
            raise ValueError(
                'Halko solver only supports left singular vectors!')
        self.left_vectors = left_vectors
        self.algo = algo
        self.d = d

    def compute(self, matrix):
        if self.algo == 'full':
            [U, s, V] = np.linalg.svd(matrix, full_matrices=False)
            U, V = U[:, :self.d], V[:self.d, :]
        elif self.algo == 'arpack':
            [U, s, V] = scipy_svds(matrix, self.d)
        else:
            svd = TruncatedSVD(n_components=self.d)
            svd.fit(matrix)
            U, s = svd.components_.T, svd.singular_values_
            V = U  # Tis a hack
        self.U = U
        self.s = s
        self.V = V

    def get(self, d=None, degree=False):
        if d is None:
            d = self.d
        if d > self.d:
            raise ValueError('Requested dimension too high!')
        if self.left_vectors:
            if degree:
                return (self.U[:, :d] @ np.diag(np.power(self.s[:d], degree))).T
            else:
                return (self.U[:, :d] @ np.diag(self.s[:d])).T
        else:
            if degree:
                return np.diag(np.power(self.s[:d], degree)) @ self.V[:d, :]
            else:
                return np.diag(self.s[:d]) @ self.V[:d, :]

    def __str__(self):
        vectors = 'left' if self.left_vectors else 'right'
        return f'{type(self).__name__}_{self.algo}_{self.d}_{vectors}'
