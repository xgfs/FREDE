import numpy as np
from scipy.linalg import svd as scipy_svd
from scipy.sparse.linalg import svds as scipy_svds

from streamEmbedding import StreamEmbedding


class FrequentDirections(StreamEmbedding):

    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.m = 2 * self.d
        self._sketch = np.zeros((self.m, self.n), dtype=np.float32)
        self.nextZeroRow = 0

    def append(self, vector):
        if np.count_nonzero(vector) == 0:
            return

        if self.nextZeroRow >= self.m:
            self.__rotate__()

        self._sketch[self.nextZeroRow, :] = vector
        self.nextZeroRow += 1

    def __rotate__(self):
        try:
            [_, s, Vt] = np.linalg.svd(self._sketch, full_matrices=False)
        except np.linalg.LinAlgError:
            [_, s, Vt] = scipy_svd(self._sketch, full_matrices=False)
        #[_,s,Vt] = scipy_svds(self._sketch, k = self.d)

        sShrunk = np.sqrt(s[:self.d]**2 - s[self.d - 1]**2)
        self._sketch[:self.d:, :] = np.dot(
            np.diag(sShrunk), Vt[:self.d, :])
        self._sketch[self.d:, :] = 0
        self.nextZeroRow = self.d

    def get(self, rotate=False, take_root=True):
        if rotate:
            try:
                [_, s, Vt] = np.linalg.svd(self._sketch, full_matrices=False)
            except np.linalg.LinAlgError:
                [_, s, Vt] = scipy_svd(self._sketch, full_matrices=False)
            if take_root:
                return np.diag(np.sqrt(s[:self.d])) @ Vt[:self.d, :]
            else:
                return np.diag(s[:self.d]) @ Vt[:self.d, :]

        return self._sketch[:self.d, :]
