import numpy as np
from fullMatrixEmbedding import FullMatrixEmbedding


class RandomProjectionsMat(FullMatrixEmbedding):

    def __init__(self, n, d):
        super().__init__(d)
        self.d = d
        self.n = n
        self.rescaled_signs = [-1.0, 1.0] / np.sqrt(self.d)
        self.R = np.random.choice(self.rescaled_signs, self.n*self.d).reshape(self.d, self.n)
        
    def compute(self, matrix):
        self.sketch = self.R @ matrix
        
    def get(self):
        return self.sketch
