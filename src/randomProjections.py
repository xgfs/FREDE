import numpy as np
from streamEmbedding import StreamEmbedding


class RandomProjections(StreamEmbedding):

    def __init__(self, n, d):
        super().__init__(n, d)
        self.rescaled_signs = [-1.0, 1.0] / np.sqrt(self.d)

    def append(self, vector):
        randomVector = np.random.choice(self.rescaled_signs, self.d)
        self._sketch += np.outer(randomVector, vector)
