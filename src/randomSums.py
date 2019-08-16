import numpy as np
from streamEmbedding import StreamEmbedding


class RandomSums(StreamEmbedding):

    def __init__(self, n, d):
        super().__init__(n, d)
        self.signs = [1.0, -1.0]

    def append(self, vector):
        row = np.random.randint(self.d)
        sign = np.random.choice(self.signs)
        #v = (sign*vector).tolist()
        #self._sketch[row,:] += v[0]
        self._sketch[row, :] += sign * vector
