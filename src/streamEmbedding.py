import numpy as np
from abc import ABC, abstractmethod


class StreamEmbedding(ABC):

    def __init__(self, n, d):
        self.n = n
        self.d = d
        self._sketch = np.zeros((self.d, self.n), dtype=np.float32)
        super().__init__()

    # Appending a row vector to sketch
    @abstractmethod
    def append(self, vector):
        pass

    # Convenient looping numpy matrices row by row
    def extend(self, vectors):
        for vector in vectors:
            self.append(vector)

    # returns the sketch matrix
    def get(self):
        return self._sketch

    # Convenience support for the += operator  append
    def __iadd__(self, vector):
        self.append(vector)
        return self

    def __str__(self):
        return f'{type(self).__name__}_{self.n}_{self.d}'
