import numpy as np
from abc import ABC, abstractmethod


class FullMatrixEmbedding(ABC):

    def __init__(self, d):
        self.d = d
        super().__init__()

    # Appending a row vector to sketch
    @abstractmethod
    def compute(self, matrix):
        pass

    # returns the embedding matrix
    @abstractmethod
    def get(self):
        pass

    def __str__(self):
        return f'{type(self).__name__}_{self.d}'
