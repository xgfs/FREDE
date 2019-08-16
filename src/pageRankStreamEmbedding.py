from ctypes import *
import numpy as np
from streamEmbedding import StreamEmbedding


class PageRankStreamEmbedding():

    def __init__(
            self,
            graph,
            rw,
            alpha=0.85,
            rw_count=100000,
            poweriter_tol=1e-5,
            poweriter_iter=100,
            nthreads=1,
            libpath='cpp/pprlib.so'):
        self.libc = cdll.LoadLibrary(libpath)
        self.libc.init_rand()

        self.n = graph.shape[0]
        self.x = np.zeros(self.n, dtype=np.float32)
        self.row = np.zeros(self.n, dtype=np.float32)
        self.inds = graph.indptr
        self.inds = np.append(self.inds, self.n).astype(np.int32)
        self.indices = graph.indices
        self.degrees = graph.sum(axis=0).A1.astype(np.int32)
        self.current = 0
        self.rw = rw
        self.nthreads = nthreads
        self.rw_count = 100000
        self.alpha = alpha
        self.poweriter_tol = poweriter_tol
        self.poweriter_iter = poweriter_iter

    def __next__(self):
        if self.current > self.n:
            raise StopIteration
        else:
            if self.rw:
                self.libc.ppr_row_rw(
                    self.x.ctypes.data_as(
                        POINTER(c_float)), self.inds.ctypes.data_as(
                        POINTER(c_int)), self.indices.ctypes.data_as(
                        POINTER(c_int)), self.degrees.ctypes.data_as(
                        POINTER(c_int)), self.n, self.current, c_float(
                        self.alpha), self.nthreads, self.rw_count)
            else:
                self.libc.ppr_row_matmul(
                    self.x.ctypes.data_as(
                        POINTER(c_float)), self.inds.ctypes.data_as(
                        POINTER(c_int)), self.indices.ctypes.data_as(
                        POINTER(c_int)), self.degrees.ctypes.data_as(
                        POINTER(c_int)), self.n, self.current, c_float(
                        self.alpha), self.nthreads, c_float(self.poweriter_tol), self.poweriter_iter, 2048)
            return self.x
