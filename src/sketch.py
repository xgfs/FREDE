import numpy as np
from ctypes import *
from ipypb import track

libc = cdll.LoadLibrary('cpp/pprlib.so')
libc.init_rand()


def sketch(sketchers, G, rw=True, nthreads=1):
    n = G.shape[0]
    x = np.zeros(n, dtype=np.float32)
    PPR = np.zeros((n, n), dtype=np.float32)
    inds = G.indptr
    inds = np.append(inds, n).astype(np.int32)
    degrees = G.sum(axis=0).A1.astype(np.int32)
    for i in track(range(n)):
        if rw:
            libc.ppr_row_rw(x.ctypes.data_as(POINTER(c_float)),
                            inds.ctypes.data_as(POINTER(c_int)),
                            G.indices.ctypes.data_as(POINTER(c_int)),
                            degrees.ctypes.data_as(POINTER(c_int)),
                            n, i, c_float(0.85), nthreads, 100000)
        else:
            libc.ppr_row_matmul(x.ctypes.data_as(POINTER(c_float)),
                                inds.ctypes.data_as(POINTER(c_int)),
                                G.indices.ctypes.data_as(POINTER(c_int)),
                                degrees.ctypes.data_as(POINTER(c_int)),
                                n, i, c_float(0.85), nthreads, c_float(1e-6), 100, 2048)

        row = np.zeros_like(x, dtype=np.float32)
        nnz = x.nonzero()
        row[nnz] = np.log(n * x[nnz])
        PPR[i] = row
        for s in sketchers:
            s.append(row)

    return PPR


def sketch_with_PPR(sketchers, PPR):
    n = PPR.shape[0]
    for i in track(range(n)):
        for s in sketchers:
            s.append(PPR[i])

    return PPR
