import ctypes
from ipypb import track
from scipy.io import loadmat
from ctypes import *
import numpy as np

from telepyth import TelepythClient
tp = TelepythClient()

mkl_rt = ctypes.CDLL('libmkl_rt.so')
print('CPUs used before: ', mkl_rt.mkl_get_max_threads())
mkl_get_max_threads = mkl_rt.mkl_get_max_threads


def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))


mkl_set_num_threads(20)
print('CPUs used now: ', mkl_get_max_threads())

NTHREADS = 20  # !

libppr = cdll.LoadLibrary('cpp/pprlib.so')
libppr.init_rand()

datapath = '/data/frededata/'

for mat in track(['academic_confs.mat',
                  'academic_coa_2014.mat',
                  'flickr.mat',
                  'vk2016.mat']):
    tp.send_text(mat[:-4] + ' started ppr evaluation')
    matf = loadmat(datapath + mat)
    G = matf['network']
    n = G.shape[0]
    print(mat, 'n = ', n)
    inds = G.indptr
    inds = np.append(inds, n).astype(np.int32)
    degrees = G.sum(axis=0).A1.astype(np.int32)
    ppr = np.zeros(n * n, dtype=np.float32)
    libppr.ppr_mat_matmul(ppr.ctypes.data_as(POINTER(c_float)),
                          inds.ctypes.data_as(POINTER(c_int)),
                          G.indices.ctypes.data_as(POINTER(c_int)),
                          degrees.ctypes.data_as(POINTER(c_int)),
                          n, c_float(0.85), 1, c_double(1e-6), 100, 2048)  # LOOONG
    ppr = ppr.reshape(n, n)
    np.save(datapath + 'ppr/%s' % mat[:-4], ppr)
    print(mat[:-4] + ' saved!')
    tp.send_text(mat[:-4] + ' ppr computed and saved.')
tp.send_text('Evaluation completed.')
