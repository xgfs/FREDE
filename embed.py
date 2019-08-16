import numpy as np
import ctypes
import click
import logging
import os
import sys
sys.path.append('src/')


from frequentDirections import FrequentDirections
from randomProjections import RandomProjections
from randomSums import RandomSums
from rowSampler import RowSampler
from svdEmbedding import SVDEmbedding
from log_transforms import log_ppr, log_ppr_maxone, log_ppr_plusone

from scipy.io import loadmat
from tqdm import tqdm
try:
    from telepyth import TelepythClient
except ImportError:
    from utils import LazyClass as TelepythClient

TPC = TelepythClient()
DATASETS = ['POS', 'blogcatalog', 'Homo_sapiens', 'flickr', 'academic_coa_2014', 'academic_confs', 'vk2016']
SKETCHERS = {'fd': FrequentDirections, 'rp': RandomProjections, 's': RowSampler, 'h': RandomSums, 'svd': SVDEmbedding}
LOG_TRANSFORMS = {'log': log_ppr, 'add': log_ppr_plusone, 'max': log_ppr_maxone}

@click.command()
@click.argument('method', type=click.Choice(SKETCHERS.keys()))
@click.argument('dataset', type=click.Choice(DATASETS))
@click.argument('log', type=click.Choice(LOG_TRANSFORMS))
@click.argument('d', type=int)
@click.argument('datapath', type=click.Path(exists=True))
@click.option('--random_seed', type=int, default=-1, help='Random seed. If -1, use random one.')
@click.option('--random_order', type=bool, default=True, help='If True, shuffle rows before feeding them to a sketch (default: True).')
@click.option('--rotate', type=bool, default=True, help='Rotate (default: True).')
@click.option('--take_root', type=bool, default=True, help='Take root (default: True).')
@click.option('--left_vectors', type=bool, default=False, help='Which vectors to take if sketcher is SVD (default: True).')
@click.option('--algo', type=str, default='full', help='Which algo to use for SVD (default: full).')
@click.option('--ctrl_threads', type=bool, default=True, help='Whether or not to control the number of used CPU cores (default: True).')
@click.option('--ncores', type=int, default=1, help='The number of CPU cores to use (default: 1).')
def main(method, dataset, log, d, datapath, random_seed, random_order, rotate, take_root, left_vectors, algo, ctrl_threads, ncores):
    TPC.send_text(f'Calculating embeddings for {dataset} with method {method}')
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    
    root.info(f'Calculating embeddings for {dataset} with method {method} rotate_{rotate} takeroot_{take_root}')
    root.info(f'd: {d}, random seed: {random_seed}')
    if method == 'svd':
        root.info(f'using left vectors: {left_vectors}, algorithm: {algo}')

    filename = f'/d_{d}_log_{log}_seed_{random_seed}_rotate_{rotate}_takeroot_{take_root}'
    if method == 'svd':
        filename = filename + f'_leftvectors_{left_vectors}_algo_{algo}'
    directory = f'results/{method}/{dataset}'

    if os.path.exists(directory+filename):
        root.info(f'File exists! Aborting')
        return
    
    # random seed
    if random_seed != -1:
        np.random.seed(random_seed)
        
    # control the number of cores to be used
    if ctrl_threads:
        mkl_rt = ctypes.CDLL('libmkl_rt.so')
        mkl_get_max_threads = mkl_rt.mkl_get_max_threads
        def mkl_set_num_threads(cores):
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

        mkl_set_num_threads(ncores)
        root.info(f'CPUs used: {mkl_get_max_threads()}')

    # load dataset matfile
    matf = loadmat(datapath + f'/{dataset}.mat')
    G = matf['network']
    n = G.shape[0]
    if d > n:
        root.error('d is too high for the given dataset (n={n}).')
        raise Exception('d is too high for the given dataset (n={n}).')

    # get precomputed PPR for the dataset
    PPR = np.fromfile(datapath + f'/ppr/{dataset}.bin', dtype=np.float32).reshape(n, n) # n,n catches errors when the matrix is of unexpected size
    log_transformer = LOG_TRANSFORMS[log]
    log_transformer(PPR, n)

    # reorder rows
    ordering = np.arange(n)
    if random_order:
        np.random.shuffle(ordering)

    # compute a sketch
    if method == 'svd':
        sketcher = SKETCHERS[method](n, d, left_vectors, algo)
    else:
        sketcher = SKETCHERS[method](n, d)
    if method == 'svd':
        sketcher.compute(PPR)
    else:
        for i in tqdm(range(n)):
            sketcher.append(PPR[ordering[i], :])

    # get embeddings
    if method == 'fd':
        embs = sketcher.get(rotate=rotate, take_root=take_root)
    elif method == 'svd':
        embs = sketcher.get(take_root=take_root)
    else:
        embs = sketcher.get()

    if take_root and rotate:
        [_, s, Vt] = np.linalg.svd(embs, full_matrices=False)
        embs = np.diag(np.sqrt(s)) @ Vt

    # save embeddings
    if not os.path.exists(directory):
        os.makedirs(directory)
    root.info(embs.shape)
    embs.T.tofile(directory + filename) # return n x d matrix
    logging.info('Embeddings saved.')
    TPC.send_text('Embeddings calculated and saved.')


if __name__ == '__main__':
    main()
