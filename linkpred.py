import numpy as np
import ctypes
import click
import logging
import os
import sys
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
sys.path.append('src/')

from linkprediction import linkpred_verse
from utils import ddict2dict

from scipy.io import loadmat
from tqdm import tqdm
try:
    from telepyth import TelepythClient
except ImportError:
    from utils import LazyClass as TelepythClient

TPC = TelepythClient()
DATASETS = {'academic_coa_2014': 'academic_coa_full', 'vk2016': 'vk2017'}

@click.command()
@click.argument('dataset', type=click.Choice(DATASETS))
@click.argument('embedding', type=click.Path(exists=True, dir_okay=False))
@click.argument('datapath', type=click.Path(exists=True, file_okay=False))
@click.option('--random_seed', type=int, default=-1, help='Random seed. If -1, use random one.')
@click.option('--cut_d', type=int, default=0, help='Cut embeddings at dimension d. If 0, do not cut (default: 0).')
@click.option('--repeats', type=int, default=100, help='Number of restarts for result averaging (default: 100).')
@click.option('--ctrl_threads', type=bool, default=True, help='Whether or not to control the number of used CPU cores (default: True).')
@click.option('--ncores', type=int, default=1, help='The number of CPU cores to use (default: 1).')
def main(dataset, embedding, datapath, random_seed, cut_d, repeats, ctrl_threads, ncores):
    TPC.send_text(f'Starting evaluation for {embedding} on dataset {dataset}')
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    
    root.info(f'Calculating embeddings for {embedding} on dataset {dataset}')
    root.info(f'repeats: {repeats}, random seed: {random_seed}')
    if cut_d != 0:
        root.info(f'cut_d: {cut_d}')

    filepath = click.format_filename(embedding)
    filename = filepath[filepath.rfind('/'):]
    directory = f'linkpred/'+filepath[:filepath.rfind('/')]

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
    g_old = loadmat(f'{datapath}/{dataset}.mat')['network']
    g_new = loadmat(f'{datapath}/{DATASETS[dataset]}.mat')['network']
    n = g_old.shape[0]
    diff = g_new - g_old
    additions = np.array(list(zip(*(diff>0).nonzero())))
    na = additions.shape[0]

    # load embeddings
    embs = np.fromfile(embedding, np.float32).reshape(n, -1)
    if cut_d != 0:
        embs = embs[:, :cut_d]

    resdic = linkpred_verse(embs, additions, g_old, repeats)
        
    # save results
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(directory+filename, 'w') as outfile:
        json.dump(ddict2dict(resdic), outfile)

    logging.info('Results saved.')
    TPC.send_text('Results saved.')


if __name__ == '__main__':
    main()
