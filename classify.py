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

from classification import evaluate_deepwalk, evaluate_nonoverlapping, evaluate_verse
from utils import ddict2dict

from scipy.io import loadmat
from tqdm import tqdm
try:
    from telepyth import TelepythClient
except ImportError:
    from utils import LazyClass as TelepythClient

TPC = TelepythClient()
DATASETS = ['POS', 'blogcatalog', 'Homo_sapiens', 'flickr', 'academic_coa_2014', 'academic_confs', 'vk2016']
METHODS = {'deepwalk': evaluate_deepwalk, 'verse': evaluate_verse, 'nonoverlapping': evaluate_nonoverlapping}

@click.command()
@click.argument('dataset', type=click.Choice(DATASETS))
@click.argument('method', type=click.Choice(METHODS.keys()))
@click.argument('embedding', type=click.Path(exists=True, dir_okay=False))
@click.argument('range_start', type=float)
@click.argument('range_end', type=float)
@click.argument('range_increment', type=float)
@click.argument('datapath', type=click.Path(exists=True, file_okay=False))
@click.option('--random_seed', type=int, default=-1, help='Random seed. If -1, use random one.')
@click.option('--cut_d', type=int, default=0, help='Cut embeddings at dimension d. If 0, do not cut (default: 0).')
@click.option('--repeats', type=int, default=100, help='Number of restarts for result averaging (default: 100).')
@click.option('--ctrl_threads', type=bool, default=True, help='Whether or not to control the number of used CPU cores (default: True).')
@click.option('--ncores', type=int, default=1, help='The number of CPU cores to use (default: 1).')
def main(dataset, method, embedding, range_start, range_end, range_increment, datapath, random_seed, cut_d, repeats, ctrl_threads, ncores):
    TPC.send_text(f'Starting evaluation for {embedding} on dataset {dataset} with method {method}')
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    
    root.info(f'Calculating embeddings for {embedding} on dataset {dataset} with method {method}')
    root.info(f'repeats: {repeats}, random seed: {random_seed}')
    if cut_d != 0:
        root.info(f'cut_d: {cut_d}')

    filepath = click.format_filename(embedding)
    filename = filepath[filepath.rfind('/'):]
    directory = f'classification/'+filepath[:filepath.rfind('/')]

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
    G, labels = matf['network'], matf['group']
    n = G.shape[0]

    # load embeddings
    resdic = defaultdict(lambda: defaultdict(list))
    embs = np.fromfile(embedding, np.float32).reshape(n, -1)
    if cut_d != 0:
        embs = embs[:, :cut_d]
    for tpc in np.arange(range_start, range_end, range_increment):
        micros, macros = METHODS[method](embs, labels, repeats, tpc)
        resdic['micro'][int(round(100*tpc))] = micros
        resdic['macro'][int(round(100*tpc))] = macros
        logging.info(f'{filepath} {int(100*tpc)}%, {100*np.mean(micros):.2f}, {100*np.mean(macros):.2f}')

    # save results
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(directory+filename, 'w') as outfile:
        json.dump(ddict2dict(resdic), outfile)

    logging.info('Results saved.')
    TPC.send_text('Results saved.')


if __name__ == '__main__':
    main()
