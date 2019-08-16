import numpy as np


def log_ppr(ppr, n: int):
    np.multiply(ppr, n, ppr)  # PPR = PPR * n
    np.log(ppr, ppr)         # PPR = np.log(PPR)


def log_ppr_plusone(ppr, n: int):
    np.multiply(ppr, n, ppr)  # PPR = PPR * n
    np.add(ppr, 1, ppr)      # PPR = PPR + 1
    np.log(ppr, ppr)         # PPR = np.log(PPR)


def log_ppr_maxone(ppr, n: int):
    np.multiply(ppr, n, ppr)  # PPR = PPR * n
    np.maximum(ppr, 1, ppr)  # PPR = max(PPR, 1)
    np.log(ppr, ppr)         # PPR = np.log(PPR)
