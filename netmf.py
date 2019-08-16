import numpy as np


def deepwalk_matrix(A, window, b):
    n = A.shape[0]
    vol = float(A.sum())
    d = np.array(1./A.sum(1).T)[0]
    D = sp.diags(d, 0)
    X = D.dot(A)
    X = X.toarray()
    S = X
    X_power = X
    if window > 1:
        for i in range(window-1):
            X_power = X_power.dot(X)
            S = S + X_power
    S *= vol / window / b
    M = S @ D
    return M


def netmf(A, window, b, d):
    M = deepwalk_matrix(A, window, b)
    M = np.maximum(M, 1)
    M = np.log(M)
    U, s, _ = np.linalg.svd(M)
    return U[:, :d] @ np.diag(np.sqrt(s[:d]))
