import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

LINKPRED_OPS = {
    'l2': lambda x,y: np.square(x-y),
    'l1': lambda x,y: np.abs(x-y),
    'had': lambda x,y: x*y,
    'avg': lambda x,y: (x+y)/2,
    'cat': lambda x,y: np.hstack([x,y])    
}

def linkpred_verse(embs, additions, g_old, n_evals=10):
    n = g_old.shape[0]
    na = additions.shape[0]
    resdic = defaultdict(list)
    for opname, op in LINKPRED_OPS.items():
        for it in range(n_evals):
            neg_examples = np.random.randint(0, n, size=additions.shape)
            for i in range(na):
                if g_old[neg_examples[i,0], neg_examples[i,1]] >= 0:
                    neg_examples[i,:] = np.random.randint(0, n, size=2)
            X_lab = np.vstack((additions, neg_examples))
            Y = np.hstack((np.ones(na), np.zeros(na)))
            if opname != 'cat':
                X = np.zeros((na*2, embs.shape[1]))
            else:
                X = np.zeros((na*2, 2*embs.shape[1]))                    
            for i in range(na*2):
                X[i, :] = op(embs[X_lab[i,0]],embs[X_lab[i,1]])
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
            clf = LogisticRegression(solver='liblinear')
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            resdic[opname].extend([roc_auc_score(y_test, preds)])
    return resdic