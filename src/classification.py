from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import lil_matrix
import numpy as np


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels


def evaluate_deepwalk(embedding, labels, number_shuffles=10, train_perc=0.1):
    micro = []
    macro = []
    for _ in range(number_shuffles):
        X_train, X_test, Y_train, Y_test = train_test_split(
            embedding, labels, test_size=1 - train_perc)
        clf = TopKRanker(LogisticRegression(solver='liblinear'))
        clf.fit(X_train, Y_train)
        top_k_list = [l.nnz for l in Y_test]
        preds = clf.predict(X_test, top_k_list)
        preds_ = lil_matrix(Y_test.shape)
        for idx, pi in enumerate(preds):
            for pii in pi:
                preds_[idx, pii] = 1
        micro.append(f1_score(Y_test, preds_, average='micro'))
        macro.append(f1_score(Y_test, preds_, average='macro'))
    return (micro, macro)


def evaluate_verse(embedding, labels, number_shuffles=10, train_perc=0.1):
    from skmultilearn.problem_transform import LabelPowerset

    micro = []
    macro = []
    sss = StratifiedShuffleSplit(
        n_splits=number_shuffles,
        test_size=1 - train_perc)
    for train_index, test_index in sss.split(embedding, labels):
        X_train, X_test = embedding[train_index], embedding[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf = LabelPowerset(LogisticRegression())
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        micro.append(f1_score(y_test, preds, average='micro'))
        macro.append(f1_score(y_test, preds, average='macro'))
    return (micro, macro)


def evaluate_nonoverlapping(
        embedding,
        labels,
        number_shuffles=10,
        train_perc=0.1):
    labels = labels.nonzero()[1]
    micro = []
    macro = []
    sss = StratifiedShuffleSplit(
        n_splits=number_shuffles,
        test_size=1 - train_perc)
    for train_index, test_index in sss.split(embedding, labels):
        X_train, X_test = embedding[train_index], embedding[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        micro.append(f1_score(y_test, preds, average='micro'))
        macro.append(f1_score(y_test, preds, average='macro'))
    return (micro, macro)
