#!usr/bin/python3
"""
this module contains implementation of cost-sensitive stacking method described in:
Bahnsen, Alejandro Correa, Djamila Aouada, and Bjorn Ottersten. "Ensemble of example-dependent cost-sensitive decision trees." arXiv preprint arXiv:1505.04637 (2015).
"""
# Author: Dec4f
# License: GPLv3

__all__ = ['BaseStacking']

import numpy as np
from split_tools import KFoldCV

class BaseStacking():
    """Base stacking model
    Stacking is an estimator that combines different base classifiers by learning a second level algorithm on top of them.
    Each base classifier are built on training set S to output a prediction, and the output prediction is used as features for the second level learner to train on.
    Parameters
    ----------
    clfs : {array-like}
        a list of base cost-sensitive classifiers
    n_fold : int, default=5
        number of folds to use in internal cross validation
    Examples
    --------
    """

    def __init__(self, n_fold=5):
        """
        init a stacking model
        ----------
        n_fold : fold number for cross validation
        """
        self.n_fold = n_fold
        self.train_set = None
        self.test_set = None

    def fit(self, train_set, test_set, clfs):
        """
        fit the stacking model to data
        ----------
        train_set : Array-like
            the training set examples
        test_set : Array-like
            the testing set examples
        clfs : List
            a list of base classifiers
        """
        stk_train_set = np.zeros(len(clfs), (len(train_set))) # shape = (n_clf, n_trainset)
        stk_test_set = np.zeros(len(clfs), len(test_set)) # shape = (n_clf, n_testset)
        for i, clf in enumerate(clfs):
            stk_train_set[i], stk_test_set[i] = self.stacking_cv(clf, train_set, test_set)
        self.train_set = stk_train_set.T # shape = (n_test, n_clf)
        self.test_set = stk_test_set.T # shape = (n_test, n_clf)

    def predict(self, _x):
        pass

    def stacking_cv(self, base_clf, train_set, test_set):
        """
        out-of-folder prediction
        ----------
        for each classifier, train them on trainset of trainset, and make prediction on testset of trainset AND the original testset.
        the output of first prediction is vertically stacked together over folds to be get a new feature column in stacking trainset, and the output of second prediction is taken average over folds to be new feature column in stacking test set
        """
        oof_train = np.zeros((len(train_set), 1)) # shape = (n_train, 1)
        oof_test_kf = np.zeros((self.n_fold, len(test_set))) # shape = (n_fold, n_test)
        oof_test = np.zeros((len(test_set), 1)) # shape = (n_test, 1)

        kcv = KFoldCV(self.n_fold, shuffle=True, seed=12345)

        for i, (train_idx, test_idx) in enumerate(kcv.split(train_set)):
            # do train-test-split within input training set by spliting it into lower level training and testing sets
            x_train_ll = train_set[train_idx][:, 1:-1] # shape = (n_train_train, n_attr)
            y_train_ll = train_set[train_idx][:, -1] # shape = (n_train_train, 1)
            x_test_ll = test_set[test_idx][:, 1:-1] # shape = (n_test_train, n_attr)

            base_clf.fit(x_train_ll, y_train_ll)
            oof_train[test_idx, 1] = np.array([base_clf.predict(row) for row in x_test_ll]) # shape = (n_test_train, 1)
            oof_test_kf[i, :] = np.array([base_clf.predict(row) for row in test_set]) # shape = (n_fold, n_test)
        oof_test = np.mean(oof_test_kf, axis=0) # shape = (1, n_test)
        return oof_train, oof_test
