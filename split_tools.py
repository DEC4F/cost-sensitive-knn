#!usr/bin/python3
"""
this module contains classes and functions that split the data set into different sets.
"""
# Author: Dec4f
# License: GPLv3

import numpy as np

class KFoldCV():
    """
    the class for standard K-fold cross validations
    """

    def __init__(self, n_fold, shuffle=False, seed=None):
        self.n_fold = n_fold
        self.shuffle = shuffle
        self.seed = seed

    def split(self, X):
        """
        split the examples
        X : {array-like}
            the training examples
        """
        indices = np.arange(len(X))
        if self.shuffle and self.seed is not None:
            np.random.seed(self.seed)
            np.random.shuffle(X)
        for ith_fold in range(self.n_fold):
            mask = self.test_mask(X, ith_fold)
            train_idx = indices[np.logical_not(mask)]
            test_idx = indices[mask]
            yield train_idx, test_idx

    def test_mask(self, X, ith_fold):
        """
        generate a test mask to mark test set
        ----------
        X : {array-like}
            the training examples
        ith_fold : int
            current fold number
        """
        fold_size = len(X) // self.n_fold
        test_mask = np.zeros(len(X), dtype=bool)
        if ith_fold + 1 == self.n_fold:
            # set all remaining example as test set if in last fold
            test_mask[ith_fold * fold_size:] = True
        else:
            # otherwise set examples in current fold as test set
            test_mask[ith_fold * fold_size : (ith_fold+1) * fold_size] = True
        return test_mask
