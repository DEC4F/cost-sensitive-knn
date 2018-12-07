#!usr/bin/python3
"""
this module contains implementation for cost-sensitive k nearest neighbors algorithm proposed in:
Qin, Zhenxing, et al. "Cost-sensitive classification with k-nearest neighbors." International Conference on Knowledge Science, Engineering and Management. Springer, Berlin, Heidelberg, 2013.
----------
a brief summary of the paper is in the .org file, which will be converted to README soon.
"""
# Author: Decaf
# License: GPLv3

__all__ = ['DirectCSkNN']

import numpy as np

class DirectCSkNN():
    """
    Direct-CS-kNN is a KNN estimator implemented with cost-sensitive feature (sensitive to misclassification matrix).
    """
    def __init__(self, cost_mat,
                 n_neighbors=5,
                 metric='euclidean',
                 m_est=None,
                 base_rate=None):
        self.cost_mat = cost_mat
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.m_est = m_est
        self.base_rate = base_rate
        self.label_encoder_dict = {}
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        stores input training example in memory for prediction
        """
        self.X = np.zeros(X.shape)
        for i, attr in enumerate(X.T):
            if isinstance(attr[0], str):
                self.X[:, i] = self.label_encode(i, attr)
            else:
                self.X[:, i] = attr
        self.y = np.array(y, dtype=bool)

    def predict(self, new_x):
        """
        predict label for new example that minimizes the loss
        """
        processed_x = []
        for i, val in enumerate(new_x):
            if isinstance(val, str):
                if val not in self.label_encoder_dict[i].keys():
                    self.label_encoder_dict[i][val] = len(self.label_encoder_dict)
                processed_x.append(self.label_encoder_dict[i][val])
            else:
                processed_x.append(val)
        k_neigh_idx = self.find_k_neighbors(np.array(processed_x))
        k_labels = self.y[k_neigh_idx]
        loss = self.compute_loss(k_labels)
        return min(loss, key=loss.get), min(loss.values())

    def compute_loss(self, k_labels):
        """
        computes the loss by mult prob estimates with cost matrix
        """
        def prob_est(k_i):
            """
            calculates the probability estimates
            """
            # using base cond prob
            if self.m_est is None or self.base_rate is None:
                return k_i / self.n_neighbors
            # using m-smoothing
            return (k_i + self.m_est*self.base_rate) / (self.n_neighbors + self.m_est)

        loss = {}
        for i, i_label in enumerate(np.unique(self.y)):
            loss[i_label] = 0
            for j, j_label in enumerate(np.unique(self.y)):
                k_j = len(k_labels[k_labels == j_label])
                prob_j = prob_est(k_j)
                loss[i_label] += prob_j * self.cost_mat[i, j]
        return loss

    def label_encode(self, i, attr):
        """
        assign an integer value to each unique string value
        """
        label_dict = {val : idx for idx, val in enumerate(np.unique(attr))}
        self.label_encoder_dict[i] = label_dict
        return np.array([label_dict[val] for val in attr])

    def find_k_neighbors(self, new_x):
        """
        calculates the distance bewteen new example and all labeled data and returns k closest
        """
        # compute distance between new x to labeled example
        dist_dict = {}
        for i, labeled_x in enumerate(self.X):
            dist_dict[i] = self.dist(new_x, labeled_x)
        # flag k neighbors
        return sorted(dist_dict, key=dist_dict.get, reverse=True)[:self.n_neighbors]

    def dist(self, a, b):
        """
        calculates the distance between two examples with three distance functions
        ----------
        """
        dist_func_dict = {
            'euclidean' : np.sqrt(sum((a - b)**2)),
            'manhattan' : sum(abs(a - b)),
            'chebyshev' : max(abs(a - b))
        }
        return dist_func_dict[self.metric]
