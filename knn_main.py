#!usr/bin/python3
"""
this script allows user to train and evaluate the performance of Direct-CS-kNN
"""
# Author: Dec4f
# License: GPLv3

import sys
import os
from copy import deepcopy
import numpy as np
import mldata # ignored private C.45 parser
from cs_knn import DirectCSkNN
from eval_tools import *
from split_tools import KFoldCV

N_FOLD = 5
COST_MATRIX = np.array([[0.0, 0.25],
                        [0.6, 0.0]])
N_NEIGHBORS = 5
BASE_CONFIG = [COST_MATRIX, N_NEIGHBORS, 'euclidean']
SMOOTHING_CONFIG = [COST_MATRIX, N_NEIGHBORS, 'euclidean', 100, 0.8]

def main():
    """
    run the naive bayes network with given command line input
    ----------
    """
    file_path, use_full_sample, opt_method = sys.argv[1:4]
    examples = get_dataset(file_path)
    clf = parse_enhance_method(opt_method)
    if int(use_full_sample):
        samples = examples[:, 1:-1]
        targets = examples[:, -1]
        clf.fit(samples, targets)
    else:
        metrics_list = cv_eval(clf, examples)
        m_acc, m_prec, m_rec, m_spec, m_cost = [np.mean(metri) for metri in metrics_list]
        std_acc, std_prec, std_rec, std_spec, std_cost = [np.std(metri) for metri in metrics_list]
        print(("Mean Accuracy: %.3f ± %.3f " + os.linesep +
               "Precision: %.3f ± %.3f " + os.linesep +
               "Recall: %.3f ± %.3f" + os.linesep +
               "Specificity: %.3f ± %.3f" + os.linesep +
               "Misclassification Cost: %.3f ± %.3f") %
              (m_acc, std_acc,
               m_prec, std_prec,
               m_rec, std_rec,
               m_spec, std_spec,
               m_cost, std_cost))

def get_dataset(file_path):
    """
    parse the dataset stored in the input file path
    ----------
    file_path : String
        the path to the dataset
    """
    raw_parsed = mldata.parse_c45(file_path.split(os.sep)[-1], file_path)
    return np.array(raw_parsed, dtype=object)

def parse_enhance_method(opt_method):
    """
    parses user input and returns corresponding enhancement methods
    ----------
    opt_method : String
        the optimization method of choice
    """
    methods = ['base', 'smooth', 'stack']
    if opt_method not in methods:
        raise Exception("Error Code: OPT_METHOD_UNAVAILABLE")
    elif opt_method == methods[0]:
        print("Learning with Base Model...")
        return DirectCSkNN(*BASE_CONFIG)
    # using m-smoothing
    elif opt_method == methods[1]:
        print("Learning with M-smoothing...")
        return DirectCSkNN(*SMOOTHING_CONFIG)
    else:
        raise Exception("Error Code: UNKNOWN_IO_ERROR")

def cv_eval(clf, examples):
    """
    perform k fold cross validation and get metric data from each fold
    """
    [acc, prec, rec, spec, cost] = [np.zeros(N_FOLD, dtype=float) for i in range(5)]

    kcv = KFoldCV(N_FOLD, shuffle=True, seed=12345)
    for i, (train_idx, test_idx) in enumerate(kcv.split(examples)):
        train_set = examples[train_idx]
        test_set = examples[test_idx]
        x_train = train_set[:, 1:-1]
        y_train = train_set[:, -1]
        x_test = test_set[:, 1:-1]
        y_test = test_set[:, -1]
        model = deepcopy(clf)
        model.fit(x_train, y_train)

        y_pred = np.zeros(len(y_test), dtype=bool)
        cost_list = np.zeros(len(y_test), dtype=float)
        for j, row_test in enumerate(x_test):
            y_pred[j], cost_list[j] = model.predict(row_test)
        acc[i] = accuracy(y_test, y_pred)
        prec[i] = precision(y_test, y_pred)
        rec[i] = recall(y_test, y_pred)
        spec[i] = specificity(y_test, y_pred)
        cost[i] = np.mean(cost_list)

    print("accuracy = ", acc)
    print("precision = ", prec)
    print("recll = ", rec)
    print("specificity = ", spec)
    print("misclassification cost = ", cost)

    return [acc, prec, rec, spec, cost]

if __name__ == '__main__':
    main()
