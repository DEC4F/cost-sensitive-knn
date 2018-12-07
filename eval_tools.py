#!usr/bin/python3
"""
This evaluation module contains implementations of several important metric functions
"""
# Author: Dec4f
# License: GPLv3

def accuracy(y_true, y_pred):
    """
    calculate the accuracy of prediction
    (TP + TN) / (TP + FP + TN + FN)
    ----------
    y_true : array-like
          true class labels
    y_pred : array-like
          predicted class labels
    """
    assert len(y_true) == len(y_pred)
    label_size = len(y_true)
    count = 0.0
    for i in range(label_size):
        if y_true[i] == y_pred[i]:
            count += 1.0
    return count / float(label_size)

def precision(y_true, y_pred):
    """
    calculates the precision score of this prediction
    TP / (TP + FP)
    ----------
    y_true : array-like
          true class labels
    y_pred : array-like
          predicted class labels
    """
    if sum(y_pred) == 0:
        return 1.0
    numer = 0.0
    for i, j in zip(y_true, y_pred):
        if i and j:
            numer += 1.0
    return numer / float(sum(y_pred))

def recall(y_true, y_pred):
    """
    calculates the recall score of this prediction
    TP / (TP + FN)
    ----------
    y_true : array-like
          true class labels
    y_pred : array-like
          predicted class labels
    """
    if sum(y_true) == 0:
        return 1.0
    numer = 0.0
    for i, j in zip(y_true, y_pred):
        if i and j:
            numer += 1.0
    return numer / float(sum(y_true))

def specificity(y_true, y_pred):
    """
    calculate the specificity of this prediction
    ----------
    y_true : array-like
          true class labels
    y_pred : array-like
          predicted class labels
    """
    n_false = y_true[y_true == False].size
    if n_false == 0:
        return 1.0
    numer = 0.0
    for i, j in zip(y_true, y_pred):
        if not (i or j):
            numer += 1.0
    return numer / float(n_false)
