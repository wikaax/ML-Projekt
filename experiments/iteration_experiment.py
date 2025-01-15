import numpy as np
from sklearn.model_selection import cross_val_score

from methods.cross_validation_method import perform_cross_val
from methods.logistic_regression_method import LogisticRegression


def find_best_n_iter(X, rkf, y):
    # list of n_iters to find the best
    n_iters_list = [100, 500, 1000, 5000]

    # storing results of n_iters experiment
    results = {}
    best_n_iter = None
    best_accuracy = 0

    for n_iters in n_iters_list:
        # create clf with current value of n_iters
        lr = LogisticRegression(lr=0.001, n_iters=n_iters)
        acc_scores = perform_cross_val({'LR': lr}, rkf, X, y)[0]
        results[n_iters] = acc_scores.mean()

        if results[n_iters] > best_accuracy:
            best_n_iter = n_iters
            best_accuracy = results[n_iters]

    # save results do file
    np.savez('number_of_iterations_results.npz', **{str(k): v for k, v in results.items()})
    return best_n_iter
