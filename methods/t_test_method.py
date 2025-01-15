import numpy as np
from scipy.stats import ttest_rel


def t_test():
    cross_validation = np.load('cross_validation_scores.npy')
    alpha = 0.5
    X = cross_validation.reshape(-1, 5)
    n_classifiers = X.shape[1]

    t_matrix = np.zeros((n_classifiers, n_classifiers))
    p_matrix = np.zeros((n_classifiers, n_classifiers))
    b_matrix = np.zeros((n_classifiers, n_classifiers), dtype=bool)
    b2_matrix = np.zeros((n_classifiers, n_classifiers), dtype=bool)

    for i in range(n_classifiers):
        for j in range(i + 1, n_classifiers):
            t, p = ttest_rel(X[:, i], X[:, j])
            t_matrix[i, j] = t
            p_matrix[i, j] = p
            if np.mean(X[:, i]) > np.mean(X[:, j]):
                b_matrix[i, j] = True
            else:
                b_matrix[j, i] = True
            if p < alpha:
                b2_matrix[i, j] = True
                b2_matrix[j, i] = True

    final_matrix = b2_matrix * b_matrix
    print(final_matrix)

    # save t_test results
    np.save('t_test_results.npy', final_matrix)