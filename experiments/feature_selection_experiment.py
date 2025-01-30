import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

from methods.cross_validation_method import perform_cross_val


def feature_selection(X, y, col_names, rkf, clf):
    n_features_list = [5, 10, 15, 20]

    results = {}
    best_n_feature = 0
    best_accuracy = 0

    for n_features in n_features_list:
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)

        acc_scores = perform_cross_val({'LR': clf}, rkf, X_selected, y)[0]

        results[n_features] = acc_scores.mean()

        if acc_scores.mean() > best_accuracy:
            best_n_feature = n_features
            best_accuracy = acc_scores.mean()

    selector = SelectKBest(score_func=f_classif, k=best_n_feature)
    X_selected = selector.fit_transform(X, y)

    selected_features = np.array(col_names)[selector.get_support()]

    np.save('selected_features.npy', selected_features)

    for n_features, accuracy in results.items():
        print(f"Number of Features: {n_features}, Accuracy: %.3f" % accuracy)
    print(selected_features)
    return X_selected
