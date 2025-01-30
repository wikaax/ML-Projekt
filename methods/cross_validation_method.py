import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, precision_score, recall_score, roc_auc_score


def perform_cross_val(classifiers, rkf, X, y):
    acc_scores = np.zeros(shape=[len(classifiers), rkf.get_n_splits()])
    precision_scores = []
    recall_scores = []
    roc_auc_scores = []
    conf_mats = {}
    y_pred_final = None
    y_test_final = None
    x_test = None
    x_train = None
    y_train = None

    for i, (train, test) in enumerate(rkf.split(X, y)):
        for j, (clf_name, clf) in enumerate(classifiers.items()):
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            acc_scores[j, i] = accuracy_score(y[test], y_pred)

            if clf_name not in conf_mats:
                conf_mats[clf_name] = []
            conf_mats[clf_name].append(confusion_matrix(y[test], y_pred))

            precision_0 = precision_score(y[test], y_pred, pos_label=0)
            precision_1 = precision_score(y[test], y_pred, pos_label=1)
            recall_0 = recall_score(y[test], y_pred, pos_label=0)
            recall_1 = recall_score(y[test], y_pred, pos_label=1)
            precision_scores.append((precision_0, precision_1))
            recall_scores.append((recall_0, recall_1))

            fpr, tpr, _ = roc_curve(y[test], clf.predict_proba(X[test])[:, 1])
            auc = roc_auc_score(y[test], clf.predict_proba(X[test])[:, 1])
            roc_auc_scores.append(auc)

            if j == 0:
                y_pred_final = y_pred
                y_test_final = y[test]
                x_test = X[test]
                x_train = X[train]
                y_train = y[train]

    np.save('cross_validation_scores.npy', acc_scores)

    return acc_scores, conf_mats, y_pred_final, y_test_final, x_test, x_train, y_train, precision_scores, recall_scores, roc_auc_scores
