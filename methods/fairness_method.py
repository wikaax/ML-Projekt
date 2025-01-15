from sklearn.metrics import confusion_matrix
import numpy as np

def check_demographic_parity(y_pred, sensitive_feature):
    groups = np.unique(sensitive_feature)
    for group in groups:
        group_mask = sensitive_feature == group
        group_pred = y_pred[group_mask]
        print(f"Demographic Parity for group {group}: {np.mean(group_pred):.2f}")

def check_equalized_odds(y_pred, y_test, sensitive_feature):
    groups = np.unique(sensitive_feature)
    for group in groups:
        group_mask = sensitive_feature == group
        group_y_pred = y_pred[group_mask]
        group_y_test = y_test[group_mask]
        tn, fp, fn, tp = confusion_matrix(group_y_test, group_y_pred, labels=[0, 1]).ravel()
        print(f"Equalized Odds for group {group}:")
        print(f"  False Positive Rate: {fp / (fp + tn):.2f}")
        print(f"  False Negative Rate: {fn / (fn + tp):.2f}")
