import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def explain_model_shap(model, X_train, K=100):
    X_train_sampled = shap.sample(X_train, K)

    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X_train_sampled, feature_perturbation="interventional")
    elif isinstance(model, KNeighborsClassifier):
        explainer = shap.KernelExplainer(model.predict_proba, X_train_sampled)
    else:
        explainer = shap.TreeExplainer(model, X_train_sampled)

    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_train)
    shap.dependence_plot(0, shap_values, X_train)
