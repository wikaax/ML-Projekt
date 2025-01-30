import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def explain_model_shap(model, X_train, feature_names=None, num_samples=100):
    X_sampled = shap.sample(X_train, num_samples)

    if isinstance(model, (LogisticRegression, KNeighborsClassifier)):
        explainer = shap.KernelExplainer(model.predict_proba, X_sampled)
        shap_values = explainer.shap_values(X_sampled)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sampled)

    print(f"shap_values shape: {np.array(shap_values).shape}")
    print(f"X_sampled shape: {X_sampled.shape}")

    if isinstance(shap_values, list):
        for i, class_shap_values in enumerate(shap_values):
            print(f"Wyświetlanie wyników dla klasy {i}")
            shap.summary_plot(class_shap_values, X_sampled, feature_names=feature_names)
    else:
        shap.summary_plot(shap_values, X_sampled, feature_names=feature_names)

    try:
        shap.dependence_plot(0, shap_values[0] if isinstance(shap_values, list) else shap_values, X_sampled, feature_names=feature_names)
    except Exception as e:
        print(f"Błąd podczas generowania dependence_plot: {e}")
