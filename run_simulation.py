import pandas as pd
import warnings
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
# from methods.explain_model_shap import explain_model_shap
from methods.fairness_method import check_demographic_parity, check_equalized_odds

from experiments.feature_selection_experiment import feature_selection
from experiments.iteration_experiment import find_best_n_iter
from methods.cross_validation_method import perform_cross_val
from methods.t_test_method import t_test
from methods.utils import (
    histograms,
    feature_selection_charts,
    precision_and_recall,
    confusion_matrix_and_classification_report,
    roc_curve_plot,
    mean_scores,
)

warnings.filterwarnings('ignore')

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(['Survived'], axis=1)
    y = data['Survived']

    # encode categorical variables
    X = pd.get_dummies(X, prefix_sep='_')
    col_names = X.columns  # Zapisanie nazw kolumn po one-hot encodingu

    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    return X, y, col_names, data


def run_simulation():
    X, y, col_names, data = preprocess_data('train.csv')

    lr = LogisticRegression()
    classifiers = {'LR': lr,
                   'kNN': KNeighborsClassifier(),
                   'DTC': DecisionTreeClassifier(),
                   'SVM': svm.SVC(probability=True),
                   'GNB': GaussianNB()
                   }

    n_splits, n_repeats = 2, 5
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    # feature selection
    print("Performing feature selection...")
    selected_features = {}
    for name, clf in classifiers.items():
        print(f"Feature selection for {name}...")
        X_selected = feature_selection(X, y, col_names, rkf, clf)
        selected_features[name] = X_selected

    # find best n_iter
    print("Finding the best number of iterations...")
    best_n_iter_results = find_best_n_iter(selected_features, rkf, y, classifiers)

    # cross-val
    print("Performing cross-validation...")
    cross_val_results = {name: perform_cross_val({name: clf}, rkf, X_selected, y) for name, clf in classifiers.items()}

    # fairness
    print("\nFairness Analysis:")
    sensitive_feature = data['Pclass']

    for name, clf in classifiers.items():
        print(f"\nFairness for {name}:")
        X_selected = selected_features[name]

        clf.fit(X_selected, y)
        y_pred = clf.predict(X_selected)

        check_demographic_parity(y_pred, sensitive_feature)
        check_equalized_odds(y_pred, y, sensitive_feature)

        # not working
        # explain_model_shap(clf, X_selected, feature_names=col_names)

    # results
    for name, results in cross_val_results.items():
        print(f"\nAnalysis for {name}:")
        precision_and_recall(results)
        mean_scores({name: classifiers[name]}, results)
        confusion_matrix_and_classification_report(results)
        roc_curve_plot(results)

    # global Analysis
    print("\nGlobal Analysis:")
    histograms(data)
    feature_selection_charts(classifiers['LR'])

    print("\nPerforming t-test for classifier comparisons...")
    t_test()


if __name__ == "__main__":
    run_simulation()
