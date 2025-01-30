import numpy as np
from sklearn.model_selection import cross_val_score


def find_best_n_iter(X_selected_features, rkf, y, classifiers):
    n_iters_list = [100, 500, 1000, 5000]

    results = {}

    for name, X_selected in X_selected_features.items():
        print(f"Finding best n_iter for {name}...")
        results[name] = {}

        for n_iters in n_iters_list:
            clf = classifiers[name]
            if hasattr(clf, 'max_iter'):
                clf.set_params(max_iter=n_iters)

            acc_scores = cross_val_score(clf, X_selected, y, cv=rkf, scoring='accuracy')
            results[name][n_iters] = acc_scores.mean()

    np.savez('number_of_iterations_results.npz',
             **{name: {str(k): v for k, v in iter_results.items()} for name, iter_results in results.items()})

    print("Results for different number of iterations:")
    for model, iter_results in results.items():
        print(f"  {model}:")
        for n_iter, acc in iter_results.items():
            print(f"    {n_iter} iterations: {acc:.4f} accuracy")

    return results
