import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, classification_report


def confusion_matrix_and_classification_report(results):
    acc_scores, conf_matrices, *_ = results
    for clf_name, conf_list in conf_matrices.items():
        if conf_list:
            conf_mat = conf_list[0]
            plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix for {clf_name}")
            plt.colorbar()
            plt.show()
        else:
            print(f"Confusion matrix for {clf_name} is not available or invalid.")



def precision_and_recall(cross_val):
    precision_scores = cross_val[7]
    recall_scores = cross_val[8]

    avg_precision_0 = np.mean([score[0] for score in precision_scores])
    avg_precision_1 = np.mean([score[1] for score in precision_scores])
    avg_recall_0 = np.mean([score[0] for score in recall_scores])
    avg_recall_1 = np.mean([score[1] for score in recall_scores])

    print(f'AVG_PRECISION_0: {avg_precision_0:.2f}')
    print(f'AVG_PRECISION_1: {avg_precision_1:.2f}')
    print(f'AVG_RECALL_0: {avg_recall_0:.2f}')
    print(f'AVG_RECALL_1: {avg_recall_1:.2f}')

    return avg_precision_0, avg_recall_0, avg_precision_1, avg_recall_1


def roc_curve_plot(cross_val):
    y_test = cross_val[3]
    y_pred = cross_val[2]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Reference line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()


def mean_scores(classifiers, cross_val):
    acc_scores = cross_val[0]
    mean_scores = np.mean(acc_scores, axis=1)
    std_scores = np.std(acc_scores, axis=1)

    for clf_id, clf_name in enumerate(classifiers):
        print(f"{clf_name}: Mean score: {mean_scores[clf_id]:.3f} \tStd: {std_scores[clf_id]:.3f}")



def feature_selection_charts(lr):
    try:
        selected_features = np.load('selected_features.npy', allow_pickle=True)
        print(selected_features)
    except ValueError as e:
        print(f"Error loading file: {e}")

    # feature importance
    feature_weights = lr.coef_[0]
    feature_importance = np.abs(feature_weights)
    features = ['Pclass', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Ticket_110152', 'Ticket_113760', 'Ticket_13502',
                'Ticket_24160', 'Ticket_2666', 'Ticket_29106', 'Ticket_347742', 'Ticket_CA. 2343', 'Ticket_PC 17572',
                'Ticket_PC 17755', 'Cabin_B96 B98', 'Cabin_E101', 'Cabin_F33', 'Embarked_C', 'Embarked_S']
    importance = [0.016031662428189018, 0.0037305908758575074, 0.01203896354791156, 0.02567313154935162,
                  0.02567313154935162, 0.003468582182947117, 0.003987848670741845, 0.0034700750345102696,
                  0.0034445361913083387, 0.004016649766817327, 0.003540189598150869, 0.003530647252465814,
                  0.0033744402951988006, 0.0034955961743442206, 0.003418423154435658, 0.003987848670741845,
                  0.003491489266030324, 0.0034955100005501258, 0.007862900947430819, 0.007280306582351379]
    # bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(features, importance)
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

    # pie chart with color map
    colors = plt.cm.tab20b(range(len(features)))
    plt.figure(figsize=(8, 8))
    plt.pie(importance, labels=features, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.axis('equal')
    plt.title('Ważność cech')
    plt.show()


def histograms(data):
    # Pclass histogram
    grouped_data = data.groupby(['Pclass', 'Survived']).size().unstack()
    fig, ax = plt.subplots()
    bar_width = 0.4
    index = np.arange(len(grouped_data.index))
    rects1 = ax.bar(index, grouped_data[0], bar_width, label="Didn't survive")
    rects2 = ax.bar(index + bar_width, grouped_data[1], bar_width, label='Survived')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(grouped_data.index)
    ax.set_ylabel('Number of people')
    ax.set_title('Dependence of survival on passenger class (Pclass)')
    ax.legend()
    plt.show()

    # Sex histogram
    grouped_data = data.groupby(['Sex', 'Survived']).size().unstack()
    fig, ax = plt.subplots()
    bar_width = 0.4
    index = np.arange(len(grouped_data.index))
    rects1 = ax.bar(index, grouped_data[0], bar_width, label="Didn't survive")
    rects2 = ax.bar(index + bar_width, grouped_data[1], bar_width, label='Survived')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(grouped_data.index)
    ax.set_ylabel('Number of people')
    ax.set_title('Dependence of survival on passenger sex')
    ax.legend()
    plt.show()


def iter_experiment_results():
    n_iters_experiment = np.load('number_of_iterations_results.npz')

    n_iters_keys = n_iters_experiment
    for key in n_iters_keys:
        print(f"Results for {key}: %.3f" % n_iters_experiment[key])


def scatter(cross_val):
    x_test = cross_val[4]
    y_pred = cross_val[2]

    plt.figure(figsize=(6, 6))
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred, cmap='viridis')
    plt.title(f'Scatter Plot')
    plt.xlabel('Feature 1')
    plt.ylabel('Surivived')
    plt.show()


def feature_reduction_and_scatter_plot(cross_val, lr, resolution=0.02):
    y_pred = cross_val[2]
    y_test = cross_val[3]
    x_train = cross_val[5]
    y = cross_val[6]
    x_test = cross_val[4]

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(x_train)

    markers = ('s', 'o')
    colors = ('blue', 'red')

    plt.figure(figsize=(8, 6))
    for idx, label in enumerate([y_test, y_pred]):
        plt.scatter(X_tsne[:len(label), 0], X_tsne[:len(label), 1], c=label, cmap=ListedColormap(colors[idx]),
                    marker=markers[idx], alpha=0.6)

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(['True', 'Predicted'])
    plt.show()

    pca = PCA(n_components=2)
    # fit and transform data
    X = pca.fit_transform(x_train)
    X_test_pca = pca.transform(x_test)
    lr.fit(X, y)

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=[cmap(idx)],
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()

    # Dimension reduction using PCA
    # pca = PCA(n_components=2)
    # X_reduced = pca.fit_transform(x_test)
    #
    # # Standardize the reduced features for better visualization
    # scaler = StandardScaler()
    # X_reduced = scaler.fit_transform(X_reduced)
    #
    # # Set up marker generator and color map
    # markers = ('s', 'x', 'o', '^', 'v')
    # colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # cmap = ListedColormap(colors[:len(np.unique(y_test))])
    # print(X_reduced)

    # Plot scatter plot
    # plt.scatter(X_reduced[y_test == 0, 0], X_reduced[y_test == 0, 1], c='red', marker='o', label='Actual Labels 0')
    # plt.scatter(X_reduced[y_test == 1, 0], X_reduced[y_test == 1, 1], c='red', marker='o', label='Actual Labels 1')
    # plt.scatter(X_reduced[y_pred == 0, 0], X_reduced[y_pred == 0, 1], c='blue', marker='s', label='Predicted Labels 0')
    # plt.scatter(X_reduced[y_pred == 1, 0], X_reduced[y_pred == 1, 1], c='blue', marker='s', label='Predicted Labels 1')
    #
    # plt.xlabel('PCA1')
    # plt.ylabel('PCA2')
    # plt.legend(loc='best')
    # plt.title('Scatter Plot of Reduced Features')
    #
    # plt.show()
    # y_pred = cross_val[2]
    # y_test = cross_val[3]
    #
    #
    # # setup marker generator and color m
    # markers = ('s', 'x', 'o', '^', 'v')
    # colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # cmap = ListedColormap(colors[:len(np.unique(y_pred))])
    #
    # # dimension reduction using PCA
    # pca = PCA(n_components=2)
    # X_reduced = pca.fit_transform(X_selected)

    # # feature weights for pc1
    # feature_weights_pc1 = pca.components_[0]
    #
    # # feature weights for pc2
    # feature_weights_pc2 = pca.components_[1]
    #
    # # pc1 feature importace
    # top_features_pc1 = np.argsort(np.abs(feature_weights_pc1))[::-1][:5]
    # print("Top features for PC1:")
    # for feature_idx in top_features_pc1:
    #     print(col_names_encoded[feature_idx])
    #
    # # pc2 feature importace
    # top_features_pc2 = np.argsort(np.abs(feature_weights_pc2))[::-1][:5]
    # print("Top features for PC2:")
    # for feature_idx in top_features_pc2:
    #     print(col_names_encoded[feature_idx])

    # plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.title('Scatter Plot after PCA')
    # plt.show()
