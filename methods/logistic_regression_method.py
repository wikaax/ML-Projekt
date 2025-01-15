import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


# initialize sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, lr, n_iters):  # learningRate (0.1 -> 0.0001), number of iterations
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights as zero
        self.bias = 0  # Initialize bias as zero

        # prediction
        for _ in range(self.n_iters):
            linear_predictions = np.dot(X, self.weights) + self.bias  # linear regression
            predictions = sigmoid(linear_predictions)  # linear regression in sigmoid function

            # calculating the gradient
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))  # gradient for weights
            db = (1 / n_samples) * np.sum(predictions - y)  # gradient for bias

            # updating weights and bias
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_predictions = sigmoid(linear_predictions)
        return [0 if y <= 0.5 else 1 for y in y_predictions]  # Classification

    def predict_proba(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_predictions = sigmoid(linear_predictions)
        return np.array([[1 - y, y] for y in y_predictions])  # Returning probabilities for both classes

    def get_params(self, deep=True):
        return {'lr': self.lr, 'n_iters': self.n_iters}
