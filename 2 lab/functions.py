import numpy as np
from collections import Counter

# -----------------------------------------------------------
# Simple Sigmoid function
# -----------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -----------------------------------------------------------
# Calculation of inary cross entropy
# -----------------------------------------------------------
def compute_error(y, hypo):
    y_zero_loss = y.T.dot(np.log(hypo))
    y_one_loss = (1 - y).T.dot(np.log(1 - hypo))
    return (1 / y.size ) * (y_zero_loss + y_one_loss)

# -----------------------------------------------------------
# Logistic regression calculator class
# alpha - gradient discent step koefficient (speed parameter)
# max_iteration_num - maximum number of iterrations
# x - matrix of parameters
# y - goal vector, the size must match row number of x
# -----------------------------------------------------------
class LogisticRegressionCalculator:
    def __init__(self, alpha = 0.01, max_iteration_num = 1000):
        self.alpha = alpha
        self.iter_num = max_iteration_num
        self.weights = None
        
    def fit(self, x, y):
        norm_coeff = 1 / y.size
        w = np.zeros(x.shape[1])
        for i in range(self.iter_num):
            curr_ans = x.dot(w)
            hypo = sigmoid(curr_ans)
            error = compute_error(y, hypo)
            grad = (1 / y.size) * (x.T.dot(hypo - y))
            w += self.alpha * grad * error
        self.weights = w

    def predict_yes_no(self, x):
        pred_ans = x.dot(self.weights)
        return np.where(sigmoid(pred_ans) > 0.5, 1, 0)

    def predict_probability(self, x):
        pred_ans = x.dot(self.weights)
        return sigmoid(pred_ans)

# -----------------------------------------------------------
# k-Nearest Neighbors calculator class (not optimized)
# n_neighbors - nearest neighbors number
# weights - uniform or distance - use scaling or not
# x - matrix of parameters
# y - goal vector, the size must match row number of x
# -----------------------------------------------------------
class KNNCalculator:
    def __init__(self, n_neighbors=3, weights='uniform'):
        self.k = n_neighbors
        self.use_scaling = weights == 'distance'

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        if self.use_scaling:
            for indx in range(X.columns.size):
                col = X.iloc[:, indx]
                if max(col) == min(col):
                    self.weights[indx] = 1
                else:
                    self.weights[indx] = 1 / (max(col) - min(col))
                X.iloc[:, indx] = X.iloc[:, indx] * self.weights[indx]
        self.X = X.to_numpy()
        self.y = y.to_numpy()


    def predict(self, X_test):
        y_ans = np.zeros(X_test.shape[0])
        for indx in range(X_test.index.size):
            obj = X_test.iloc[indx].to_numpy()
            if self.use_scaling:
                dist = np.linalg.norm(self.X - self.weights * obj, axis=1)
            else:
                dist = np.linalg.norm(self.X - obj, axis=1)
            nearest_ids = dist.argsort()[:self.k]
            nearest_classes = self.y[nearest_ids]
            y_ans[indx] = Counter(nearest_classes).most_common(1)[0][0]
        return y_ans
        