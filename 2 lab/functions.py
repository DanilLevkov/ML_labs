import numpy as np

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
    return -np.mean(y_zero_loss + y_one_loss)

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
            grad = np.mean(x.T.dot(hypo - y))
            w -= self.alpha * grad * error
        self.weights = w

    def predict_yes_no(self, x):
        pred_ans = x.dot(self.w)
        return np.where(sigmoid(pred_ans) > 0.5, 1, 0)

    def predict_probability(self, x):
        pred_ans = x.dot(self.w)
        return sigmoid(pred_ans)