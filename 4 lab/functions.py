import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------------------------
# Help function
# -----------------------------------------------------------
def most_frequent_in_array(arr):
    tmp = np.array(arr)
    unique, counts = np.unique(tmp, return_counts=True)
    return unique[np.argmax(counts)]


# -----------------------------------------------------------
# Bagging calculator class
# bag_num - number of classifiers
# bag_size - number of samples
# classifier = classifier type
# -----------------------------------------------------------
class BaggingCalculator:
    def __init__(self, bag_num = 10, bag_size = 10, classifier = DecisionTreeClassifier):
        self.bag_num = bag_num
        self.bag_size = bag_size
        self.classifier = classifier

    def fit(self, x_train, y_train):
        results = []
        
        for i in range(self.bag_num):
            classifier = self.classifier()
            indexes = np.random.choice(self.bag_size, size = len(x_train))
            x_i = x_train[indexes]
            y_i = y_train[indexes]
            classifier.fit(x_i, y_i)
            results.append(classifier)
        
        self.predictors = results

    def predict(self, x_test):
        result_stack =  np.zeros((1, len(x_test)), dtype=int)
        for predictor in self.predictors:
            result_stack = np.vstack([result_stack, predictor.predict(x_test)])

        result_stack = np.delete(result_stack, 0, 0)
        result = np.zeros(len(x_test))
        for i in range(len(x_test)):
            # most frequent value
            unique, counts = np.unique(result_stack[:, i], return_counts=True)
            result[i] = unique[np.argmax(counts)]
        return result

# -----------------------------------------------------------
# AdaBoost calculator class
# bag_num - number of classifiers
# learning_koeff - learning koefficient
# classifier = classifier type
# -----------------------------------------------------------
class AdaBoostCalculator:
    def __init__(self, bag_num = 10, learning_koeff = 1, classifier = DecisionTreeClassifier):
        self.bag_num = bag_num
        self.learning_koeff = learning_koeff
        self.classifier = classifier

    def fit(self, x_train, y_train):
        results = []
        weights = np.zeros(self.bag_num)
        sample_weight = np.ones(len(x_train))
        sample_weight /= np.sum(sample_weight)
        
        for i in range(self.bag_num):
            classifier = self.classifier()
            classifier.fit(x_train, y_train, sample_weight = sample_weight)
            y_train_pred = classifier.predict(x_train)
            false_indexes = y_train_pred != y_train
            error = np.sum(sample_weight[false_indexes])
            new_weight = 1
            if error > 0:
                new_weight = self.learning_koeff * np.log((1 - error) / error)
            weights[i] = new_weight
            weight_indexes = ((new_weight < 0) | (sample_weight > 0))
            sample_weight *= np.exp(new_weight * false_indexes * weight_indexes)
            # norming
            sample_weight /= np.sum(sample_weight)
            results.append(classifier)

        self.predictors = results
        self.weights = weights

    def predict(self, x_test):
        result_stack =  np.zeros((1, len(x_test)), dtype=int)
        for predictor in self.predictors:
            result_stack = np.vstack([result_stack, predictor.predict(x_test)])
        result_stack = np.delete(result_stack, 0, 0)
        result = np.zeros(len(x_test))
        for i in range(len(x_test)):
            # most frequent value
            answers = result_stack[:, i]
            answers_with_weigths = []
            for j in range(answers.shape[0]):
                percent = np.round(self.weights[j] * 100)
                answers_with_weigths = np.append(answers_with_weigths, np.repeat(answers[j], percent))
            unique, counts = np.unique(answers_with_weigths, return_counts=True)
            result[i] = unique[np.argmax(counts)]
        return result