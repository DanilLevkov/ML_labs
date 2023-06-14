import numpy as np

# -----------------------------------------------------------
# Euler distance between centers and data
# X - data points
# centers - list of choosen cluster's centers
# -----------------------------------------------------------
def k_distance(X, centers):
    distances = np.zeros((X.shape[0], centers.shape[0]))
    for i, center in enumerate(centers):
        distances[:, i] = np.linalg.norm(X - center, axis=1)
    return distances

# -----------------------------------------------------------
# K-means calculator class (not optimized)
# cluster_num - number of clusters
# max_iteration_num - iterations limit
# tolerance - accuracy limit
# -----------------------------------------------------------
class MyKMeans:
    def __init__(self, cluster_num, max_iteration_num=100, tolerance=1e-3):
        self.k = cluster_num
        self.max_iter = max_iteration_num
        self.tol = tolerance

    def recompute_centers(self, X):
        centers_length = X.shape[1]
        centers = np.zeros((self.k, centers_length))
        for i in range(self.k):
            centers[i] = np.mean(X[self.cluster_res == i], axis=0)
        return centers

    def fit(self, X):
        random_center_indx = np.random.choice(X.shape[0], self.k, replace=False)
        self.centers = X[random_center_indx]

        for iter in range(self.max_iter):
            distances = k_distance(X, self.centers)
            self.cluster_res = np.argmin(distances, axis=1)
            new_centers = self.recompute_centers(X)

            if np.linalg.norm(new_centers - self.centers) < self.tol:
                print("Accuracy is reached on iter: ", iter)
                break

            self.centers = new_centers

        return self

    def predict(self, X):
        return np.argmin(k_distance(X, self.centers), axis=1)