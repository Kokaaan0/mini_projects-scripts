import numpy as np

class Kmeans:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def initialize_centroids(self, X):
        # Randomly initialize centroids by selecting k data points
        indices = np.random.choice(len(X), size=self.n_clusters, replace=False)
        return X[indices]

    def assign_clusters(self, X):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        # Update centroids based on the mean of data points in each cluster
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def fit(self, X):
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid
            labels = self.assign_clusters(X)

            # Update centroids
            new_centroids = self.update_centroids(X, labels)

            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        return labels