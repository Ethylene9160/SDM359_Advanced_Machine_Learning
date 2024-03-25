import numpy as np

# import py.model.model as m

# This is a simple implementation of KMeans algorithm,
# Specially for RBF model, so this is a bit different from that in Professor Lin's lecture
class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.clusters = None

    def train(self, x_train, lr = 0.075, epochs = 100):
        self._get_clusters(x_train)
        for epoch in range(epochs):
            # we slightly move the center to the x, rather than calculate the centroid.
            for i in range(len(x_train)):
                x = x_train[i]
                cluster = self._get_nearest_cluster(x)
                self.clusters[cluster] = self.clusters[cluster] + lr * (x - self.clusters[cluster])

    def predict(self, x):
        return self._get_nearest_cluster(x)
    
    def _get_clusters(self, x_train, n_clusters = None):
        if n_clusters is None:
            n_clusters = self.n_clusters
        self.clusters = x_train[np.random.choice(len(x_train), n_clusters, replace=False)]

    # find which cluster is the nearest to x,
    # return the index of the nearest cluster
    def _get_nearest_cluster(self, x):
        return np.argmin(np.linalg.norm(self.clusters - x, axis=1))