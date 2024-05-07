import numpy as np

def _distance_square(x1, x2):
    n = len(x1)
    sum = 0
    for i in range(n):
        sum += (x1[i]-x2[i])**2
    return sum

def calculate_responsibilities(X, centroids, beta):
    k = len(centroids)
    lx = len(X)
    responsibilities = np.zeros((len(X), k))
    # print('len of x: ' ,len(X))
    for i in range(lx):
        # distances = np.sum((X[i] - self.centroids)**2, axis=1)
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        for j in range(k):
            # numerator = np.sqrt(_distance_square(X[i], centroids[k]))
            numerator = distances[j]
            # print(f'the {i}-{j}-th iter, numerator is {numerator} and distances sum is {np.sum(distances)}')
            if numerator == 0:
                responsibilities[i] = np.zeros(k)
                responsibilities[i, j] = 1
                continue
            denominator = np.sum((numerator / distances) ** (2 / (beta - 1)))
            responsibilities[i, j] = 1 / denominator

    return responsibilities

class CrispKNN:
    def __init__(self, X, labels, beta=2.0):
        self.X = X
        self.labels = labels
        self.beta = beta

    def knn(self, k, y):
        """
        Classify the input vector y using the k-nearest neighbors algorithm, by calculating
        membership degrees for each class based on inverse distances.

        Parameters:
            k (int): The number of nearest neighbors to consider.
            y (np.array): The input vector to classify.

        Returns:
            int: The predicted class for the input vector y, based on higher membership degree.
        """
        # Calculate the Euclidean distances from y to each point in X
        distances = np.linalg.norm(self.X - y, axis=1)

        # Get the indices of the k smallest distances
        nearest_indices = np.argsort(distances)[:k]

        # Get the corresponding k-nearest distances
        nearest_distances = distances[nearest_indices]

        # Get the labels of the k nearest neighbors
        nearest_labels = self.labels[nearest_indices]

        # Compute the inverse of the distances, preventing division by zero
        inv_distances = 1 / np.maximum(nearest_distances, np.finfo(float).eps)

        # Calculate membership scores for each class
        membership_scores = {}
        for class_label in np.unique(self.labels):
            class_mask = nearest_labels == class_label
            membership_scores[class_label] = np.sum(inv_distances[class_mask])

        # Normalize membership scores by the sum of all inverse distances
        total_inv_distance = np.sum(inv_distances)
        for class_label in membership_scores:
            membership_scores[class_label] /= total_inv_distance

        # Determine the class with the highest membership score
        # predicted_class = max(membership_scores, key=membership_scores.get)

        return membership_scores

class FuzzyKNN(CrispKNN):
    def __init__(self, X, labels, beta, softlabels):
        super().__init__(X, labels)
        self.beta = beta
        self.softlabels = softlabels

    def knn(self, k, y):
        """
        Classify the input vector y using a fuzzy k-nearest neighbors algorithm,
        utilizing soft labels and a modified distance weighting scheme.

        Parameters:
            k (int): The number of nearest neighbors to consider.
            y (np.array): The input vector to classify.

        Returns:
            int: The predicted class for the input vector y, based on higher membership degree.
        """
        # Calculate the Euclidean distances from y to each point in X
        distances = np.linalg.norm(self.X - y, axis=1)

        # Get the indices of the k smallest distances
        nearest_indices = np.argsort(distances)[:k]

        # Compute the distance weights using the beta parameter
        weights = 1 / (distances[nearest_indices] ** (2 / (self.beta - 1)) + np.finfo(float).eps)

        # Calculate membership scores for each class using soft labels and weights
        membership_scores = np.zeros(len(np.unique(self.labels)))  # Assuming labels are 0 and 1
        # membership_scores = {0:0.0,1:0.0}
        for i, index in enumerate(nearest_indices):
            # For each class, accumulate the product of soft labels and weights
            membership_scores += self.softlabels[index] * weights[i]

        # Normalize membership scores by the sum of weights for stability
        membership_scores /= np.sum(weights)

        # Determine the class with the highest membership score
        # predicted_class = np.argmax(membership_scores)
        res = {0:membership_scores[0], 1:membership_scores[1]}
        return res