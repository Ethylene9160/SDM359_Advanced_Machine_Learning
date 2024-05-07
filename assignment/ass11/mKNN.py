import numpy as np

def calculate_responsibilities(X, centroids, beta,k=2):
    responsibilities = np.zeros((len(X), k))
    for i in range(len(X)):
        # distances = np.sum((X[i] - self.centroids)**2, axis=1)
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        for k in range(k):
            numerator = distances[k]
            if numerator == 0:
                responsibilities[i] = np.zeros(k)
                responsibilities[i, k] = 1
                continue
            denominator = np.sum((numerator / distances) ** (2 / (beta - 1)))
            responsibilities[i, k] = 1 / denominator
    return responsibilities

class CrispKNN:
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels

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

        for i, index in enumerate(nearest_indices):
            # For each class, accumulate the product of soft labels and weights
            membership_scores += self.softlabels[index] * weights[i]

        # Normalize membership scores by the sum of weights for stability
        membership_scores /= np.sum(weights)

        # Determine the class with the highest membership score
        # predicted_class = np.argmax(membership_scores)

        return membership_scores