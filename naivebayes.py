import numpy as np

class NaiveBayesClassifierCustom:
    def fit(self, features, labels):
        """
        Fit the Naive Bayes model to the training data.

        Parameters:
        - features: array-like, feature matrix
        - labels: array-like, target vector

        Returns:
        None
        """
        num_samples, num_features = features.shape
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)

        # Initialize arrays to store mean, variance, and priors for each class
        self._means = np.zeros((num_labels, num_features), dtype=np.float64)
        self._variances = np.zeros((num_labels, num_features), dtype=np.float64)
        self._priors = np.zeros(num_labels, dtype=np.float64)

        # Calculate mean, variance, and priors for each class
        for idx, label in enumerate(unique_labels):
            features_with_label = features[labels == label]
            self._means[idx, :] = features_with_label.mean(axis=0)
            self._variances[idx, :] = features_with_label.var(axis=0)
            self._priors[idx] = features_with_label.shape[0] / float(num_samples)

    def predict(self, X):
        """
        Predict the class labels for input data.

        Parameters:
        - X: array-like, feature matrix of shape (n_samples, n_features)

        Returns:
        - array-like, predicted class labels
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, sample):
        """
        Predict the class label for a single sample.

        Parameters:
        - sample: array-like, single sample of shape (n_features,)

        Returns:
        - int, predicted class label
        """
        posteriors = []

        # Calculate posterior probability for each class
        for idx, _ in enumerate(self._priors):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, sample)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # Return the class with the highest posterior probability
        return np.argmax(posteriors)

    def _pdf(self, label_index, sample):
        """
        Calculate the probability density function (pdf) for a given class.

        Parameters:
        - label_index: int, index of the class
        - sample: array-like, single sample of shape (n_features,)

        Returns:
        - array-like, pdf values for each feature
        """
        mean = self._means[label_index]
        variance = self._variances[label_index]
        numerator = np.exp(-((sample - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator

# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        """
        Compute accuracy score.

        Parameters:
        - y_true: array-like, true labels
        - y_pred: array-like, predicted labels

        Returns:
        - float, accuracy score
        """
        return np.sum(y_true == y_pred) / len(y_true)

    # Generate sample data
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Train the Naive Bayes classifier
    nb_classifier = NaiveBayesClassifierCustom()
    nb_classifier.fit(X_train, y_train)
    predictions = nb_classifier.predict(X_test)

    # Calculate and print accuracy
    print("Naive Bayes classification accuracy:", accuracy(y_test, predictions))
