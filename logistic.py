import numpy as np

class LogisticRegressionModel:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        """
        Initialize Logistic Regression model.

        Parameters:
        - learning_rate: float, learning rate for gradient descent
        - n_iterations: int, number of iterations for gradient descent
        """
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
        - X: array-like, features
        - y: array-like, target

        Returns:
        None
        """
        num_samples, num_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            # Linear combination of weights and features, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict output for input data.

        Parameters:
        - X: array-like, input features

        Returns:
        - array-like, predicted classes
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        """
        Sigmoid activation function.

        Parameters:
        - x: array-like, input

        Returns:
        - array-like, sigmoid output
        """
        return 1 / (1 + np.exp(-x))

# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy_score(y_true, y_pred):
        """
        Compute accuracy score.

        Parameters:
        - y_true: array-like, true labels
        - y_pred: array-like, predicted labels

        Returns:
        - float, accuracy score
        """
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # Load breast cancer dataset
    breast_cancer = datasets.load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # Train the logistic regression model
    logistic_regressor = LogisticRegressionModel(learning_rate=0.0001, n_iterations=1000)
    logistic_regressor.fit(X_train, y_train)
    predictions = logistic_regressor.predict(X_test)

    # Calculate and print accuracy
    print("Logistic Regression classification accuracy:", accuracy_score(y_test, predictions))
