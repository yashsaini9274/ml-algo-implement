import numpy as np


def r_squared_score(true_values, predicted_values):
    """Calculate the coefficient of determination (R-squared)"""
    corr_matrix = np.corrcoef(true_values, predicted_values)
    corr = corr_matrix[0, 1]
    return corr ** 2


class LinearRegressionModel:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        """Initialize Linear Regression model."""
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Fit the model to the training data."""
        num_samples, num_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            predicted_values = np.dot(X, self.weights) + self.bias
            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (predicted_values - y))
            db = (1 / num_samples) * np.sum(predicted_values - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """Predict output for input data."""
        predicted_values = np.dot(X, self.weights) + self.bias
        return predicted_values


# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def mean_squared_error(true_values, predicted_values):
        """Compute Mean Squared Error."""
        return np.mean((true_values - predicted_values) ** 2)

    # Generate sample data
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # Train the regressor
    regressor = LinearRegressionModel(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)

    # Make predictions on test data
    predictions = regressor.predict(X_test)

    # Compute MSE and R-squared
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error:", mse)

    r_squared = r_squared_score(y_test, predictions)
    print("R-squared:", r_squared)

    # Plot results
    predicted_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    training_data = plt.scatter(X_train, y_train, color=cmap(0.9), s=30, label="Training Data")
    testing_data = plt.scatter(X_test, y_test, color=cmap(0.5), s=30, label="Testing Data")
    plt.plot(X, predicted_line, color="red", linewidth=3, label="Prediction")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.legend()
    plt.grid(True)
    plt.show()
