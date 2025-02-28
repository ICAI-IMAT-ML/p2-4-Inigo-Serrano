import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            print("ENTRE")
            X = X.reshape(-1, 1)
        else:
            print("NO ENTRE")

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            print("AAAA")
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session

        # Store the intercept and the coefficients of the model
        X = np.c_[np.ones(X.shape[0]), X]
        w =  np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
        self.intercept = w[0]
        self.coefficients = w[1:]

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01
        # Implement gradient descent (TODO)
        for epoch in range(iterations):
            predictions = X[:, 1:] @ self.coefficients + self.intercept  # Compute predictions
            error = predictions - y  # Compute error
            # Compute gradients
            gradient = (1/m) * X.T @ error
            
            # Update parameters
            self.intercept -= learning_rate * gradient[0]
            self.coefficients -= learning_rate * gradient[1]
            
            # Calculate and print the loss every 10 epochs
            if epoch % 1000 == 0:
                mse = (1 / m) * np.sum(error ** 2)
                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            predictions = np.array([self.intercept + self.coefficients * x for x in X])
        else:
            predictions = self.intercept + X @ self.coefficients 
            
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    rss = np.sum((y_true - y_pred) ** 2)  # Suma de los residuos al cuadrado
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (rss/tss)

    # Root Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)  
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    for index in sorted(categorical_indices, reverse=True):
        # Extract the categorical column
        categorical_column = X_transformed[:, index]
        
        # Find the unique categories (works with strings)
        unique_values = np.unique(categorical_column)
        
        # Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot = np.array([unique_values == val for val in categorical_column], dtype=int)
        
        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]
        
        # Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        X_transformed = np.delete(X_transformed, index, axis=1)
        X_transformed = np.insert(X_transformed, index, one_hot.T, axis=1)

    return X_transformed
