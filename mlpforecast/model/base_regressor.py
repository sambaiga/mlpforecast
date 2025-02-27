from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import joblib
import os

class BaseRegressor(BaseEstimator, RegressorMixin):
    """
    A base class for regression models, compliant with scikit-learn pipelines,
    providing common functionality for feature formatting, model fitting, and forecasting.

    Attributes:
        model: The regression model to be used, initially set to None.
    """

    def __init__(self, data_pipeline=None):
        """
        Initializes the BaseRegressor class with a model set to None.
        """
        self.data_pipeline = data_pipeline
        self.num_targets=None


    

    

    def format_features(self, X):
        """
        Formats the input features by flattening multi-dimensional arrays into 2D arrays.

        Args:
            X (np.ndarray): The input features to be formatted. If X has more than 2 dimensions,
                            it will be reshaped to have shape (samples, features).

        Returns:
            np.ndarray: The formatted 2D feature array.
        """
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return X

    def fit(self, X, y):
        """
        Fits the regression model to the training data.

        Args:
            X (np.ndarray): The training input features.
            y (np.ndarray): The target values corresponding to the training input.

        Returns:
            self: Returns an instance of self.
        """
        self.num_targets=y.shape[-1] if y.ndim==3 else 1
        X = self.format_features(X)
        y = self.format_features(y)
        if self.model is not None:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts the output using the fitted regression model.

        Args:
            X (np.ndarray): The input features for which predictions are to be made.

        Returns:
            np.ndarray: Predicted values.
        """
        N, T, C = X.shape
        X = self.format_features(X)
        return self.model.predict(X).reshape(N, -1, self.num_targets)

    def forecast(self, X):
        """
        Forecasts the output using the fitted regression model.

        Args:
            X (np.ndarray): The input features for which predictions are to be made.

        Returns:
            dict: A dictionary containing the predicted values with the key 'loc'.
        """
        loc = self.predict(X)
        return {'loc': loc}

    def get_params(self, deep=True):
        """
        Returns the model parameters.

        Args:
            deep (bool): If True, return the parameters for this estimator and contained subobjects.

        Returns:
            dict: A dictionary of parameters.
        """
        return {"model": self.model}

    def set_params(self, **params):
        """
        Set the model parameters.

        Args:
            **params: Parameters to set in the model.

        Returns:
            self: Returns an instance of self.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
