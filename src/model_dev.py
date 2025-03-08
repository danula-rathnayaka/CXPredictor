import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, Y_train):
        """
        Train the model
        :param X_train: x training data
        :param Y_train: y training labels
        :return: model
        """


class LinerRegressionModel(Model):
    """
    Linear Regression model
    """

    def train(self, X_train, Y_train, **kwargs):
        try:
            rag = LinearRegression(**kwargs)
            rag.fit(X_train, Y_train)
            logging.info("Model training completed")
            return rag

        except Exception as e:
            logging.error(f"Error while training the linear regression model: {e}")
