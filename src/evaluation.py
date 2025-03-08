import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np


class Evaluation(ABC):
    """
    Abstract class defining the strategy for evaluating the models
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores of the models
        :param y_true: True labels of the data
        :param y_pred: Predicted labels of the data
        """
        pass


class MSE(Evaluation):
    """
    Evaluation strategy for calculating the mean squared error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error(f"Error calculating MSE: {e}")
            raise e


class R2(Evaluation):
    """
    Evaluation strategy for calculating the R2 score
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error(f"Error calculating R2 Score: {e}")
            raise e


class RMSE(Evaluation):
    """
    Evaluation strategy for calculating the root mean squared error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np):
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error(f"Error calculating RMSE: {e}")
            raise e
