import logging
import pandas as pd
from zenml import step
from src.model_dev import LinerRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


@step
def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame,
                y_test: pd.DataFrame, config: ModelNameConfig) -> LinerRegressionModel:
    """
    Train a linear regression model
    :param X_train: pd.DataFrame
    :param y_train: pd.DataFrame
    :param X_test: pd.DataFrame
    :param y_test: pd.DataFrame
    :param config: ModelNameConfig
    :return:
    """
    try:
        model = None
        if config.model_name == "Linear Regression":
            model = LinerRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging.error(f"Error while training the model: {e}")
        raise e
