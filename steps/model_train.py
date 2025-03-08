import logging
import pandas as pd
from zenml import step
from src.model_dev import LinerRegressionModel, Model
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame,
                y_test: pd.DataFrame) -> RegressorMixin:
    """
    Train a linear regression model
    :param X_train: pd.DataFrame
    :param y_train: pd.DataFrame
    :param X_test: pd.DataFrame
    :param y_test: pd.DataFrame
    :return:
    """
    config = ModelNameConfig()
    try:
        model = None
        if config.name == "Linear Regression":
            mlflow.sklearn.autolog()
            model = LinerRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.name} not supported")
    except Exception as e:
        logging.error(f"Error while training the model: {e}")
        raise e
