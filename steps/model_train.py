import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression


@step
def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame,
                y_test: pd.DataFrame) -> LinearRegression:
    """
    Train a linear regression model
    :param X_train: pd.DataFrame
    :param y_train: pd.DataFrame
    :param X_test: pd.DataFrame
    :param y_test: pd.DataFrame
    :return:
    """
    try:
        model = LinearRegression()
        trained_model = model.fit(X_train, y_train)
        return trained_model
    except Exception as e:
        logging.error(f"Error while training the model: {e}")
        raise e
