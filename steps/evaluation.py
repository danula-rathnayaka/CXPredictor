import logging
import pandas as pd
import numpy as np
from zenml import step
from src.evaluation import MSE, R2, RMSE
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
]:
    try:
        y_test_np = y_test.to_numpy()

        prediction = model.predict(X_test)

        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test_np, prediction)
        mlflow.log_metric("mse", mse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test_np, prediction)
        mlflow.log_metric("r2", float(r2))
        print(type(r2))

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test_np, prediction)
        mlflow.log_metric("rmse", rmse)

        return r2, rmse
    except Exception as e:
        logging.error(f"Error while evaluating model: {str(e)}")
