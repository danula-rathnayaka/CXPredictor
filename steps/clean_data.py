import logging
from typing import Tuple

from typing_extensions import Annotated

import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataSplittingStrategy, DataPreProcessingStrategy


@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleans the data and divide
    :param df: raw data
    :return: training and testing data and labels
    """
    try:
        data_cleaning = DataCleaning(df, DataPreProcessingStrategy())
        processed_data = data_cleaning.handle_data()

        data_cleaning = DataCleaning(processed_data, DataSplittingStrategy())
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()

        logging.info("Data cleaning completed")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error while cleaning data: {e}")
        raise e
