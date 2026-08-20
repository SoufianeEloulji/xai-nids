from src import logger
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, PreProcessingStrategy, DataEncodingScalingStrategy, DataDivideStrategy
from typing import Tuple, Annotated

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    A ZenML step that cleans the ingested data.
    Args:
        df (pd.DataFrame): The ingested data as a pandas DataFrame.
    Returns:
        tuple: A tuple containing the cleaned training and testing data (X_train, X_test, y_train, y_test).
    """
    try:
        preprocessing_strategy = PreProcessingStrategy()
        data_preprocessing = DataCleaning(df, preprocessing_strategy)
        processed_df = data_preprocessing.handle_data()

        divide_strategy = DataDivideStrategy()
        data_divider = DataCleaning(processed_df, divide_strategy)
        X_train, X_test, y_train, y_test = data_divider.handle_data()

        encoding_scaling_strategy = DataEncodingScalingStrategy()
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = encoding_scaling_strategy.handle_data(X_train, X_test, y_train, y_test)
        logger.info("Data cleaning step completed successfully.")
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
    except Exception as e:
        logger.error(f"Error in clean_data step: {e}")
        raise