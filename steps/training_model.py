import torch
from src import logger
import pandas as pd
from zenml import step
from src.model_dev import MLPTrainingStrategy
from typing import Annotated


@step(enable_cache=False)
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Annotated[torch.nn.Module, "Trained_model"]:
    """
    Trains a model on the provided data.

    Args:
        X_train (pd.DataFrame): The features to train the model on.
        y_train (pd.Series): The target values to train the model on.

    Returns:
        torch.nn.Module: The trained model.
    """
    try:
        training_strategy = MLPTrainingStrategy()
        trained_model = training_strategy.train_model(X_train, y_train)
        logger.info("Model training step completed successfully.")
        return trained_model
    except Exception as e:
        logger.error(f"Error in training model step: {e}")
        raise
    