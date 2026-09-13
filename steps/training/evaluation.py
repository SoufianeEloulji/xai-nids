from torch import nn
from src import logger
import pandas as pd
from zenml import step
from src.training.evaluating_model import ModelEvaluation

@step(enable_cache=False)
def evaluate_model(model: nn.Module, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluates the trained model on the provided data.

    Args:
        model (nn.Module): The trained model.
        X_test (pd.DataFrame): The features of the test dataset.
        y_test (pd.Series): The labels of the test dataset.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    try:
        evaluator_instance = ModelEvaluation()
        metrics = evaluator_instance.evaluate(model, X_test, y_test)
        evaluator_instance.log_into_mlflow(model, metrics, X_test)
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise
    