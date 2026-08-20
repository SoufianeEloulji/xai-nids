import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import yaml

from src.utils import TabularDataset
from src import logger

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("CICIDS2017_Experiment")

class ModelEvaluation:
    """
    Strategy for evaluating trained models and logging to MLflow.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self, model: nn.Module, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate the trained model on the test dataset.
        Args:
            model (nn.Module): The trained model.
            X_test (pd.DataFrame): The features of the test dataset.
            y_test (pd.Series): The labels of the test dataset.
        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        logger.info("Evaluation started")
        model.eval()
        model.to(self.device)
        
        test_dataset = TabularDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1_macro": f1_score(all_labels, all_preds, average="macro"),
            "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall_macro": recall_score(all_labels, all_preds, average="macro")
        }
        
        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics

    def log_into_mlflow(self, model: nn.Module, metrics: dict, X_test: pd.DataFrame):
        """
        Log metrics, parameters, and the model into MLflow.
        """


        logger.info("Logging results and model to MLflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with open("params.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        with mlflow.start_run(run_name=config['experiment']['run_name']):
            mlflow.log_params(config['training'])
            mlflow.log_params({
                "hidden_layers": str(config['model']['hidden_layers']),
                "dropout_rate": config['model']['dropout_rate']
            })
            
            mlflow.log_metrics(metrics)

            model.eval()
            model.to(self.device)
            sample_input = torch.tensor(X_test.iloc[:5].values, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                sample_output = model(sample_input)
                
            signature = infer_signature(
                model_input=X_test.iloc[:5].to_numpy(),
                model_output=sample_output.cpu().numpy()
            )

            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name="model",
                    signature=signature,
                    registered_model_name=config['experiment']['run_name'],
                    input_example=sample_input.cpu().numpy().astype(np.float32)
                )
            else:
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name="model",
                    signature=signature,
                    input_example=sample_input.cpu().numpy().astype(np.float32)
                )
                
        logger.info("MLflow logging completed successfully.")