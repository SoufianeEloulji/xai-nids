from src import logger
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

class ModelLoader:
    def __init__(self, model_name: str, model_alias: str = "production"):
        self.model_name = model_name
        self.model_alias = model_alias
        self.model = None

    def load_model(self):
        """
        Load the model from MLflow using the specified model name and alias.
        """
        try:
            model_uri = f"models:/{self.model_name}@{self.model_alias}"
            self.model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Type du modèle chargé : {type(self.model)}")
            logger.info(f"MRO : {type(self.model).__mro__}")
            logger.info(f"Model '{self.model_name}' loaded successfully.")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}' from MLflow: {e}")
            raise