from src.inference.preprocessor import PreProcessingStrategy, DataEncodingScalingStrategy
from src.inference.model_loader import ModelLoader
from src.explainability.shap_explainer import ShapExplainer
from src import logger
import numpy as np
import torch


CLASS_NAMES = ['BENIGN', 'DoS Hulk', 'DDoS', 'PortScan', 'DoS GoldenEye', 'FTP-Patator',
               'DoS slowloris', 'DoS Slowhttptest', 'SSH-Patator', 'Bot',
               'Web Attack  -  Brute Force', 'Web Attack  -  XSS', 'Infiltration',
               'Web Attack  -  Sql Injection', 'Heartbleed']


class Predictor:
    '''
    Predictor class that handles model loading, preprocessing, scaling, prediction, and optional SHAP explanations for feature contributions.
    '''
    def __init__(self, model_name, model_alias, artifacts_path,
                 background_path: str = "artifacts/shap_background.joblib",
                 enable_explanations: bool = True):
        self.model_name = model_name
        self.model_alias = model_alias
        self.artifacts_path = artifacts_path
        self.model = ModelLoader(self.model_name, self.model_alias).load_model()
        self.preprocess_strategy = PreProcessingStrategy()
        self.scaling_strategy = DataEncodingScalingStrategy(self.artifacts_path)

        self.enable_explanations = enable_explanations
        self.explainer = None
        self._background_path = background_path

    def _ensure_explainer(self, feature_names):
        if self.explainer is None and self.enable_explanations:
            try:
                self.explainer = ShapExplainer(
                    model=self.model,
                    background_path=self._background_path,
                    feature_names=feature_names,
                )
            except Exception as e:
                logger.error(f"Explications désactivées : échec d'initialisation de SHAP ({e})")
                self.enable_explanations = False

    def predict(self, input_data):
        """
        Predicts the output for the given input data and optionally computes SHAP explanations for feature contributions.
        Args:
            input_data (pd.DataFrame): The input data for prediction.
        Returns:
            dict: A dictionary containing the predicted class ID, label, confidence, and optionally top feature contributions if SHAP explanations are enabled.
        """
        processed_data = self.preprocess_strategy.handle_data(input_data)
        scaled_data = self.scaling_strategy.handle_data(processed_data)

        raw_values = processed_data.iloc[0].to_dict()

        X = torch.tensor(scaled_data.values, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(X)

        probabilities = torch.softmax(outputs, dim=1)
        class_id = torch.argmax(probabilities, dim=1)
        confidences = torch.max(probabilities, dim=1).values

        class_id_int = class_id.item()
        label = CLASS_NAMES[class_id_int]

        result = {
            "class_id": class_id_int,
            "Label": label,
            "confidence": confidences.item(),
            "top_features": [],
        }

        if self.enable_explanations:
            self._ensure_explainer(feature_names=scaled_data.columns.tolist())
            if self.explainer is not None:
                try:
                    explanation = self.explainer.explain(X, class_id_int, raw_values=raw_values)
                    result["top_features"] = explanation["top_features"]
                except Exception as e:
                    logger.error(f"Échec du calcul SHAP, prédiction renvoyée sans explication : {e}")

        return result
