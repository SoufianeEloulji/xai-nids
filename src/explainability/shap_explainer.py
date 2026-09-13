from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import types
import torch
import shap
import joblib
from src import logger


class ShapExplainer:
    """
    Wrapper around SHAP's GradientExplainer to compute feature contributions for a PyTorch model.
    It uses a precomputed background sample for SHAP and provides methods to explain individual predictions.
    """

    def __init__(self, model, background_path: str, feature_names: List[str], top_k: int = 8):
        self.model = model
        self.feature_names = feature_names
        self.top_k = top_k

        if not self._supports_eval(self.model):
            logger.warning(
                f"[SHAP] {type(self.model).__name__} ne supporte pas .eval()/.train() "
                "après export (mode figé à l'export) : ces appels sont neutralisés. "
                f"État actuel model.training={getattr(self.model, 'training', 'inconnu')} "
                "(devrait être False si le modèle a bien été exporté après model.eval())."
            )
            self.model.eval = types.MethodType(lambda self_: self_, self.model)
            self.model.train = types.MethodType(lambda self_, mode=True: self_, self.model)

        try:
            background = joblib.load(background_path)
        except FileNotFoundError:
            logger.error(
                f"Background SHAP introuvable à '{background_path}'. "
                "Lancez scripts/build_shap_background.py avant de démarrer l'API."
            )
            raise

        background_tensor = torch.tensor(background, dtype=torch.float32)
        self.model.eval()
        self.explainer = shap.GradientExplainer(self.model, background_tensor)
        logger.info(f"SHAP GradientExplainer initialisé avec un background de shape {background_tensor.shape}")

    @staticmethod
    def _supports_eval(model) -> bool:
        try:
            model.eval()
            return True
        except Exception:
            return False

    def explain(
        self,
        X: torch.Tensor,
        class_id: int,
        raw_values: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        """
        Calculate SHAP values for a given input tensor X and a specific class_id.
        Optionally, raw_values can be provided to include the original feature values in the output.
        Args:
            X (torch.Tensor): Input tensor for which to compute SHAP values.
            class_id (int): Index of the class for which to compute SHAP values.
            raw_values (Dict[str, float], optional): Original feature values before scaling.
        Returns:
            Dict[str, Any]: Dictionary containing SHAP values and top features.
        """
        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            values_for_class = np.array(shap_values[class_id])[0]
        else:
            values_for_class = np.array(shap_values)[0, :, class_id]

        contributions = dict(zip(self.feature_names, values_for_class.tolist()))

        top_features = sorted(
            contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[: self.top_k]

        top_features_out = []
        for f, v in top_features:
            entry = {"feature": f, "shap_value": round(float(v), 6)}
            if raw_values is not None:
                raw_v = raw_values.get(f)
                entry["value"] = float(raw_v) if raw_v is not None else None
            top_features_out.append(entry)

        return {
            "shap_values": contributions,
            "top_features": top_features_out,
        }