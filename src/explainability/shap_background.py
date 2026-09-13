import numpy as np
import pandas as pd
import joblib
from src import logger



def stratified_sample(X: pd.DataFrame, y: pd.Series, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Creating a stratified sample of the training data to be used as background for SHAP.
    The sample is drawn from the original training data, not the undersampled data used for training the model.
    Args:
        X (pd.DataFrame): training features (all classes, not the undersampled subset used for training).
        y (pd.Series): corresponding encoded labels (same index as X, or reindexable positionally).
        n_samples (int): total size of the background sample.
        seed (int): random seed for reproducibility.
    Returns:
        pd.DataFrame: stratified sample of the training data, to be used as background for SHAP.
    """
    rng = np.random.RandomState(seed)
    y_aligned = y.loc[X.index]

    class_counts = y_aligned.value_counts()
    proportions = (
        (class_counts / class_counts.sum() * n_samples)
        .round()
        .astype(int)
        .clip(lower=1)
    )

    sampled_idx = []
    for cls, n_cls in proportions.items():
        cls_idx = y_aligned[y_aligned == cls].index
        n_take = min(n_cls, len(cls_idx))
        sampled_idx.extend(rng.choice(cls_idx, size=n_take, replace=False))

    sample = X.loc[pd.Index(sampled_idx)]
    logger.info(
        f"[SHAP background] Échantillon stratifié : {len(sample)} lignes sur "
        f"{len(class_counts)} classes\n{proportions.to_string()}"
    )
    return sample


def build_and_save_background(
    X: pd.DataFrame,
    y: pd.Series,
    output_path: str = "artifacts/shap_background.joblib",
    n_samples: int = 150,
    seed: int = 42,
) -> str:
    """
    Complete pipeline to build a stratified sample of the training data and save it to disk for SHAP background.
    Args:
        X (pd.DataFrame): training features (all classes, not the undersampled subset used for training).
        y (pd.Series): corresponding encoded labels (same index as X, or reindexable positionally).
        output_path (str): path to save the background sample.
        n_samples (int): total size of the background sample.
        seed (int): random seed for reproducibility.
    Returns:
        str: path to the saved background sample.
    """
    y_reindexed = y.copy()
    y_reindexed.index = X.index

    background = stratified_sample(X, y_reindexed, n_samples=n_samples, seed=seed)

    joblib.dump(background.values.astype(np.float32), output_path)
    logger.info(f"[SHAP background] Sauvegardé : shape={background.shape} -> {output_path}")
    return output_path