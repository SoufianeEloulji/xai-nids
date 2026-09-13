import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.training.utils import TabularDataset, IntrusionDetectionModel
from abc import ABC, abstractmethod

#from src.explainability.shap_background import build_and_save_background


class TrainingStrategy(ABC):
    """abstract base class for training strategies."""
    @abstractmethod
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> nn.Module:
        pass


class MLPTrainingStrategy(TrainingStrategy):
    """
    Concrete training strategy for MLP models."""

    def __init__(self, config_path: str = "params.yaml"):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.config = config
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        self.num_classes = config['data']['num_classes']
        self.benign_label_idx = config['data']['benign_label_idx']
        self.hidden_layers = config['model']['hidden_layers']
        self.dropout_rate = config['model']['dropout_rate']

        '''
        to uncomment if you want to save a new background for the explainer module
        
        shap_cfg = config.get('shap', {})
        self.build_shap_background = shap_cfg.get('build_background', True)
        self.shap_background_path = shap_cfg.get('background_path', 'artifacts/shap_background.joblib')
        self.shap_background_size = shap_cfg.get('background_size', 500)
        self.shap_background_seed = shap_cfg.get('background_seed', 42)
        '''

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> nn.Module:
        print(f"Training started on {self.device} ")
        
        input_dim = X_train.shape[1]
        model = IntrusionDetectionModel(
            input_dim,
            self.num_classes,
            self.hidden_layers,
            self.dropout_rate
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        y_values = y_train.values
        benign_positions = np.where(y_values == self.benign_label_idx)[0]
        attack_positions = np.where(y_values != self.benign_label_idx)[0]
        num_attacks = len(attack_positions)
        print(f"Total attaques : {num_attacks} | Total bénin initial : {len(benign_positions)}")

        for epoch in range(self.epochs):
            # dynamic undersampling: sample benign positions to match the number of attack samples
            sampled_benign = np.random.choice(benign_positions, size=num_attacks, replace=False)
            combined_pos = np.concatenate([sampled_benign, attack_positions])
            np.random.shuffle(combined_pos)

    

            X_epoch = X_train.iloc[combined_pos]
            y_epoch = y_train.iloc[combined_pos]

            train_loader = DataLoader(TabularDataset(X_epoch, y_epoch), batch_size=self.batch_size, shuffle=True)
            
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_train_loss:.4f}")

        '''
                to uncomment if you want to save a new background for the explainer module
        if self.build_shap_background:
            print("Génération du background SHAP (nettoyage + échantillonnage stratifié)...")
            build_and_save_background(
                X=X_train,
                y=y_train,
                output_path=self.shap_background_path,
                n_samples=self.shap_background_size,
                seed=self.shap_background_seed,
        )
        '''

        return model

