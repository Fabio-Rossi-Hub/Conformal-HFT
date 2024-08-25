import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple, Dict

class VennAbersMultiClass:
    def __init__(self, model: nn.Module, cal_size: float = 0.2, random_state: int = None, 
                 shuffle: bool = True, stratify: bool = True):
        self.model = model
        self.cal_size = cal_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        self.calibrators: List[IsotonicRegression] = []
        self.n_classes: int = None
        self.classes: np.ndarray = None
        self.is_fitted: bool = False
        self.device = next(model.parameters()).device
        self.model = self.model.to(self.device)

    def fit(self, loader: torch.utils.data.DataLoader) -> None:
        all_outputs, all_labels = self._get_model_outputs(loader)

        self.classes = np.unique(all_labels)
        self.n_classes = len(self.classes)

        indices = np.arange(len(all_outputs))
        train_indices, calib_indices = train_test_split(
            indices, test_size=self.cal_size, random_state=self.random_state, 
            shuffle=self.shuffle, stratify=all_labels if self.stratify else None)

        self._fit_calibrators(all_outputs[train_indices], all_labels[train_indices],
                              all_outputs[calib_indices], all_labels[calib_indices])
        self.is_fitted = True

    def _fit_calibrators(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_calib: np.ndarray, y_calib: np.ndarray) -> None:
        self.calibrators = [
            IsotonicRegression(out_of_bounds='clip').fit(X_calib[:, i], y_calib == i)
            for i in range(self.n_classes)
        ]

    def predict_proba(self, outputs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Calibrator is not fitted yet. Call 'fit' first.")
        
        return np.column_stack([
            self.calibrators[i].transform(outputs[:, i])
            for i in range(self.n_classes)
        ])

    def _get_model_outputs(self, loader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        all_outputs, all_labels = [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
        return torch.cat(all_outputs).numpy(), torch.cat(all_labels).numpy()

    def predict_set(self, calibrated_probs: np.ndarray, alpha: float = 0.05) -> List[List[int]]:
        return [
            [idx for idx in np.argsort(prob)[::-1] if sum(sorted(prob, reverse=True)[:idx+1]) < 1 - alpha]
            for prob in calibrated_probs
        ]

    def get_statistics(self, loader: torch.utils.data.DataLoader, alpha: float = 0.05) -> Dict[str, float]:
        """
        Get all statistics in a dictionary format.
        """
        uncalibrated_outputs, true_labels = self._get_model_outputs(loader)
        calibrated_probs = self.predict_proba(uncalibrated_outputs)
        
        prediction_sets = self.predict_set(calibrated_probs, alpha)
        avg_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])

        #uncalibrated_brier = brier_score_loss(true_labels.flatten(), uncalibrated_outputs.flatten())
        calibrated_brier = brier_score_loss(true_labels.flatten(), calibrated_probs.flatten())
        
        #uncalibrated_log = log_loss(true_labels, uncalibrated_outputs)
        calibrated_log = log_loss(true_labels, calibrated_probs)
        
        #uncalibrated_acc = accuracy_score(true_labels, (uncalibrated_outputs > 0.5).astype(int))
        calibrated_acc = accuracy_score(true_labels, (calibrated_probs > 0.5).astype(int))

        return {
        #    "Uncalibrated Brier Loss": uncalibrated_brier,
            "Calibrated Brier Loss": calibrated_brier,
        #    "Uncalibrated Log Loss": uncalibrated_log,
            "Calibrated Log Loss": calibrated_log,
        #    "Uncalibrated Accuracy": uncalibrated_acc,
            "Calibrated Accuracy": calibrated_acc,
            "Average Prediction Set Size": avg_set_size
        }