"""
Evaluation metrics for lung nodule classification.

Computes the metrics reported in Maqbool et al. (2026):
accuracy, sensitivity, specificity, PPV, NPV, F1-score, MCC, AUC.

DOI: 10.1155/je/1367255
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Tuple


class ClassificationMetrics:
    """
    Comprehensive evaluation metrics for binary medical classification.

    Computes metrics used in the original paper:
        - Accuracy
        - Sensitivity (Recall / True Positive Rate)
        - Specificity (True Negative Rate)
        - PPV (Precision / Positive Predictive Value)
        - NPV (Negative Predictive Value)
        - F1-Score
        - MCC (Matthews Correlation Coefficient)
        - AUC (Area Under ROC Curve)
    """

    def __init__(self):
        self.results_ = None

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            y_true: Ground-truth class labels
            y_pred: Predicted class labels
            y_proba: Predicted class probabilities (for AUC)

        Returns:
            Dictionary mapping metric names to values
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = precision_score(y_true, y_pred, zero_division=0)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        results = {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
            'f1_score': float(f1),
            'mcc': float(mcc),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        }

        if y_proba is not None:
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]
            try:
                results['auc'] = float(roc_auc_score(y_true, y_proba))
            except ValueError:
                results['auc'] = None

        self.results_ = results
        return results

    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Print scikit-learn classification report."""
        print(classification_report(
            y_true,
            y_pred,
            target_names=['Normal', 'Cancer'],
            digits=4,
        ))

    def summary_table(self) -> str:
        """Return formatted summary of all metrics."""
        if self.results_ is None:
            return "No results computed. Call compute() first."

        lines = [
            "=" * 50,
            "       LungNet-Hybrid — Evaluation Summary",
            "=" * 50,
            f"  Accuracy     : {self.results_['accuracy']:.4f}",
            f"  Sensitivity  : {self.results_['sensitivity']:.4f}",
            f"  Specificity  : {self.results_['specificity']:.4f}",
            f"  PPV          : {self.results_['ppv']:.4f}",
            f"  NPV          : {self.results_['npv']:.4f}",
            f"  F1-Score     : {self.results_['f1_score']:.4f}",
            f"  MCC          : {self.results_['mcc']:.4f}",
        ]

        if 'auc' in self.results_ and self.results_['auc'] is not None:
            lines.append(f"  AUC          : {self.results_['auc']:.4f}")

        lines.extend([
            "-" * 50,
            f"  TP: {self.results_['true_positives']}  "
            f"TN: {self.results_['true_negatives']}  "
            f"FP: {self.results_['false_positives']}  "
            f"FN: {self.results_['false_negatives']}",
            "=" * 50,
        ])
        return "\n".join(lines)


if __name__ == "__main__":
    np.random.seed(42)

    n_samples = 1339
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.62, 0.38])

    y_pred = y_true.copy()
    flip_mask = np.random.rand(n_samples) < 0.024
    y_pred[flip_mask] = 1 - y_pred[flip_mask]

    y_proba = np.random.rand(n_samples)
    y_proba[y_true == 1] = 0.5 + 0.5 * np.random.rand(np.sum(y_true == 1))

    metrics = ClassificationMetrics()
    results = metrics.compute(y_true, y_pred, y_proba)
    print(metrics.summary_table())
