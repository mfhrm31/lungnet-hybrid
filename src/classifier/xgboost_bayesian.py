"""
XGBoost classifier with Bayesian hyperparameter optimization.

Uses Bayesian optimization to efficiently search the hyperparameter
space for an XGBoost classifier on the hybrid feature space.

Based on Maqbool et al. (2026), Journal of Engineering.
DOI: 10.1155/je/1367255
"""

import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from typing import Dict, Tuple


class XGBoostBayesianClassifier:
    """
    XGBoost classifier with Bayesian hyperparameter optimization.

    Optimizes the following hyperparameters via Bayesian optimization
    using cross-validation accuracy as the objective:
        - max_depth
        - learning_rate
        - n_estimators
        - gamma
        - min_child_weight

    Bayesian optimization builds a probabilistic model of the objective
    function and uses an acquisition function (Upper Confidence Bound)
    to intelligently select hyperparameter combinations to evaluate,
    typically requiring far fewer iterations than grid or random search.

    Args:
        n_iter: Number of Bayesian optimization iterations
        init_points: Number of random initial points
        cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        n_iter: int = 20,
        init_points: int = 5,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.n_iter = n_iter
        self.init_points = init_points
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.best_params_ = None
        self.best_score_ = None
        self.model_ = None
        self.optimizer_ = None

        self.pbounds = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (100, 1000),
            'gamma': (0, 10),
            'min_child_weight': (1, 10),
        }

    def _objective(
        self,
        max_depth: float,
        learning_rate: float,
        n_estimators: float,
        gamma: float,
        min_child_weight: float,
    ) -> float:
        """
        Objective function for Bayesian optimization.

        Returns mean cross-validation accuracy with the given
        hyperparameters. Uses cross-validation only (no test-set leakage).
        """
        model = xgb.XGBClassifier(
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            gamma=gamma,
            min_child_weight=min_child_weight,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
        )

        scores = cross_val_score(
            model,
            self._X_train,
            self._y_train,
            cv=self.cv_folds,
            scoring='accuracy',
            n_jobs=-1,
        )
        return scores.mean()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBayesianClassifier":
        """
        Optimize hyperparameters and fit final model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Class labels of shape (n_samples,)

        Returns:
            self
        """
        self._X_train = X
        self._y_train = y

        self.optimizer_ = BayesianOptimization(
            f=self._objective,
            pbounds=self.pbounds,
            random_state=self.random_state,
            verbose=2,
        )

        self.optimizer_.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter,
        )

        best = self.optimizer_.max
        self.best_score_ = best['target']
        self.best_params_ = {
            'max_depth': int(best['params']['max_depth']),
            'learning_rate': best['params']['learning_rate'],
            'n_estimators': int(best['params']['n_estimators']),
            'gamma': best['params']['gamma'],
            'min_child_weight': best['params']['min_child_weight'],
        }

        self.model_ = xgb.XGBClassifier(
            **self.best_params_,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
        )
        self.model_.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model_ is None:
            raise RuntimeError("Must call fit() before predict()")
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model_ is None:
            raise RuntimeError("Must call fit() before predict_proba()")
        return self.model_.predict_proba(X)

    def get_best_params(self) -> Dict[str, float]:
        """Return best hyperparameters found by Bayesian optimization."""
        if self.best_params_ is None:
            raise RuntimeError("Must call fit() before this method")
        return self.best_params_


if __name__ == "__main__":
    # Sanity check with synthetic binary classification data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300,
        n_features=50,
        n_informative=20,
        random_state=42,
    )

    classifier = XGBoostBayesianClassifier(n_iter=5, init_points=3, cv_folds=3)
    classifier.fit(X, y)

    print(f"\nBest CV accuracy: {classifier.best_score_:.4f}")
    print(f"Best hyperparameters: {classifier.get_best_params()}")
