"""
Kruskal-Wallis non-parametric feature selection.

Selects the top-K most discriminative features based on the
Kruskal-Wallis H-test, a non-parametric alternative to ANOVA.

Based on Maqbool et al. (2026), Journal of Engineering.
DOI: 10.1155/je/1367255
"""

import numpy as np
from scipy.stats import kruskal
from typing import List, Tuple


class KruskalWallisSelector:
    """
    Feature selector based on Kruskal-Wallis H-test.

    For each feature, computes the H-statistic and p-value of the
    Kruskal-Wallis test across the classes. Features with the highest
    H-statistic (most discriminative) are retained.

    This non-parametric method does not assume normal distributions,
    making it well-suited for hybrid feature spaces combining
    handcrafted and deep features that may have different distributions.

    Args:
        top_k: Number of top features to retain (default: 400 per paper)
    """

    def __init__(self, top_k: int = 400):
        self.top_k = top_k
        self.h_statistics_ = None
        self.p_values_ = None
        self.selected_indices_ = None
        self.feature_names_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
    ) -> "KruskalWallisSelector":
        """
        Compute Kruskal-Wallis statistics for each feature.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Class labels of shape (n_samples,)
            feature_names: Optional list of feature names

        Returns:
            self
        """
        n_features = X.shape[1]
        h_stats = np.zeros(n_features)
        p_values = np.zeros(n_features)

        classes = np.unique(y)

        for f in range(n_features):
            groups = [X[y == c, f] for c in classes]

            try:
                h, p = kruskal(*groups)
                h_stats[f] = h
                p_values[f] = p
            except ValueError:
                # All values identical → no discrimination
                h_stats[f] = 0
                p_values[f] = 1.0

        self.h_statistics_ = h_stats
        self.p_values_ = p_values

        # Top-K by H-statistic (higher = more discriminative)
        top_k = min(self.top_k, n_features)
        self.selected_indices_ = np.argsort(h_stats)[::-1][:top_k]

        if feature_names is not None:
            self.feature_names_ = [feature_names[i] for i in self.selected_indices_]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce feature matrix to the top-K selected features.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Reduced matrix of shape (n_samples, top_k)
        """
        if self.selected_indices_ is None:
            raise RuntimeError("Must call fit() before transform()")
        return X[:, self.selected_indices_]

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
    ) -> np.ndarray:
        """Fit selector and return reduced matrix."""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_selected_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Return information about the selected features.

        Returns:
            Tuple of (selected_indices, selected_feature_names)
        """
        if self.selected_indices_ is None:
            raise RuntimeError("Must call fit() before this method")
        return self.selected_indices_, self.feature_names_


if __name__ == "__main__":
    # Sanity check with synthetic data
    np.random.seed(42)
    n_samples, n_features = 200, 1000

    X_class0 = np.random.randn(n_samples // 2, n_features)
    X_class1 = np.random.randn(n_samples // 2, n_features)
    X_class1[:, :50] += 2.0  # First 50 features are discriminative

    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    feature_names = [f"feature_{i}" for i in range(n_features)]

    selector = KruskalWallisSelector(top_k=50)
    X_reduced = selector.fit_transform(X, y, feature_names)

    print(f"Original shape: {X.shape}")
    print(f"Reduced shape: {X_reduced.shape}")

    selected_indices, selected_names = selector.get_selected_features()
    truly_discriminative = sum(1 for i in selected_indices if i < 50)
    print(f"Truly discriminative features captured: {truly_discriminative}/50")
