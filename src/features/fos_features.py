"""
First-Order Statistics (FOS) feature extraction.

Based on Maqbool et al. (2026), Journal of Engineering.
DOI: 10.1155/je/1367255
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


class FOSFeatureExtractor:
    """
    Extract First-Order Statistics features from medical images.

    FOS features describe the distribution of pixel intensities
    without considering spatial relationships. They are computed
    directly from the image histogram.

    Features include: mean, variance, skewness, kurtosis, entropy,
    energy, median, range, percentiles, and standard deviation.
    """

    def __init__(self):
        self.feature_names = [
            'mean',
            'variance',
            'std',
            'skewness',
            'kurtosis',
            'entropy',
            'energy',
            'median',
            'min',
            'max',
            'range',
            'percentile_10',
            'percentile_25',
            'percentile_75',
            'percentile_90',
            'mean_absolute_deviation',
            'root_mean_square',
            'uniformity',
        ]

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract FOS features from a single image.

        Args:
            image: 2D grayscale image array

        Returns:
            Dictionary mapping feature names to values.
        """
        # Flatten to 1D
        pixels = image.flatten().astype(np.float64)

        # Normalize to [0, 1] if needed
        if pixels.max() > 1.0:
            pixels = pixels / 255.0

        features = {}

        # Basic statistics
        features['fos_mean'] = float(np.mean(pixels))
        features['fos_variance'] = float(np.var(pixels))
        features['fos_std'] = float(np.std(pixels))
        features['fos_skewness'] = float(stats.skew(pixels))
        features['fos_kurtosis'] = float(stats.kurtosis(pixels))

        # Entropy (Shannon)
        hist, _ = np.histogram(pixels, bins=256, range=(0, 1), density=True)
        hist = hist + 1e-10  # Avoid log(0)
        features['fos_entropy'] = float(-np.sum(hist * np.log2(hist)))

        # Energy (sum of squared probabilities)
        features['fos_energy'] = float(np.sum(hist ** 2))

        # Median and range
        features['fos_median'] = float(np.median(pixels))
        features['fos_min'] = float(np.min(pixels))
        features['fos_max'] = float(np.max(pixels))
        features['fos_range'] = features['fos_max'] - features['fos_min']

        # Percentiles
        features['fos_percentile_10'] = float(np.percentile(pixels, 10))
        features['fos_percentile_25'] = float(np.percentile(pixels, 25))
        features['fos_percentile_75'] = float(np.percentile(pixels, 75))
        features['fos_percentile_90'] = float(np.percentile(pixels, 90))

        # Additional descriptors
        features['fos_mean_absolute_deviation'] = float(
            np.mean(np.abs(pixels - features['fos_mean']))
        )
        features['fos_root_mean_square'] = float(
            np.sqrt(np.mean(pixels ** 2))
        )
        features['fos_uniformity'] = float(np.sum(hist ** 2))

        return features

    def extract_batch(
        self, images: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract FOS features from a batch of images.

        Args:
            images: Array of shape (N, H, W)

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        feature_dicts = [self.extract(img) for img in images]
        feature_names = list(feature_dicts[0].keys())
        feature_matrix = np.array(
            [[d[k] for k in feature_names] for d in feature_dicts]
        )
        return feature_matrix, feature_names


if __name__ == "__main__":
    extractor = FOSFeatureExtractor()
    dummy_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    features = extractor.extract(dummy_image)
    print(f"Extracted {len(features)} FOS features")
    print(f"Sample features:")
    for k, v in list(features.items())[:5]:
        print(f"  {k}: {v:.4f}")
