"""
Histogram-based feature extraction.

Based on Maqbool et al. (2026), Journal of Engineering.
DOI: 10.1155/je/1367255
"""

import numpy as np
from typing import Dict, List, Tuple


class HistogramFeatureExtractor:
    """
    Extract histogram-based features from medical images.

    Histogram features describe the intensity distribution of an image
    by binning pixel values and computing statistics over the resulting
    distribution. Useful for capturing global intensity patterns in
    lung nodule CT scans.

    Args:
        num_bins: Number of histogram bins (default: 32)
        normalize: Whether to normalize histogram to probability distribution
        equalize: Whether to apply histogram equalization before binning
    """

    def __init__(
        self,
        num_bins: int = 32,
        normalize: bool = True,
        equalize: bool = False,
    ):
        self.num_bins = num_bins
        self.normalize = normalize
        self.equalize = equalize

    def _equalize_histogram(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to enhance contrast."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * (255 / cdf[-1])
        equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        return equalized.reshape(image.shape).astype(np.uint8)

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract histogram features from a single image.

        Args:
            image: 2D grayscale image array

        Returns:
            Dictionary mapping feature names to values.
        """
        # Normalize to [0, 1]
        pixels = image.flatten().astype(np.float64)
        if pixels.max() > 1.0:
            pixels = pixels / 255.0

        if self.equalize:
            equalized = self._equalize_histogram(image)
            pixels = equalized.flatten().astype(np.float64) / 255.0

        # Compute histogram
        hist, bin_edges = np.histogram(
            pixels, bins=self.num_bins, range=(0, 1)
        )

        if self.normalize:
            hist = hist / (hist.sum() + 1e-10)

        features = {}

        # Per-bin values
        for i in range(self.num_bins):
            features[f'hist_bin_{i}'] = float(hist[i])

        # Aggregate histogram statistics
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Weighted mean and variance
        mean = np.sum(bin_centers * hist)
        variance = np.sum(((bin_centers - mean) ** 2) * hist)

        features['hist_mean'] = float(mean)
        features['hist_variance'] = float(variance)
        features['hist_std'] = float(np.sqrt(variance))

        # Skewness and kurtosis from histogram
        std = np.sqrt(variance) + 1e-10
        features['hist_skewness'] = float(
            np.sum(((bin_centers - mean) ** 3) * hist) / (std ** 3)
        )
        features['hist_kurtosis'] = float(
            np.sum(((bin_centers - mean) ** 4) * hist) / (std ** 4) - 3
        )

        # Mode and peak features
        features['hist_mode_bin'] = float(np.argmax(hist))
        features['hist_peak_value'] = float(np.max(hist))

        # Entropy
        hist_safe = hist + 1e-10
        features['hist_entropy'] = float(-np.sum(hist_safe * np.log2(hist_safe)))

        # Smoothness and uniformity
        features['hist_smoothness'] = float(1 - 1 / (1 + variance))
        features['hist_uniformity'] = float(np.sum(hist ** 2))

        return features

    def extract_batch(
        self, images: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract histogram features from a batch of images.

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
    extractor = HistogramFeatureExtractor(num_bins=32)
    dummy_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    features = extractor.extract(dummy_image)
    print(f"Extracted {len(features)} histogram features")
    print(f"Sample bins:")
    for k, v in list(features.items())[:5]:
        print(f"  {k}: {v:.4f}")
    print(f"Aggregate stats:")
    for k in ['hist_mean', 'hist_variance', 'hist_entropy']:
        print(f"  {k}: {features[k]:.4f}")
