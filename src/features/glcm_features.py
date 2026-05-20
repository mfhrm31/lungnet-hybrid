"""
GLCM (Gray-Level Co-occurrence Matrix) feature extraction.

Based on Maqbool et al. (2026), Journal of Engineering.
DOI: 10.1155/je/1367255
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from typing import Dict, List, Tuple


class GLCMFeatureExtractor:
    """
    Extract GLCM texture features from medical images.

    GLCM captures spatial relationships between pixel intensities,
    providing texture descriptors useful for distinguishing benign
    from malignant lung nodules in CT scans.

    Args:
        distances: List of pixel pair distances
        angles: List of angles in radians (0, 45, 90, 135 degrees)
        levels: Number of gray levels (256 for 8-bit images)
        symmetric: Whether to use symmetric GLCM
        normed: Whether to normalize the GLCM
    """

    def __init__(
        self,
        distances: List[int] = [1, 2, 3],
        angles: List[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels: int = 256,
        symmetric: bool = True,
        normed: bool = True,
    ):
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.symmetric = symmetric
        self.normed = normed

        self.properties = [
            'contrast',
            'dissimilarity',
            'homogeneity',
            'energy',
            'correlation',
            'ASM',
        ]

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract GLCM features from a single 2D image.

        Args:
            image: 2D grayscale image array

        Returns:
            Dictionary mapping feature names to values.
        """
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Compute GLCM
        glcm = graycomatrix(
            image,
            distances=self.distances,
            angles=self.angles,
            levels=self.levels,
            symmetric=self.symmetric,
            normed=self.normed,
        )

        # Extract properties for each distance/angle pair
        features = {}
        for prop in self.properties:
            values = graycoprops(glcm, prop)
            for d_idx, d in enumerate(self.distances):
                for a_idx, _ in enumerate(self.angles):
                    key = f"glcm_{prop}_d{d}_a{a_idx}"
                    features[key] = float(values[d_idx, a_idx])

        return features

    def extract_batch(
        self, images: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract GLCM features from a batch of images.

        Args:
            images: Array of shape (N, H, W) containing N grayscale images

        Returns:
            Tuple of:
                - feature_matrix: Array of shape (N, num_features)
                - feature_names: List of feature names
        """
        feature_dicts = [self.extract(img) for img in images]
        feature_names = list(feature_dicts[0].keys())
        feature_matrix = np.array(
            [[d[k] for k in feature_names] for d in feature_dicts]
        )
        return feature_matrix, feature_names


if __name__ == "__main__":
    # Quick sanity check
    extractor = GLCMFeatureExtractor()
    dummy_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    features = extractor.extract(dummy_image)
    print(f"Extracted {len(features)} GLCM features")
    print(f"Sample features:")
    for k, v in list(features.items())[:5]:
        print(f"  {k}: {v:.4f}")
