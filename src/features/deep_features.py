"""
ResNet50 deep feature extraction.

Extracts 2048-dimensional dynamic features from the avgpool layer
of a ResNet50 model pretrained on ImageNet.

Based on Maqbool et al. (2026), Journal of Engineering.
DOI: 10.1155/je/1367255
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import List, Tuple


class ResNet50FeatureExtractor:
    """
    Extract deep features from ResNet50 pretrained on ImageNet.

    Removes the final classification layer and extracts the 2048-dim
    feature vector from the global average pooling layer. These dynamic
    features capture high-level semantic and textural patterns
    complementary to handcrafted static features.

    Args:
        device: 'cuda' or 'cpu'
        pretrained: Whether to load ImageNet pretrained weights
        input_size: Input image size (ResNet50 expects 224x224)
    """

    def __init__(
        self,
        device: str = None,
        pretrained: bool = True,
        input_size: int = 224,
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.input_size = input_size

        # Load ResNet50 and remove final FC layer
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        full_model = models.resnet50(weights=weights)
        self.model = nn.Sequential(*list(full_model.children())[:-1])
        self.model = self.model.to(device)
        self.model.eval()

        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(num_output_channels=3),  # CT scans are grayscale
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.feature_dim = 2048

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to ResNet50-ready tensor."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        tensor = self.transform(image)
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract deep features from a single image.

        Args:
            image: 2D grayscale image array

        Returns:
            1D array of 2048 deep features
        """
        tensor = self._preprocess(image).to(self.device)
        features = self.model(tensor)
        features = features.squeeze().cpu().numpy()
        return features

    @torch.no_grad()
    def extract_batch(
        self, images: np.ndarray, batch_size: int = 32
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract deep features from a batch of images.

        Args:
            images: Array of shape (N, H, W)
            batch_size: Number of images per forward pass

        Returns:
            Tuple of:
                - feature_matrix: Array of shape (N, 2048)
                - feature_names: List of feature names
        """
        n_images = len(images)
        feature_matrix = np.zeros((n_images, self.feature_dim))

        for start in range(0, n_images, batch_size):
            end = min(start + batch_size, n_images)
            batch = images[start:end]
            tensors = torch.cat([self._preprocess(img) for img in batch])
            tensors = tensors.to(self.device)
            features = self.model(tensors)
            features = features.squeeze(-1).squeeze(-1).cpu().numpy()
            feature_matrix[start:end] = features

        feature_names = [f"resnet50_feat_{i}" for i in range(self.feature_dim)]
        return feature_matrix, feature_names


if __name__ == "__main__":
    extractor = ResNet50FeatureExtractor()
    print(f"Device: {extractor.device}")

    dummy_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    features = extractor.extract(dummy_image)
    print(f"Extracted {features.shape[0]} ResNet50 features from single image")

    dummy_batch = np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8)
    batch_features, names = extractor.extract_batch(dummy_batch)
    print(f"Batch features shape: {batch_features.shape}")
    print(f"Sample feature names: {names[:3]}")
