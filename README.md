# LungNet-Hybrid

**Hybrid GLCM + ResNet50 Feature Fusion for Lung Nodule Prediction with Bayesian-Optimized XGBoost**

[![Paper](https://img.shields.io/badge/Paper-Wiley-blue)](https://doi.org/10.1155/je/1367255)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

Official implementation of:
> **Maqbool, M., Awais, M., Tarif, A., Zafar, N.A., Shim, S., Hussain, L., Waheed, A., Nadeem, M.A.** (2026). Enhancing Lung Nodule Prediction Accuracy: A Data-Driven Approach With Dynamic Feature Extraction. *Journal of Engineering (Wiley)*. DOI: [10.1155/je/1367255](https://doi.org/10.1155/je/1367255)

---

##  Overview

Lung cancer is the leading cause of cancer-related deaths globally. Early nodule detection from CT scans can improve 5-year survival from 4% to over 54%. This work proposes a hybrid feature-space approach that combines:

- **Static texture descriptors**: GLCM, First-Order Statistics (FOS), Histogram features
- **Dynamic deep features**: ResNet50 pretrained on ImageNet
- **Feature selection**: Kruskal-Wallis test selecting top 400 most discriminative features
- **Classification**: XGBoost with Bayesian hyperparameter optimization

##  Key Results (LUNA16 Dataset)

| Metric | Value |
|---|---|
| Accuracy | **97.62%** |
| AUC | **0.985** |
| MCC | **0.949** |
| F1-Score | **0.940** |
| Sensitivity | 97.10% |
| Specificity | 97.94% |

##  Pipeline

1. **Preprocessing**: Adaptive thresholding → ROI extraction → morphological dilation → augmentation
2. **Feature Extraction**: Parallel static (GLCM/FOS/Histogram) + dynamic (ResNet50) feature streams
3. **Feature Fusion**: Concatenation into hybrid feature space (HFS)
4. **Feature Selection**: Kruskal-Wallis ranking → top 400 features
5. **Classification**: XGBoost with Bayesian-optimized hyperparameters

##  Quickstart

### Installation

```bash
git clone https://github.com/mfhrm31/lungnet-hybrid.git
cd lungnet-hybrid
pip install -r requirements.txt
```

### Download LUNA16

The LUNA16 dataset is publicly available at [luna16.grand-challenge.org](https://luna16.grand-challenge.org/).

### Reproduce paper results

```bash
make reproduce
```

This runs the full pipeline: preprocessing → feature extraction → selection → training → evaluation.

## 📂 Project Structure
