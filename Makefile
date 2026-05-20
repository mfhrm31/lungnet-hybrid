# LungNet-Hybrid — Reproducibility Makefile
# Companion code for Maqbool et al. (2026), Wiley Journal of Engineering
# DOI: 10.1155/je/1367255

.PHONY: help install download-data preprocess features train evaluate reproduce test clean

help:
	@echo "LungNet-Hybrid — Available Commands"
	@echo "  make install        - Install Python dependencies"
	@echo "  make download-data  - Instructions for downloading LUNA16"
	@echo "  make preprocess     - Run preprocessing pipeline"
	@echo "  make features       - Extract hybrid features (GLCM + FOS + Hist + ResNet50)"
	@echo "  make train          - Train XGBoost with Bayesian optimization"
	@echo "  make evaluate       - Compute evaluation metrics"
	@echo "  make reproduce      - Run full pipeline (preprocess + features + train + evaluate)"
	@echo "  make test           - Run unit tests"
	@echo "  make clean          - Remove cached files and results"

install:
	pip install -r requirements.txt

download-data:
	@echo "LUNA16 must be downloaded manually due to size and license."
	@echo "Visit: https://luna16.grand-challenge.org/"
	@echo "After downloading, place files in: data/luna16/"

preprocess:
	python scripts/preprocess.py --config configs/default.yaml

features:
	python scripts/extract_features.py --config configs/default.yaml

train:
	python scripts/train.py --config configs/default.yaml

evaluate:
	python scripts/evaluate.py --config configs/default.yaml

reproduce: preprocess features train evaluate
	@echo "Reproduction complete. Results saved in results/"

test:
	pytest tests/ -v

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ */*/*/__pycache__
	rm -rf .pytest_cache
	rm -rf results/*.csv results/*.png
	@echo "Cleaned cache and result files"
