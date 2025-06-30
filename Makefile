install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	@echo "Installation complete. You can now run the project."

lint:
	python -m pylint --disable=R,C,W0401,W0614 src/*.py tests/*.py
	@echo "Linting complete (import warnings disabled)."

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "Testing complete."

test-verbose:
	python -m pytest tests/ -v -s --cov=src --cov-report=term-missing
	@echo "Verbose testing complete."

test-train:
	python -m pytest tests/test_train.py -v --cov=src.train --cov-report=term-missing
	@echo "Train testing complete."

test-eda:
	python -m pytest tests/test_eda.py -v --cov=src.eda --cov-report=term-missing
	@echo "EDA testing complete."

build:
	python -m build
	@echo "Build complete. Check the dist/ directory."

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache __pycache__ models/*.pth data/*.png
	@echo "Clean complete."

eda:
	python -m src.eda
	@echo "EDA completed. Visualizations saved in the data/ folder."

train:
	python -m src.train
	@echo "Training completed. Model saved in the models/ directory."

predict:
	@read -p "Inserisci il path dell'immagine: " IMAGE_PATH; \
	python -c "from src.train import predict_image; print(f'Classe predetta: {predict_image(\"$$IMAGE_PATH\")}')"

# Workflow completo
workflow: install eda train test show-results
	@echo "Complete workflow: installation, EDA, training and testing executed."

# Training veloce per debug
quick-train:
	python -c "from src.train import train_model; train_model(epochs=5)"
	@echo "Quick training completed (5 epochs)."

# Visualizza metriche
show-results:
	@echo "=== TRAINING RESULTS ==="
	@if [ -f "models/best_model.pth" ]; then echo "✓ Model saved: models/best_model.pth"; else echo "✗ No model found"; fi
	@if [ -f "models/training_plots.png" ]; then echo "✓ Plots saved: models/training_plots.png"; else echo "✗ No plots found"; fi
	@if [ -f "data/class_distribution.png" ]; then echo "✓ EDA completed: data/class_distribution.png"; else echo "✗ EDA not performed"; fi

# Cleanup completo
deep-clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache __pycache__ 
	rm -rf models/ data/ .coverage
	rm -rf src/__pycache__ tests/__pycache__
	mkdir -p models data
	@echo "Deep clean complete. Directories recreated."

help:
	@echo "Available commands:"
	@echo "  install          - Install dependencies"
	@echo "  lint             - Run linter"
	@echo "  test             - Run tests"
	@echo "  test-verbose     - Run verbose tests"
	@echo "  test-train       - Test training module"
	@echo "  test-eda         - Test EDA module"
	@echo "  build            - Build the project"
	@echo "  clean            - Clean build artifacts"
	@echo "  eda              - Run EDA analysis"
	@echo "  train            - Train the model"
	@echo "  predict          - Predict class of an image"
	@echo "  workflow         - Run complete workflow (install, eda, train, test)"
	@echo "  quick-train      - Quick training for debugging (5 epochs)"
	@echo "  show-results     - Show training results and EDA outputs"
	@echo "  deep-clean       - Deep clean all artifacts and recreate directories"