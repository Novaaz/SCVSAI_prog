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

run-eda:
	python -m src.eda
	@echo "EDA completata. Visualizzazioni salvate nella cartella data/"

train:
	python -m src.train
	@echo "Training completato. Modello salvato nella cartella models/"

predict:
	@read -p "Inserisci il path dell'immagine: " IMAGE_PATH; \
	python -c "from src.train import predict_image; print(f'Classe predetta: {predict_image(\"$$IMAGE_PATH\")}')"

# Workflow completo
workflow: install run-eda train test
	@echo "Workflow completo: installazione, EDA, training e test eseguiti."

# Training veloce per debug
quick-train:
	python -c "from src.train import train_model; train_model(epochs=5)"
	@echo "Quick training completato (5 epoche)."

# Visualizza metriche
show-results:
	@echo "=== RISULTATI TRAINING ==="
	@if [ -f "models/best_model.pth" ]; then echo "✓ Modello salvato: models/best_model.pth"; else echo "✗ Nessun modello trovato"; fi
	@if [ -f "models/training_plots.png" ]; then echo "✓ Grafici salvati: models/training_plots.png"; else echo "✗ Nessun grafico trovato"; fi
	@if [ -f "data/class_distribution.png" ]; then echo "✓ EDA completata: data/class_distribution.png"; else echo "✗ EDA non eseguita"; fi

# Cleanup completo
deep-clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache __pycache__ 
	rm -rf models/ data/ .coverage
	mkdir -p models data
	@echo "Deep clean complete. Directories recreated."