# SCVSAI - Fashion-MNIST Classifier 👗🧥👠

Un classificatore di immagini basato su **ResNet18** per il dataset **Fashion-MNIST**, sviluppato per il corso di "Sviluppo e ciclo di vita di software di artificial intelligence".

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Indice

- [Caratteristiche](#-caratteristiche)
- [Struttura del progetto](#-struttura-del-progetto)
- [Installazione](#-installazione)
  - [Metodo 1: Installazione locale](#metodo-1-installazione-locale)
  - [Metodo 2: Docker (Raccomandato)](#metodo-2-docker-raccomandato)
- [Utilizzo](#-utilizzo)
  - [Comandi Make](#comandi-make)
  - [Comandi Docker](#comandi-docker)
- [Dataset](#-dataset)
- [Architettura del modello](#-architettura-del-modello)
- [Testing](#-testing)
- [Sviluppo](#-sviluppo)
- [Risultati](#-risultati)
- [Autore](#-autore)

## 🚀 Caratteristiche

- **Modello**: ResNet18 pre-addestrato con fine-tuning per Fashion-MNIST
- **Dataset**: 60,000 immagini di capi di abbigliamento (10 classi)
- **Analisi EDA**: Analisi esplorativa completa dei dati
- **Training robusto**: Con early stopping e salvataggio del miglior modello
- **Metriche complete**: Accuracy, Precision, Recall, F1-Score per classe
- **Visualizzazioni**: Grafici di training, confusion matrix, distribuzioni
- **Testing**: Suite completa di test con pytest
- **Package Python**: Installabile via pip
- **Docker**: Ambiente completamente containerizzato
- **CI/CD Ready**: Configurazione per GitHub Actions

## 📁 Struttura del progetto

```
progetto_cvs_ai/
├── src/                          # Codice sorgente
│   ├── __init__.py
│   ├── train.py                  # Training e predizione
│   └── eda.py                    # Analisi esplorativa
├── tests/                        # Test suite
│   ├── test_train.py
│   └── test_eda.py
├── models/                       # Modelli salvati (creata automaticamente)
├── data/                         # Visualizzazioni EDA (creata automaticamente)
├── requirements.txt              # Dipendenze Python
├── pyproject.toml               # Configurazione package
├── Dockerfile                   # Container configuration
├── setup_docker.sh             # Script setup Docker
├── Makefile                     # Automazione comandi
└── README.md                    # Questo file
```

## 🛠 Installazione

### Metodo 1: Installazione locale

#### Prerequisiti
- Python 3.8+
- pip
- make (opzionale ma raccomandato)

#### Installazione

```bash
# 1. Clona il repository
git clone https://github.com/yourusername/progetto_cvs_ai.git
cd progetto_cvs_ai

# 2. Installa il package
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 3. Verifica installazione
python -c "from src.train import get_model; print('Package installato correttamente!')"
```

#### Usando Make (raccomandato)

```bash
# Installa tutto automaticamente
make install
```

### Metodo 2: Docker (Raccomandato)

#### Prerequisiti
- Docker Desktop
- Bash/Shell

#### Setup automatico

```bash
# 1. Clona il repository
git clone https://github.com/yourusername/progetto_cvs_ai.git
cd progetto_cvs_ai

# 2. Esegui setup automatico
chmod +x setup_docker.sh
./setup_docker.sh
```

Lo script automaticamente:
- ✅ Verifica installazione Docker
- ✅ Costruisce l'immagine Docker
- ✅ Avvia menu interattivo

#### Setup manuale Docker

```bash
# Build immagine
docker build -t scvsai:latest .

# Verifica build
docker images | grep scvsai
```

## 🎯 Utilizzo

### Comandi Make

```bash
# === ANALISI DATI ===
make eda                    # Analisi esplorativa completa

# === TRAINING ===
make train                  # Training completo (10 epochs)
make quick-train           # Training veloce (5 epochs)

# === TESTING ===
make test                  # Test completi
make test-verbose          # Test con output dettagliato
make test-train            # Solo test di training
make test-eda              # Solo test EDA

# === QUALITY ASSURANCE ===
make lint                  # Linting del codice

# === WORKFLOW ===
make workflow              # EDA + Training + Testing completo
make show-results          # Mostra risultati salvati

# === PACKAGE ===
make build                 # Build package per distribuzione
make clean                 # Pulizia file temporanei
```

### Comandi Docker

#### Setup interattivo
```bash
./setup_docker.sh
# Segui il menu interattivo per:
# 1) Shell Docker interattiva
# 2) EDA analysis
# 3) Training
# 4) Testing
# 5) Workflow completo
```

#### Comandi diretti
```bash
# EDA
docker run --rm -v $(pwd)/data:/app/data scvsai:latest make eda

# Training
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data scvsai:latest make train

# Testing
docker run --rm scvsai:latest make test-direct

# Shell interattiva
docker run --rm -it -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data scvsai:latest /bin/bash
```

### Uso come Python Package

```python
from src.train import get_model, train_model, predict_image
from src.eda import run_eda, load_data, analyze_data

# Analisi esplorativa
run_eda()

# Training
model = train_model(epochs=10)

# Predizione
prediction = predict_image("path/to/image.jpg")
print(f"Classe predetta: {prediction}")

# Caricamento dati
train_loader, val_loader, test_loader = load_data()
```

## 📊 Dataset

**Fashion-MNIST** - Dataset di immagini 28x28 in scala di grigi

### Classi (10 totali):
0. T-shirt/top
1. Trouser
2. Pullover  
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

### Statistiche:
- **Training set**: 60,000 immagini
- **Test set**: 10,000 immagini
- **Dimensione**: 28x28 pixel
- **Formato**: Grayscale (1 canale)
- **Bilanciamento**: 6,000 immagini per classe (training)

## 🧪 Testing

```bash
# Test completi
make test

# Coverage report
make test-verbose

# Test specifici
make test-train    # Solo modulo training
make test-eda      # Solo modulo EDA
```

### Test suite include:
- ✅ **Test unitari**: Singole funzioni
- ✅ **Test integrazione**: Workflow completi
- ✅ **Test modello**: Architettura e output
- ✅ **Test dati**: Caricamento e preprocessing
- ✅ **Test metriche**: Calcoli accuracy/precision/recall
- ✅ **Mock tests**: Simulazione training senza GPU

## 🔧 Sviluppo

### Setup ambiente di sviluppo

```bash
# Linting e formatting
make lint
make format
```

### Struttura codice

```python
# src/train.py - Modulo principale
├── get_model()              # Caricamento ResNet18
├── get_data_loaders()       # Dataset loaders
├── train_one_epoch()        # Training singola epoch
├── validate()               # Validazione modello
├── train_model()            # Training completo
├── evaluate_model()         # Valutazione finale
├── predict_image()          # Predizione singola immagine
└── compute_metrics_pytorch() # Calcolo metriche

# src/eda.py - Analisi esplorativa  
├── load_data()              # Caricamento dati
├── analyze_data()           # Statistiche dataset
├── visualize_data()         # Grafici e visualizzazioni
└── run_eda()               # EDA completa
```

## 📈 Risultati

### Output generati:
```
models/
├── best_model.pth           # Miglior modello salvato
└── training_plots.png       # Grafici loss/accuracy

data/
├── class_distribution.png   # Distribuzione classi
├── sample_images.png        # Esempi per classe
└── data_statistics.txt      # Statistiche numeriche
```

## 🐳 Docker

### Vantaggi dell'approccio Docker:
- ✅ **Ambiente isolato**: Nessun conflitto dipendenze
- ✅ **Riproducibilità**: Stessi risultati ovunque
- ✅ **Facilità deploy**: Un container, ovunque
- ✅ **CI/CD ready**: Perfetto per automazione

### Dockerfile highlights:
```dockerfile
FROM python:3.10-slim
RUN apt-get install make
COPY requirements.txt pyproject.toml ./
COPY src/ tests/ Makefile ./
RUN pip install -e .
```

## 📄 Licenza

Questo progetto è licenziato sotto la Licenza MIT.

## 👨‍💻 Autore

**Novaaz** (Leonardo Novazzi)
- Email: leonardonovazzi@gmail.com
- GitHub: [@Novaaz](https://github.com/Novaaz)

---

## 🎓 Progetto Universitario

Sviluppato per il corso di **"Sviluppo e ciclo di vita di software di artificial intelligence"**

### Tecnologie utilizzate:
- **Machine Learning**: PyTorch, ResNet18, Transfer Learning
- **Data Science**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- **Software Engineering**: Package Python, Testing, Linting, Docker
- **DevOps**: Makefile, Docker, GitHub Actions (ready)
- **Quality Assurance**: Pytest, Coverage, Pylint, Black

---

**⭐ Se questo progetto ti è stato utile, lascia una stella!**