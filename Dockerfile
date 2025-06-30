FROM python:3.10-slim

# Installa make e pulisce la cache di apt
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia requirements
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY Makefile ./

# Installa dipendenze
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .

RUN mkdir -p models data