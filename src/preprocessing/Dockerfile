# Dockerfile — Step: preprocessing
FROM python:3.12-slim

# Instalar dependencias de machine learning
RUN pip install --no-cache-dir \
    numpy==2.4.2 \
    pandas==3.0.1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar código fuente
COPY src/ ./src/

# PYTHONPATH para que los imports funcionen (ej: from utils.logging_config import ...)
ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "preprocessing"]
