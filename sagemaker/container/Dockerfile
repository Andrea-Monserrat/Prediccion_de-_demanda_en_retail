FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/log/nginx

RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    joblib \
    numpy \
    flask \
    gunicorn

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY retail_forecast /opt/program
WORKDIR /opt/program
