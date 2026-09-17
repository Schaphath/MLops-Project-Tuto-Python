<<<<<<< Updated upstream
# ÉTAPE 1 : BUILDER (Installation isolée)   #

FROM python:3.12-slim AS builder

WORKDIR /app

# Copie uniquement le requirements-prod.txt
COPY requirements-prod.txt .

# Installation des paquets dans un dossier local isolé 
# --no-cache-dir évite de stocker des fichiers temporaires et réduit la taille
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements-prod.txt


# ÉTAPE 2 : RUNTIME 
FROM python:3.12-slim AS runtime

WORKDIR /app

# Création d'un utilisateur non-root pour l'exécution
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Récupération uniquement des dépendances installées à l'étape 1
COPY --from=builder /root/.local /home/appuser/.local

# Copie du code de l'API et le dossier Save_models
COPY --chown=appuser:appuser ./Api/api.py .
COPY --chown=appuser:appuser ./Save_models/ ./save_models/

# Configuration des variables d'environnement 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Passage sous l'utilisateur sécurisé
USER appuser

# Port d'écoute de l'API
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
=======
# syntax=docker/dockerfile:1.7

# -----------------------------------------------------------------------------
# 1) BUILD STAGE
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

RUN python -m venv "$VIRTUAL_ENV" && \
    pip install --upgrade pip setuptools wheel

COPY Api/requirements_api.txt ./requirements_api.txt

RUN pip install --no-cache-dir -r requirements_api.txt

# -----------------------------------------------------------------------------
# 2) RUNTIME STAGE
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -g 10001 app && \
    useradd -u 10001 -g app -m -d /home/app app && \
    mkdir -p /app/api && \
    chown -R app:app /app

COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --chown=app:app ./Api /app/api

WORKDIR /app/api

USER 10001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)' || exit 1

EXPOSE 8000

STOPSIGNAL SIGTERM

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
>>>>>>> Stashed changes
