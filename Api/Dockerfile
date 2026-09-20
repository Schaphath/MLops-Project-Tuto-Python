# syntax=docker/dockerfile:1.7

# Version de Python définie une seule fois (utilisée par les 2 étapes)
ARG PYTHON_VERSION=3.12

#---------------#
#  BUILD STAGE  #
#---------------#
# Étape 1 : on installe les dépendances Python dans un environnement virtuel.
# Cette étape ne sera PAS dans l'image finale (image plus légère et plus sûre).
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

# Variables d'environnement pour l'étape de build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Dossier de travail temporaire pour le build
WORKDIR /build

# Création de l'environnement virtuel + mise à jour de pip
RUN python -m venv "$VIRTUAL_ENV" && \
    pip install --upgrade pip

# On copie d'abord SEULEMENT le fichier des dépendances :
# tant qu'il ne change pas, Docker réutilise le cache et le build est rapide
COPY Api/requirements_api.txt ./requirements_api.txt

# Installation des dépendances (le cache pip est déjà désactivé par PIP_NO_CACHE_DIR)
RUN pip install -r requirements_api.txt

#-----------------#
#  RUNTIME STAGE  #
#-----------------#
# Étape 2 : image finale, la seule qui sera déployée en production
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Métadonnées de l'image (pratique pour retrouver qui/quoi dans un registre)
LABEL org.opencontainers.image.title="api" \
      org.opencontainers.image.description="API FastAPI"

# Variables d'environnement pour l'exécution (pip n'est plus utilisé ici)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Installation des bibliothèques système nécessaires + création d'un utilisateur
# non-root (UID/GID fixes pour être compatible avec Kubernetes, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 app && \
    useradd --uid 10001 --gid app --create-home --home-dir /home/app app

# Récupération de l'environnement virtuel construit à l'étape 1
# (propriétaire root + lecture seule pour l'utilisateur app = plus sûr)
COPY --from=builder /opt/venv /opt/venv

# Dossier de l'application (créé automatiquement)
WORKDIR /app/api

# Copie du code de l'API (le .dockerignore filtre les fichiers inutiles/sensibles)
COPY Api/ /app/api/

# Dossier des logs d'inférence : le SEUL endroit où l'API a le droit d'écrire.
# (le code et le venv restent en lecture seule)
RUN mkdir -p /app/logs && chown app:app /app/logs
ENV INFERENCE_LOG_DIR=/app/logs

# À partir d'ici, plus aucun droit root : le processus tourne avec l'utilisateur app
USER 10001:10001

# Vérification de l'état de l'API (l'endpoint /health doit exister dans le code)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# Port d'écoute de l'API (documentation, ne publie pas le port)
EXPOSE 8000

# Signal d'arrêt propre (déjà la valeur par défaut, gardé pour être explicite)
STOPSIGNAL SIGTERM

# Lancement de l'API (format "exec" : uvicorn reçoit directement les signaux d'arrêt)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]