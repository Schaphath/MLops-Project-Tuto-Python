# syntax=docker/dockerfile:1.7

# Version de Python définie une seule fois 
ARG PYTHON_VERSION=3.12

#---------------#
#  BUILD STAGE  #
#---------------#
# Étape 1 : on installe les dépendances dans un environnement virtuel.
# Bon savoir : image "slim" Debian est compatible avec les wheels précompilés des libs scientifiques 
# (numpy, pandas, scipy...) - Attention : slim n'a PAS de compilateur.
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

# On copie le fichier des dépendances :
# tant qu'il ne change pas, Docker réutilise le cache et le build est rapide
COPY Interface/requirements_interface.txt ./requirements_interface.txt

# Installation des dépendances
RUN pip install -r requirements_interface.txt

#-----------------#
#  RUNTIME STAGE  #
#-----------------#
# Étape 2 : image finale, la seule qui sera déployée en production
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Métadonnées de l'image
LABEL org.opencontainers.image.title="interface" \
      org.opencontainers.image.description="Interface web Streamlit"

# Variables d'environnement pour l'exécution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Bibliothèques système nécessaires + création d'un utilisateur non-root
# UID/GID fixes pour être compatible avec Kubernetes, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libstdc++6 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 app && \
    useradd --uid 10001 --gid app --create-home --home-dir /home/app app

# Récupération de l'environnement virtuel construit à l'étape 1
COPY --from=builder /opt/venv /opt/venv

# Dossier de l'application
WORKDIR /app/interface

# Copie du code de l'interface
COPY Interface/ /app/interface/

# À partir d'ici, plus aucun droit root : le processus tourne avec l'utilisateur app
USER 10001:10001

# Vérification de l'état de l'app
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"]

# Port d'écoute de Streamlit 
EXPOSE 8501

# Signal d'arrêt propre 
STOPSIGNAL SIGTERM

# Lancement de l'interface 
CMD ["streamlit", "run", "app.py"]