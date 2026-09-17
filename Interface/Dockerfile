# syntax=docker/dockerfile:1.7

# BUILD STAGE
# On construit sur `slim` car ce projet contient des dépendances scientifiques lourdes
# et des librairies native qui compilent plus facilement et plus stablement sur Debian slim.

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Ce stage sert à créer un venv propre contenant les dépendances métier.
RUN python -m venv "$VIRTUAL_ENV" && \
    pip install --upgrade pip setuptools wheel

COPY Interface/requirements_interface.txt ./requirements_interface.txt

RUN pip install --no-cache-dir -r requirements_interface.txt

# 2) RUNTIME STAGE
# L'image finale reste légère tout en restant compatible avec les dépendances web/ML.

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libstdc++6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -g 10001 app && \
    useradd -u 10001 -g app -m -d /home/app app && \
    mkdir -p /app/.streamlit && \
    chown -R app:app /app

COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --chown=app:app ./Interface /app/interface

WORKDIR /app/interface

USER 10001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2)' || exit 1

EXPOSE 8501

STOPSIGNAL SIGTERM

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]