"""API REST de prédiction d'une tumeur maligne (OncoScan)."""

import csv
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

# --- Logging ---
# .upper() : accepte "info" comme "INFO" (une valeur invalide ferait planter le démarrage)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration (tout vient des variables d'environnement) ---
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR.parent / "Save_models" / "adaboost_pipeline.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
API_KEY = os.getenv("API_KEY")
MODEL_VERSION = os.getenv("MODEL_VERSION", "2.0.0")

# Pages /docs et /redoc : désactivées par défaut en production (ENABLE_DOCS=true pour les activer)
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "false").lower() == "true"

# Dossier des logs d'inférence (nécessaires pour Evidently AI).
# En Docker : /app/logs, un volume dans lequel l'utilisateur non-root peut écrire.
INFERENCE_LOG_DIR = Path(os.getenv("INFERENCE_LOG_DIR", BASE_DIR / "logs"))
INFERENCE_LOG_FILE = INFERENCE_LOG_DIR / "inference_logs.csv"
try:
    INFERENCE_LOG_DIR.mkdir(parents=True, exist_ok=True)
except OSError as err:
    # L'API doit continuer à prédire même si on ne peut pas écrire les logs
    logger.warning("Dossier de logs d'inférence inutilisable (%s) : %s", INFERENCE_LOG_DIR, err)

# Verrou : empêche deux requêtes d'écrire en même temps dans le CSV
_log_lock = threading.Lock()

# Ordre des colonnes attendu par le modèle (ne pas modifier sans réentraîner)
FEATURE_ORDER = [
    "perimeter",
    "concave_points",
    "texture",
    "smoothness",
    "symmetry",
]

# --- Métriques Prometheus Métier / Data Drift ---
PREDICTION_COUNTER = Counter(
    "oncoscan_predictions_total",
    "Nombre total de prédictions par classe",
    ["prediction_label"],
)

PROBABILITY_HISTOGRAM = Histogram(
    "oncoscan_prediction_probability",
    "Distribution des probabilités de malignité prédites",
    buckets=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
)

# Attention : un Gauge ne garde que la DERNIÈRE valeur reçue (peu fiable si plusieurs
# requêtes arrivent en même temps). Pour le drift, préférez les logs CSV / Histogram.
FEATURE_GAUGES = {
    feature: Gauge(
        f"oncoscan_feature_latest_{feature}",
        f"Dernière valeur reçue pour la feature {feature}",
    )
    for feature in FEATURE_ORDER
}


# --- Schemas Pydantic ---
class PredictionRequest(BaseModel):
    """Features d'entrée requises par le pipeline."""

    model_config = ConfigDict(
        extra="forbid",  # refuse tout champ inconnu
        allow_inf_nan=False,  # refuse NaN et infini
        json_schema_extra={
            "examples": [
                {
                    "perimeter": 87.46,
                    "concave_points": 0.04781,
                    "texture": 17.77,
                    "smoothness": 0.08474,
                    "symmetry": 0.1812,
                }
            ]
        },
    )

    perimeter: float = Field(..., gt=40.0, le=250.0)
    concave_points: float = Field(..., ge=0.0, le=0.3)
    texture: float = Field(..., gt=5.0, le=50.0)
    smoothness: float = Field(..., gt=0.01, le=0.3)
    symmetry: float = Field(..., gt=0.05, le=0.7)


class PredictionResponse(BaseModel):
    prediction: Literal["M", "B"]
    label: Literal["Maligne", "Bénigne"]
    probability_malignant: float | None = Field(default=None, ge=0.0, le=1.0)
    model_version: str = MODEL_VERSION


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    pipeline_loaded: bool
    model_version: str = MODEL_VERSION


# --- Helpers & Background Tasks ---
def save_inference_log(data: dict, prediction: str, probability: float | None):
    """Enregistre chaque inférence dans un CSV (analyse ultérieure avec Evidently AI)."""
    try:
        # Le verrou garantit une seule écriture à la fois (pas de lignes mélangées,
        # pas d'en-tête écrit en double)
        with _log_lock:
            file_exists = INFERENCE_LOG_FILE.exists()
            with open(INFERENCE_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if not file_exists:
                    # Premier appel : on écrit l'en-tête
                    header = ["timestamp", "model_version"] + FEATURE_ORDER + [
                        "prediction",
                        "probability_malignant",
                    ]
                    writer.writerow(header)

                writer.writerow(
                    [
                        datetime.now(timezone.utc).isoformat(),
                        MODEL_VERSION,
                        *[data.get(feat) for feat in FEATURE_ORDER],
                        prediction,
                        probability if probability is not None else "",
                    ]
                )
    except Exception as err:
        # Une erreur de log ne doit jamais faire échouer la prédiction
        logger.error("Erreur lors de la sauvegarde de l'inférence : %s", err)


def load_pipeline(path: Path):
    """Charge le modèle depuis le disque (erreur claire si absent ou illisible)."""
    if not path.is_file():
        raise RuntimeError(f"Artefact Joblib introuvable : {path}")
    try:
        return joblib.load(path)
    except Exception as error:
        raise RuntimeError(f"Erreur de chargement du pipeline {path.name}") from error


def find_positive_index(pipeline) -> int | None:
    """Position de la classe 'maligne' dans predict_proba (calculée une seule fois)."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    classes = list(pipeline.classes_)
    for positive in (1, "M"):
        if positive in classes:
            return classes.index(positive)
    raise RuntimeError(f"Classe maligne introuvable dans {classes}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Vérifie la config et charge le pipeline au démarrage (échec rapide si problème)."""
    if not API_KEY:
        # Sans clé, l'API serait ouverte à tout le monde : on refuse de démarrer
        raise RuntimeError("La variable d'environnement API_KEY doit être définie")

    logger.info("Chargement du pipeline depuis : %s", MODEL_PATH)
    app.state.pipeline = load_pipeline(MODEL_PATH)
    app.state.positive_idx = find_positive_index(app.state.pipeline)
    logger.info("Pipeline chargé avec succès.")
    yield
    app.state.pipeline = None


# --- FastAPI App ---
app = FastAPI(
    title="OncoScan API",
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)

# --- Instrumentation Prometheus ---
instrumentator = Instrumentator(
    should_group_status_codes=True,
    excluded_handlers=["/metrics", "/health", "/docs", "/openapi.json"],
)

instrumentator.instrument(app).expose(app, endpoint="/metrics", tags=["Monitoring"])

# --- CORS Middleware ---
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,  # l'API utilise un en-tête X-API-Key, pas de cookies
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )


# --- Sécurité ---
def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Vérifie la clé API envoyée dans l'en-tête X-API-Key."""
    # .encode() : compare_digest plante (erreur 500) sur les caractères non-ASCII
    # si on lui donne du texte ; avec des octets, ça renvoie proprement "faux"
    is_valid = (
        bool(API_KEY)
        and bool(x_api_key)
        and secrets.compare_digest(x_api_key.encode(), API_KEY.encode())
    )
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou manquante",
        )


# --- Endpoints ---
@app.get("/", tags=["Info"])
def root() -> dict[str, str]:
    return {"name": "OncoScan API", "version": MODEL_VERSION}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(request: Request, response: Response) -> HealthResponse:
    """État de l'API. Renvoie 503 si le modèle n'est pas chargé (utile pour Docker)."""
    pipeline_loaded = getattr(request.app.state, "pipeline", None) is not None
    if not pipeline_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="healthy" if pipeline_loaded else "unhealthy",
        pipeline_loaded=pipeline_loaded,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    dependencies=[Depends(require_api_key)],
)
def predict(
    data: PredictionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non disponible",
        )

    try:
        # 1. Préparation du DataFrame (colonnes dans l'ordre attendu par le modèle)
        features_dict = data.model_dump()
        features_df = pd.DataFrame(
            [[features_dict[feature] for feature in FEATURE_ORDER]],
            columns=FEATURE_ORDER,
        )

        # 2. Inférence
        prediction_raw = pipeline.predict(features_df)[0]
        is_malignant = prediction_raw in (1, "M")
        pred_label = "M" if is_malignant else "B"
        human_label = "Maligne" if is_malignant else "Bénigne"

        # 3. Probabilité (la position de la classe "maligne" est calculée au démarrage)
        probability = None
        positive_idx = getattr(request.app.state, "positive_idx", None)
        if positive_idx is not None:
            probas = pipeline.predict_proba(features_df)[0]
            probability = round(float(probas[positive_idx]), 4)

        # 4. Mise à jour des métriques Prometheus métiers
        PREDICTION_COUNTER.labels(prediction_label=pred_label).inc()
        if probability is not None:
            PROBABILITY_HISTOGRAM.observe(probability)

        for feature in FEATURE_ORDER:
            FEATURE_GAUGES[feature].set(features_dict[feature])

        # 5. Enregistrement en tâche de fond (après l'envoi de la réponse)
        background_tasks.add_task(
            save_inference_log,
            data=features_dict,
            prediction=pred_label,
            probability=probability,
        )

        return PredictionResponse(
            prediction=pred_label,
            label=human_label,
            probability_malignant=probability,
        )

    except Exception as error:
        # Le détail technique reste dans les logs, le client reçoit un message générique
        logger.exception("Erreur lors de l'inférence")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de la prédiction",
        ) from error