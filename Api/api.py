"""API REST de prédiction d'une tumeur maligne (OncoScan)."""

import csv
import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

# --- Logging ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

logger = logging.getLogger(__name__)

# --- Configuration & Artefacts ---
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR.parent / "Save_models" / "adaboost_pipeline.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
API_KEY = os.getenv("API_KEY")
MODEL_VERSION = "2.0.0"

# Dossier pour les logs d'inférence (nécessaires pour Evidently AI)
INFERENCE_LOG_DIR = BASE_DIR / "logs"
INFERENCE_LOG_DIR.mkdir(exist_ok=True)
INFERENCE_LOG_FILE = INFERENCE_LOG_DIR / "inference_logs.csv"

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
        extra="forbid",
        allow_inf_nan=False,
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
    """Sauvegarde asynchrone des inférences en CSV pour analyse ultérieure par Evidently AI."""
    try:
        file_exists = INFERENCE_LOG_FILE.exists()
        with open(INFERENCE_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                # Écrire l'entête lors du premier appel
                header = ["timestamp", "model_version"] + FEATURE_ORDER + ["prediction", "probability_malignant"]
                writer.writerow(header)

            row = [
                datetime.now(timezone.utc).isoformat(),
                MODEL_VERSION,
                *[data.get(feat) for feat in FEATURE_ORDER],
                prediction,
                probability if probability is not None else "",
            ]
            writer.writerow(row)
    except Exception as err:
        logger.error("Erreur lors de la sauvegarde de l'inférence : %s", err)


def load_pipeline(path: Path):
    if not path.is_file():
        raise RuntimeError(f"Artefact Joblib introuvable : {path}")
    try:
        return joblib.load(path)
    except Exception as error:
        raise RuntimeError(f"Erreur de chargement du pipeline {path.name}") from error


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le pipeline Joblib au démarrage."""
    logger.info(f"Chargement du pipeline depuis : {MODEL_PATH}")
    app.state.pipeline = load_pipeline(MODEL_PATH)
    logger.info("Pipeline chargé avec succès.")
    yield
    app.state.pipeline = None


# --- FastAPI App ---
app = FastAPI(
    title="OncoScan API",
    version=MODEL_VERSION,
    lifespan=lifespan,
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
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )


# --- Sécurité ---
def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and (not x_api_key or not secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou manquante",
        )


# --- Endpoints ---
@app.get("/", tags=["Info"])
def root() -> dict[str, str]:
    return {"name": "OncoScan API", "version": MODEL_VERSION}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(request: Request) -> HealthResponse:
    pipeline_loaded = getattr(request.app.state, "pipeline", None) is not None
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
        # 1. Préparation du DataFrame
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

        # 3. Calcul de la probabilité
        probability = None
        if hasattr(pipeline, "predict_proba"):
            classes = list(pipeline.classes_)
            pos_idx = classes.index(1) if 1 in classes else classes.index("M")
            probas = pipeline.predict_proba(features_df)[0]
            probability = round(float(probas[pos_idx]), 4)

        # 4. Mise à jour des métriques Prometheus métiers
        PREDICTION_COUNTER.labels(prediction_label=pred_label).inc()
        if probability is not None:
            PROBABILITY_HISTOGRAM.observe(probability)

        for feature in FEATURE_ORDER:
            FEATURE_GAUGES[feature].set(features_dict[feature])

        # 5. Enregistrement asynchrone des données pour Evidently AI
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
        logger.exception("Erreur lors de l'inférence : %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de la prédiction",
        ) from error