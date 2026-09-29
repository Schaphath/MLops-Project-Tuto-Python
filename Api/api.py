"""API REST"""

import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

# Chargement des variables d'environnement
# Charge automatiquement le fichier .env 
load_dotenv()

# Logging 
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

logger = logging.getLogger("oncoscan_api")

# Configuration 
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR.parent / "Save_models" / "adaboost_pipeline.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
# MODEL_VERSION = os.getenv("VERSION")
MODEL_VERSION = os.getenv("MODEL_VERSION") or os.getenv("VERSION") or "2.0.0"

# Récupération de la clé API depuis l'environnement
API_KEY = os.getenv("API_KEY")

# Variables d'entrainement
FEATURE_ORDER = [
    "perimeter",
    "concave_points",
    "texture",
    "smoothness",
    "symmetry"
    ]

# Schemas Pydantic
class PredictionRequest(BaseModel):
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

# Helpers & Lifespan 
def load_pipeline(path: Path):
    if not path.is_file():
        raise RuntimeError(f"Artefact introuvable : {path}")
    try:
        return joblib.load(path)
    except Exception as error:
        raise RuntimeError(f"Erreur lors du chargement de {path.name}") from error

# Trouve l'index de la classe maligne dans predict_proba.
def find_positive_index(pipeline) -> int | None:
    if not hasattr(pipeline, "predict_proba"):
        return None
    classes = list(pipeline.classes_)
    for positive in (1, "M"):
        if positive in classes:
            return classes.index(positive)
    return None

# Vérification de la configuration et chargement du modèle
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError("La variable API_KEY est introuvable")

    logger.info("Chargement du pipeline : %s", MODEL_PATH)
    app.state.pipeline = load_pipeline(MODEL_PATH)
    app.state.positive_idx = find_positive_index(app.state.pipeline)
    logger.info("Pipeline chargé avec succès.")
    yield
    
    app.state.pipeline = None


# FastAPI App 
app = FastAPI(
    title="OncoScan API",
    version=MODEL_VERSION,
    lifespan=lifespan)

# Sécurité
def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    is_valid = (
        bool(API_KEY)
        and bool(x_api_key)
        and secrets.compare_digest(x_api_key.encode(), API_KEY.encode()))
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou manquante")

# Endpoints 
@app.get("/", tags=["Info"])
def info() -> dict[str, str]:
    return {
        "name": "OncoScan API",
        "version": MODEL_VERSION,
        "status": "online"
        }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(request: Request, response: Response) -> HealthResponse:
    pipeline_loaded = getattr(request.app.state, "pipeline", None) is not None

    if not pipeline_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="healthy" if pipeline_loaded else "unhealthy",
        pipeline_loaded=pipeline_loaded)

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    dependencies=[Depends(require_api_key)])

# Inférence sécurisée par la clé API
def predict(data: PredictionRequest, request: Request) -> PredictionResponse:
    pipeline = getattr(request.app.state, "pipeline", None)

    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non disponible")

    try:
        features_dict = data.model_dump()
        features_df = pd.DataFrame(
            [[features_dict[feat] for feat in FEATURE_ORDER]],
            columns=FEATURE_ORDER)

        prediction_raw = pipeline.predict(features_df)[0]
        is_malignant = prediction_raw in (1, "M")
        pred_label = "M" if is_malignant else "B"
        human_label = "Maligne" if is_malignant else "Bénigne"

        probability = None
        positive_idx = getattr(request.app.state, "positive_idx", None)
        if positive_idx is not None:
            probas = pipeline.predict_proba(features_df)[0]
            probability = round(float(probas[positive_idx]), 4)

        return PredictionResponse(
            prediction=pred_label,
            label=human_label,
            probability_malignant=probability)

    except Exception as error:
        logger.exception("Erreur lors de l'inférence")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de la prédiction",
            ) from error