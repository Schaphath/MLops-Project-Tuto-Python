# Tests de validation du pipeline ML.
# pytest Tests/test_model.py -v --json-report --json-report-file=Tests_results/results.json


import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "Data" / "process" / "cancer_select.csv"
DEFAULT_MODELS_DIR = BASE_DIR / "Save_models"

DATA_PATH = Path(os.getenv("TEST_DATA_PATH", str(DEFAULT_DATA_PATH)))
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(DEFAULT_MODELS_DIR)))
MODEL_PATH_OVERRIDE = Path(os.getenv("MODEL_PATH")) if os.getenv("MODEL_PATH") else None

SELECTED_FEATURES = [
    "perimeter",
    "concave_points",
    "texture",
    "smoothness",
    "symmetry",
]

SEUIL_RECALL = 0.95
SEUIL_PRECISION = 0.90
SEUIL_AUC = 0.95
SEUIL_F1 = 0.90


def _resolve_repo_path(path: Path, fallback: Path) -> Path:
    """Normalise les chemins relatifs au dossier racine du projet."""
    if path.is_absolute():
        return path
    candidate = BASE_DIR / path
    if candidate.exists():
        return candidate
    return fallback


def _resolve_data_path() -> Path:
    """Retourne le chemin du dataset en tenant compte du contexte d'exécution."""
    return _resolve_repo_path(DATA_PATH, DEFAULT_DATA_PATH)


def _resolve_model_dir() -> Path:
    """Retourne le dossier contenant les modèles sauvegardés."""
    return _resolve_repo_path(MODELS_DIR, DEFAULT_MODELS_DIR)


def _resolve_model_path() -> Path:
    """Retourne le chemin du pipeline Joblib disponible."""
    if MODEL_PATH_OVERRIDE is not None:
        return MODEL_PATH_OVERRIDE if MODEL_PATH_OVERRIDE.is_absolute() else BASE_DIR / MODEL_PATH_OVERRIDE

    model_dir = _resolve_model_dir()
    joblib_files = sorted(
        model_dir.glob("*.joblib"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not joblib_files:
        raise FileNotFoundError(f"Aucun fichier .joblib dans {model_dir}")
    return joblib_files[0]


def _normalize_target(series: pd.Series) -> pd.Series:
    """Convertit la cible en valeurs numériques 0/1."""
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return series.map({"M": 1, "B": 0}).astype(int)
    return series.astype(int)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """Charge et valide le dataset du projet."""
    data_path = _resolve_data_path()
    assert data_path.exists(), f"Dataset introuvable : {data_path}"

    df = pd.read_csv(data_path)
    assert "diagnosis" in df.columns, "Colonne 'diagnosis' absente."
    for feature in SELECTED_FEATURES:
        assert feature in df.columns, f"Feature manquante : {feature}"
    return df


@pytest.fixture(scope="module")
def pipeline():
    """Charge le pipeline Joblib enregistré dans le dépôt."""
    model_path = _resolve_model_path()
    assert model_path.exists(), f"Fichier pipeline introuvable : {model_path}"
    return joblib.load(model_path)


@pytest.fixture(scope="module")
def predictions(dataset, pipeline):
    """Génère le split de validation et les prédictions associées."""
    X = dataset[SELECTED_FEATURES]
    y = _normalize_target(dataset["diagnosis"])

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    y_pred = pipeline.predict(X_test)
    y_proba = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

    return {
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "X_test": X_test,
    }


class TestQualiteDonnees:
    def test_integrite_donnees(self, dataset):
        assert len(dataset) >= 100, "Dataset trop petit."
        assert dataset[SELECTED_FEATURES].isna().sum().sum() == 0, "Valeurs manquantes détectées."

    def test_encodage_cible(self, dataset):
        classes = set(dataset["diagnosis"].dropna().unique())
        assert {"M", "B"}.issubset(classes), "Classes M ou B manquantes."


class TestMetriquesModele:
    def test_recall_critique_medical(self, predictions):
        recall = recall_score(predictions["y_test"], predictions["y_pred"])
        assert recall >= SEUIL_RECALL, f"Recall insuffisant : {recall:.3f}"

    def test_precision(self, predictions):
        precision = precision_score(predictions["y_test"], predictions["y_pred"])
        assert precision >= SEUIL_PRECISION, f"Précision insuffisante : {precision:.3f}"

    def test_f1_score(self, predictions):
        f1 = f1_score(predictions["y_test"], predictions["y_pred"])
        assert f1 >= SEUIL_F1, f"F1-score insuffisant : {f1:.3f}"

    def test_auc_score(self, predictions):
        if predictions["y_proba"] is None:
            pytest.skip("predict_proba non disponible.")
        auc = roc_auc_score(predictions["y_test"], predictions["y_proba"])
        assert auc >= SEUIL_AUC, f"AUC insuffisant : {auc:.3f}"


class TestRobustesseInference:
    def test_inference_mono_exemple(self, pipeline, predictions):
        single_sample = predictions["X_test"].iloc[[0]]
        pred = pipeline.predict(single_sample)
        assert pred[0] in [0, 1], "Format de prédiction invalide."

    def test_determinisme_prediction(self, pipeline, predictions):
        pred_a = pipeline.predict(predictions["X_test"])
        pred_b = pipeline.predict(predictions["X_test"])
        np.testing.assert_array_equal(pred_a, pred_b, err_msg="Prédictions non déterministes.")