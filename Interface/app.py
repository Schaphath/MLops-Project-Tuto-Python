"""Interface web"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import bcrypt
import pandas as pd
import psycopg2
import requests
import streamlit as st
from psycopg2 import pool
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# Paramètres globaux de l'application
APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
MAX_HISTORY_ROWS = 50

# Protection contre les tentatives de connexion répétées
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 60
MAX_TRACKED_USERNAMES = 10_000

# Règles sur les identifiants et mots de passe
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72
MAX_PATIENT_ID_LENGTH = 50
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,30}$")

# Libellés affichés pour le diagnostic
LABEL_MALIGNANT = "Malignant"
LABEL_BENIGN = "Benign"

# Noms des variables du modèle
FEATURES = ["perimeter", "concave_points", "texture", "smoothness", "symmetry"]

FEATURE_BOUNDS = {
    "perimeter": (40.1, 250.0, 90.0, 1.0),
    "concave_points": (0.0, 0.30, 0.05, 0.001),
    "texture": (5.1, 50.0, 17.0, 0.1),
    "smoothness": (0.011, 0.30, 0.09, 0.001),
    "symmetry": (0.051, 0.70, 0.18, 0.001),
}

FEATURE_HELP = {
    "perimeter": "Périmètre total de la tumeur observée",
    "concave_points": "Nombre de points concaves présents sur le contour tumoral.",
    "texture": "Variation du niveau de gris dans la zone tumorale.",
    "smoothness": "Variation locale de la longueur du contour tumoral.",
    "symmetry": "Degré d'asymétrie de la forme tumorale.",
}

# Chargement du style CSS externe (même dossier que ce fichier)
CSS_PATH = Path(__file__).resolve().parent / "style.css"

def load_css(css_file_path: Path) -> None:
    """Lit le fichier CSS externe et l'injecte dans l'application Streamlit.
    L'injection se fait via st.markdown(..., unsafe_allow_html=True)"""
    try:
        css = css_file_path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Fichier CSS introuvable ou illisible : %s", css_file_path)
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Configuration (variables d'environnement)
@dataclass(frozen=True)
class Settings:
    api_url: str
    api_key: str
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    allow_registration: bool
    timezone: str

@st.cache_resource
def get_settings() -> Settings:
    api_key = os.getenv("API_KEY", "")
    db_password = os.getenv("DB_PASSWORD", "")
    missing = [
        name
        for name, value in (("API_KEY", api_key), ("DB_PASSWORD", db_password))
        if not value
        ]
    
    if missing:
        raise RuntimeError(f"Variable(s) d'environnement manquante(s) : {', '.join(missing)}")

    return Settings(
        api_url=os.getenv("API_URL", "http://api:8000/predict"),
        api_key=api_key,
        db_host=os.getenv("DB_HOST", "postgres"),
        db_port=int(os.getenv("DB_PORT", "5432")),
        db_name=os.getenv("DB_NAME", "oncoscan"),
        db_user=os.getenv("DB_USER", "oncoscan"),
        db_password=db_password,
        allow_registration=os.getenv("ALLOW_REGISTRATION", "true").lower() == "true",
        timezone=os.getenv("APP_TIMEZONE", "Europe/Paris"),
    )

# Base de données PostgreSQL
@st.cache_resource
def get_connection_pool(settings: Settings) -> pool.ThreadedConnectionPool:
    return pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        connect_timeout=5
    )

@contextmanager
def db_cursor(settings: Settings) -> Iterator[Any]:
    connection_pool = get_connection_pool(settings)
    connection = connection_pool.getconn()
    try:
        with connection.cursor() as cursor:
            yield cursor
        connection.commit()
    except Exception:
        if not connection.closed:
            connection.rollback()
        raise
    finally:
        connection_pool.putconn(connection, close=bool(connection.closed))

# Mots de passe
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False

@st.cache_resource
def get_dummy_hash() -> str:
    return hash_password("mot-de-passe-factice-1")

def valid_username(username: str) -> bool:
    return bool(USERNAME_PATTERN.fullmatch(username))

def valid_password(password: str) -> bool:
    return (
        MIN_PASSWORD_LENGTH <= len(password)
        and len(password.encode("utf-8")) <= MAX_PASSWORD_BYTES
        and any(character.isalpha() for character in password)
        and any(character.isdigit() for character in password)
    )

@st.cache_resource
def get_login_tracker() -> dict[str, Any]:
    return {"lock": threading.Lock(), "entries": {}}

def lockout_remaining(username: str) -> int:
    tracker = get_login_tracker()
    with tracker["lock"]:
        entry = tracker["entries"].get(username)
        if entry is None:
            return 0
        return max(0, math.ceil(entry["locked_until"] - time.time()))

def record_failed_login(username: str) -> None:
    tracker = get_login_tracker()
    now = time.time()
    with tracker["lock"]:
        entries = tracker["entries"]
        entry = entries.get(
            username, {"failures": 0, "last_failure": now, "locked_until": 0.0}
        )
        if now - entry["last_failure"] > LOCKOUT_SECONDS:
            entry["failures"] = 0

        entry["failures"] += 1
        entry["last_failure"] = now

        if entry["failures"] >= MAX_LOGIN_ATTEMPTS:
            entry["locked_until"] = now + LOCKOUT_SECONDS
            entry["failures"] = 0

        entries[username] = entry

        if len(entries) > MAX_TRACKED_USERNAMES:
            stale = [
                name
                for name, item in entries.items()
                if now - item["last_failure"] > LOCKOUT_SECONDS
                and item["locked_until"] < now
            ]
            for name in stale:
                del entries[name]

def clear_failed_logins(username: str) -> None:
    tracker = get_login_tracker()
    with tracker["lock"]:
        tracker["entries"].pop(username, None)

# Session et comptes utilisateurs
def initialise_session() -> None:
    defaults = {
        "authenticated": False,
        "username": None,
        "user_id": None,
        "last_prediction": None
        }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

def login(settings: Settings, username: str, password: str) -> tuple[bool, str]:
    remaining = lockout_remaining(username)
    if remaining > 0:
        return False, f"Trop de tentatives. Réessayez dans {remaining} seconde(s)."

    try:
        with db_cursor(settings) as cursor:
            cursor.execute(
                "SELECT id, password_hash FROM users WHERE username = %s",
                (username,))
            row = cursor.fetchone()

    except (psycopg2.Error, RuntimeError):
        logger.exception("Échec de lecture de l'utilisateur")
        return False, "Le service de comptes est temporairement indisponible."

    password_ok = verify_password(password, row[1] if row else get_dummy_hash())

    if row and password_ok:
        clear_failed_logins(username)
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.user_id = row[0]
        return True, "Connexion réussie."

    record_failed_login(username)
    return False, "Identifiant ou mot de passe incorrect."

def register_user(settings: Settings, username: str, password: str) -> tuple[bool, str]:
    try:
        with db_cursor(settings) as cursor:
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, hash_password(password)),
            )
        return True, "Compte créé. Vous pouvez maintenant vous connecter."
    except psycopg2.errors.UniqueViolation:
        return False, "Cet identifiant existe déjà."
    except (psycopg2.Error, RuntimeError):
        logger.exception("Échec de création du compte")
        return False, "Le service de comptes est temporairement indisponible."

# Appel de l'API & Historique
def request_prediction(settings: Settings, inputs: dict[str, float]) -> dict[str, Any]:
    response = requests.post(
        settings.api_url,
        json=inputs,
        headers={"X-API-Key": settings.api_key},
        timeout=(3, 15),
    )
    if response.status_code != 200:
        if response.status_code in (401, 403):
            raise RuntimeError("L'API refuse la requête.")
        if response.status_code == 422:
            raise ValueError("Les valeurs saisies sont hors des limites acceptées.")
        if response.status_code == 503:
            raise RuntimeError("Le modèle est indisponible.")
        raise RuntimeError(
            f"Le service de prédiction a répondu {response.status_code}."
            )

    result = response.json()
    if not isinstance(result, dict) or result.get("prediction") not in ("M", "B"):
        raise RuntimeError("Réponse invalide du service de prédiction.")
    return result

def save_prediction(
    settings: Settings,
    user_id: int,
    patient_id: str,
    inputs: dict[str, float],
    result: dict[str, Any]
    ) -> None:
    probability = result.get("probability_malignant")
    probability_value = round(float(probability), 4) if probability is not None else None
    values = [inputs[feature] for feature in FEATURES]
    with db_cursor(settings) as cursor:
        cursor.execute(
            """
            INSERT INTO predictions (
                user_id, patient_id, perimeter, concave_points, texture,
                smoothness, symmetry, prediction, probability_malignant
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                patient_id,
                *values,
                result["prediction"],
                probability_value,
            ),
        )

def get_history(settings: Settings, user_id: int) -> pd.DataFrame:
    with db_cursor(settings) as cursor:
        cursor.execute(
            """
            SELECT created_at, prediction, probability_malignant,
                   perimeter, concave_points, texture,
                   smoothness, symmetry
            FROM predictions
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, MAX_HISTORY_ROWS),
        )
        rows = cursor.fetchall()
    columns = [
        "Date & Time",
        "Diagnosis",
        "Probability (%)",
        *[feature.replace("_", " ").title() for feature in FEATURES],
    ]
    return pd.DataFrame(rows, columns=columns)

# Composants d'Affichage
def render_prediction(result: dict[str, Any]) -> None:
    prediction = result["prediction"]
    probability = result.get("probability_malignant")
    probability_text = (
        f"{float(probability) * 100:.2f} %" if probability is not None else "non disponible"
        )

    if prediction == "M":
        st.error("Résultat du modèle : classe maligne")
        st.warning(
            f"Probabilité estimée : {probability_text}. "
            "Ce résultat doit être confirmé par un professionnel de santé."
            )
    else:
        st.success("Résultat du modèle : classe bénigne")
        st.info(f"Probabilité estimée : {probability_text}.")
    st.caption(f"Version du modèle : {result.get('model_version', 'inconnue')}")

def render_login_form(settings: Settings) -> None:
    with st.form("login_form"):
        username = st.text_input(
            "Identifiant", placeholder="Votre identifiant", key="login_username"
            )
        password = st.text_input(
            "Mot de passe",
            type="password",
            placeholder="Votre mot de passe",
            key="login_password")
        
        submitted = st.form_submit_button(
            "Se connecter", type="primary", use_container_width=False
            )
    if submitted:
        if not username.strip() or not password:
            st.warning("Saisissez votre identifiant et votre mot de passe.")
        else:
            success, message = login(settings, username.strip().lower(), password)
            (st.success if success else st.error)(message)
            if success:
                st.rerun()

def render_register_form(settings: Settings) -> None:
    with st.form("register_form"):
        username = st.text_input(
            "Nouvel identifiant", placeholder="ex. dr.martin", key="register_username"
        )
        password = st.text_input(
            "Mot de passe",
            type="password",
            placeholder="8 caractères minimum",
            key="register_password",
        )
        confirmation = st.text_input(
            "Confirmez le mot de passe", type="password", key="register_confirmation"
        )
        # Texte aligné sur USERNAME_PATTERN (3 à 30 caractères)
        st.caption(
            "Identifiant : 3 à 30 caractères (lettres, chiffres, _ . -). "
            "Le mot de passe doit contenir une lettre et un chiffre."
        )
        submitted = st.form_submit_button("Créer le compte", use_container_width=False)
    if submitted:
        username = username.strip().lower()
        if not valid_username(username):
            st.warning(
                "Identifiant invalide : 3 à 30 caractères "
                "(lettres, chiffres, _ . -), sans espace."
            )
        elif password != confirmation:
            st.warning("Les mots de passe ne correspondent pas.")
        elif not valid_password(password):
            st.warning(
                "Le mot de passe doit contenir au moins 8 caractères, "
                "dont une lettre et un chiffre (72 octets maximum)."
            )
        else:
            success, message = register_user(settings, username, password)
            (st.success if success else st.error)(message)

def render_login(settings: Settings) -> None:
    """Écran de connexion."""
    st.markdown(
        '<div class="auth-brand"><span class="brand-mark">O</span><span>OncoScan AI</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="auth-heading">'
        '<p class="auth-kicker">ESPACE PROFESSIONNEL</p>'
        '<h2 class="auth-title">Bienvenue dans votre espace</h2>'
        "</div>",
        unsafe_allow_html=True,
    )
    if settings.allow_registration:
        login_tab, register_tab = st.tabs(["Connexion", "Créer un compte"])
        with login_tab:
            render_login_form(settings)
        with register_tab:
            render_register_form(settings)
    else:
        render_login_form(settings)

def render_analysis(settings: Settings) -> None:
    """Saisie des mesures et lancement de la prédiction."""
    st.markdown(
        '<p class="analysis-subtitle">Entrez les informations cliniques et obtenez une prédiction.</p>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        patient_id = st.text_input(
            "Identifiant patient",
            placeholder="ex. PAT-001",
            help="Identifiant patient requis pour l'historique.",
            max_chars=MAX_PATIENT_ID_LENGTH,
        )
        inputs: dict[str, float] = {}
        columns = st.columns(2)
        for index, feature in enumerate(FEATURES):
            minimum, maximum, default, step = FEATURE_BOUNDS[feature]
            with columns[index % 2]:
                inputs[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                    format="%.3f",
                    help=FEATURE_HELP[feature],
                )
        submitted = st.form_submit_button(
            "Lancer l'analyse", type="primary", use_container_width=False
        )

    if submitted:
        if not patient_id.strip():
            st.warning("Saisissez un identifiant patient pour enregistrer l'analyse.")
        else:
            with st.spinner("Analyse en cours..."):
                try:
                    result = request_prediction(settings, inputs)
                    st.session_state.last_prediction = result
                    try:
                        save_prediction(
                            settings,
                            st.session_state.user_id,
                            patient_id.strip(),
                            inputs,
                            result,
                        )
                    except (psycopg2.Error, RuntimeError):
                        logger.exception("Échec de sauvegarde")
                        st.warning(
                            "Analyse terminée, mais l'historique n'a pas pu être sauvegardé."
                        )
                except requests.exceptions.Timeout:
                    st.error("Le service de prédiction met trop de temps à répondre.")
                except requests.exceptions.RequestException:
                    logger.exception("Échec de communication avec l'API")
                    st.error("Impossible de contacter le service API.")
                except (RuntimeError, ValueError) as error:
                    st.error(str(error))

    if st.session_state.last_prediction:
        render_prediction(st.session_state.last_prediction)

def render_history(settings: Settings) -> None:
    """Affiche l'historique des prédictions."""
    try:
        history = get_history(settings, st.session_state.user_id)
    except (psycopg2.Error, RuntimeError):
        logger.exception("Échec de lecture de l'historique")
        st.error("Impossible de charger l'historique pour le moment.")
        return
    if history.empty:
        st.info("Aucune analyse enregistrée pour le moment.")
        return

    history["Diagnosis"] = history["Diagnosis"].map(
        {"M": LABEL_MALIGNANT, "B": LABEL_BENIGN}
    )
    history["Probability (%)"] = pd.to_numeric(history["Probability (%)"], errors="coerce") * 100

    col_filter, _ = st.columns([1, 1.8])
    with col_filter:
        choice = st.selectbox(
            "Filter", ["All", LABEL_MALIGNANT, LABEL_BENIGN], key="history_filter"
        )
    if choice != "All":
        history = history[history["Diagnosis"] == choice]

    show_details = st.checkbox("Show clinical measurements", key="history_details")
    visible_columns = ["Date & Time", "Diagnosis", "Probability (%)"]
    if show_details:
        visible_columns += [
            column for column in history.columns if column not in visible_columns
        ]

    visible_history = history[visible_columns].copy()
    visible_history["Date & Time"] = (
        pd.to_datetime(visible_history["Date & Time"], utc=True)
        .dt.tz_convert(settings.timezone)
        .dt.strftime("%d/%m/%Y %H:%M")
    )
    visible_history["Probability (%)"] = visible_history["Probability (%)"].map(
        lambda value: f"{value:.2f} %" if pd.notna(value) else "-"
    )

    table = visible_history.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#f8eef1"),
                    ("color", "#8f4b60"),
                    ("font-weight", "600"),
                    ("font-size", "0.72rem"),
                    ("border-bottom", "1px solid #ead6dc"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#fffafb"),
                    ("color", "#30242a"),
                    ("font-size", "0.75rem"),
                    ("border-bottom", "1px solid #f1e3e7"),
                ],
            },
        ]
    ).set_properties(
        **{"text-align": "left", "padding": "0.4rem", "font-size": "0.75rem"}
    )
    st.table(table)
    st.download_button(
        "Télécharger (CSV)",
        data=visible_history.to_csv(index=False).encode("utf-8-sig"),
        file_name="oncoscan_history.csv",
        mime="text/csv",
        use_container_width=False,
    )

def render_dashboard(settings: Settings) -> None:
    """Tableau de bord principal."""
    with st.sidebar:
        st.markdown(
            '<div class="app-brand"><span class="app-brand-mark">O</span><span>OncoScan</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Session : {st.session_state.username}")
        st.divider()
        if st.button("Se déconnecter", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    st.markdown(
        '<div class="dashboard-heading">'
        '<p class="auth-kicker">ESPACE PROFESSIONNEL</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    analysis_tab, history_tab = st.tabs(["Nouvelle analyse", "Historique"])
    with analysis_tab:
        render_analysis(settings)
    with history_tab:
        render_history(settings)

# Point d'entrée
def main() -> None:
    logo_path = Path(__file__).resolve().parent / "favicon.svg"
    st.set_page_config(
        page_title="OncoScan",
        page_icon=str(logo_path) if logo_path.is_file() else "O",
        layout="centered",
    )

    # Le CSS est injecté en premier pour éviter un flash de contenu non stylé.
    load_css(CSS_PATH)
    initialise_session()

    try:
        settings = get_settings()
    except RuntimeError as error:
        logger.error("Configuration invalide : %s", error)
        st.error("Configuration de l'application incomplète. Contactez l'administrateur.")
        st.stop()

    if st.session_state.authenticated:
        render_dashboard(settings)
    else:
        render_login(settings)

    st.divider()
    st.markdown(
        f'<div class="app-footer">OncoScan AI v{APP_VERSION} | '
        "Outil d'aide à la décision médicale | by @Madiba.</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()