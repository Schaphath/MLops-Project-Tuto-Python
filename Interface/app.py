"""Interface web OncoScan (Streamlit) : connexion, analyse et historique."""

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paramètres globaux de l'application, de la sécurité et des seuils métier
# ---------------------------------------------------------------------------
APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
MAX_HISTORY_ROWS = 50

# Protection contre les tentatives de connexion répétées
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 60
MAX_TRACKED_USERNAMES = 10_000  # limite la mémoire utilisée par le suivi

# Règles sur les identifiants et mots de passe
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72  # limite technique de bcrypt
MAX_PATIENT_ID_LENGTH = 50
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,30}$")

# Libellés affichés pour le diagnostic
LABEL_MALIGNANT = "Malignant"
LABEL_BENIGN = "Benign"

# Noms des variables du modèle (même ordre que dans l'API)
FEATURES = ["perimeter", "concave_points", "texture", "smoothness", "symmetry"]

# (minimum, maximum, valeur par défaut, pas) de chaque champ de saisie.
# Les minimums sont légèrement au-dessus des bornes "strictement supérieur à" de
# l'API : saisir exactement la borne donnerait une erreur 422.
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


# ---------------------------------------------------------------------------
# Configuration (variables d'environnement)
# ---------------------------------------------------------------------------
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
    """Lit les variables d'environnement (mise en cache : elles ne changent pas
    pendant la durée de vie du conteneur). Refuse de démarrer si un secret manque.
    """
    api_key = os.getenv("API_KEY", "")
    db_password = os.getenv("DB_PASSWORD", "")
    missing = [
        name
        for name, value in (("API_KEY", api_key), ("DB_PASSWORD", db_password))
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Variable(s) d'environnement manquante(s) : {', '.join(missing)}"
        )

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


# ---------------------------------------------------------------------------
# Base de données PostgreSQL
# ---------------------------------------------------------------------------
@st.cache_resource
def get_connection_pool(settings: Settings) -> pool.ThreadedConnectionPool:
    """Crée un pool de connexions partagé (évite d'ouvrir une connexion par requête)."""
    return pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        connect_timeout=5,
    )


@contextmanager
def db_cursor(settings: Settings) -> Iterator[Any]:
    """Fournit un curseur : commit si tout va bien, rollback sinon,
    et remet toujours la connexion dans le pool.
    """
    connection_pool = get_connection_pool(settings)
    connection = connection_pool.getconn()
    try:
        with connection.cursor() as cursor:
            yield cursor
        connection.commit()
    except Exception:
        # Si la connexion est déjà coupée (ex. base redémarrée), rollback impossible
        if not connection.closed:
            connection.rollback()
        raise
    finally:
        # close=True : une connexion cassée est jetée au lieu d'être réutilisée
        connection_pool.putconn(connection, close=bool(connection.closed))


# ---------------------------------------------------------------------------
# Mots de passe
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


@st.cache_resource
def get_dummy_hash() -> str:
    """Faux hash utilisé quand l'identifiant n'existe pas : le temps de réponse est
    le même qu'avec un vrai compte (on ne peut pas deviner quels comptes existent).
    """
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


# ---------------------------------------------------------------------------
# Protection contre les tentatives de connexion répétées
# ---------------------------------------------------------------------------
@st.cache_resource
def get_login_tracker() -> dict[str, Any]:
    """Suivi des échecs de connexion, partagé par TOUTES les sessions du serveur.
    (Un compteur dans st.session_state se contourne en ouvrant un nouvel onglet.)
    """
    return {"lock": threading.Lock(), "entries": {}}


def lockout_remaining(username: str) -> int:
    """Secondes restantes avant de pouvoir réessayer (0 = pas bloqué)."""
    tracker = get_login_tracker()
    with tracker["lock"]:
        entry = tracker["entries"].get(username)
        if entry is None:
            return 0
        return max(0, math.ceil(entry["locked_until"] - time.time()))


def record_failed_login(username: str) -> None:
    """Compte un échec ; après MAX_LOGIN_ATTEMPTS échecs, bloque l'identifiant."""
    tracker = get_login_tracker()
    now = time.time()
    with tracker["lock"]:
        entries = tracker["entries"]
        entry = entries.get(
            username, {"failures": 0, "last_failure": now, "locked_until": 0.0}
        )
        # Les anciens échecs (plus vieux que la période de blocage) sont oubliés
        if now - entry["last_failure"] > LOCKOUT_SECONDS:
            entry["failures"] = 0
        entry["failures"] += 1
        entry["last_failure"] = now
        if entry["failures"] >= MAX_LOGIN_ATTEMPTS:
            entry["locked_until"] = now + LOCKOUT_SECONDS
            entry["failures"] = 0
        entries[username] = entry

        # Nettoyage des entrées périmées pour ne pas remplir la mémoire
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


# ---------------------------------------------------------------------------
# Session et comptes utilisateurs
# ---------------------------------------------------------------------------
def initialise_session() -> None:
    defaults = {
        "authenticated": False,
        "username": None,
        "user_id": None,
        "last_prediction": None,
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
                (username,),
            )
            row = cursor.fetchone()
    except (psycopg2.Error, RuntimeError):
        logger.exception("Échec de lecture de l'utilisateur")
        return False, "Le service de comptes est temporairement indisponible."

    # On vérifie toujours un hash (vrai ou factice) pour un temps de réponse identique
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


# ---------------------------------------------------------------------------
# Appel de l'API de prédiction
# ---------------------------------------------------------------------------
def request_prediction(settings: Settings, inputs: dict[str, float]) -> dict[str, Any]:
    """Envoie les mesures du patient à l'API FastAPI et renvoie le résultat."""
    response = requests.post(
        settings.api_url,
        json=inputs,
        headers={"X-API-Key": settings.api_key},
        timeout=(3, 15),  # (connexion, lecture) en secondes
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
    # Sécurité médicale : on n'affiche JAMAIS un résultat par défaut si la réponse
    # est incomplète (sinon un résultat manquant serait présenté comme "bénin")
    if not isinstance(result, dict) or result.get("prediction") not in ("M", "B"):
        raise RuntimeError("Réponse invalide du service de prédiction.")
    return result


def save_prediction(
    settings: Settings,
    user_id: int,
    patient_id: str,
    inputs: dict[str, float],
    result: dict[str, Any],
) -> None:
    """Enregistre l'analyse dans l'historique de l'utilisateur."""
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
    """Lit les dernières analyses de l'utilisateur connecté (et de lui seul)."""
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
    # En-têtes de colonnes traduits en anglais
    columns = [
        "Date & Time",
        "Diagnosis",
        "Probability (%)",
        *[feature.replace("_", " ").title() for feature in FEATURES],
    ]
    return pd.DataFrame(rows, columns=columns)

# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------
def render_prediction(result: dict[str, Any]) -> None:
    """Affiche le résultat de la prédiction."""
    prediction = result["prediction"]  # déjà validé dans request_prediction
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
            key="login_password",
        )
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
    """Écran de connexion (avec création de compte si autorisée)."""
    _, content_column, _ = st.columns([1, 2.2, 1])
    with content_column:
        st.markdown(
            '<div class="auth-brand"><span class="brand-mark">O</span><span>OncoScan AI</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="auth-heading">'
            '<p class="auth-kicker">ESPACE PROFESSIONNEL</p>'
            '<h2 class="auth-title">Bienvenue dans votre espace</h2>'
            '<p class="auth-subtitle">Analysez et retrouvez les résultats en un seul endroit.</p>'
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
    """Affiche le formulaire de saisie des mesures et lance la prédiction."""
    st.caption("Entrez les informations cliniques et obtenez une prédiction.")

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
            "Lancer l'analyse", type="primary", use_container_width=True
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
    """Récupère et affiche les prédictions enregistrées pour l'utilisateur."""

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
    # La base stocke une probabilité entre 0 et 1 : on la convertit en pourcentage
    history["Probability (%)"] = pd.to_numeric(history["Probability (%)"], errors="coerce") * 100

    # Réduction visuelle du bloc Filter en le plaçant dans une colonne étroite
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
    # Les dates sont stockées en UTC : on les affiche à l'heure locale
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
                    ("font-size", "0.72rem"),  # En-têtes plus petits
                    ("border-bottom", "1px solid #ead6dc"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#fffafb"),
                    ("color", "#30242a"),
                    ("font-size", "0.75rem"),  # Contenu plus petit
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
    """Affiche la navigation latérale et les deux onglets principaux."""
    with st.sidebar:
        st.markdown(
            '<div class="app-brand"><span class="app-brand-mark">O</span><span>OncoScan</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Session : {st.session_state.username}")
        st.divider()
        if st.button("Se déconnecter", use_container_width=True):
            st.session_state.clear()  # efface toute la session (y compris le dernier résultat)
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


# ---------------------------------------------------------------------------
# Style de l'interface (CSS)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ---------------------------------------------------------------------------
   1. IMPORTATION DES POLICES & VARIABLES GLOBALES
   --------------------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
    --rose-50: #fffafb;   /* Fond principal */
    --rose-100: #f8eef1;  /* En-têtes, sidebar */
    --rose-200: #ead6dc;  /* Bordures */
    --rose-400: #c98596;  
    --rose-600: #9e5367;  /* Boutons & accent */
    --ink: #30242a;       /* Texte foncé */
    --muted: #76656b;     /* Texte secondaire */
    --white: #ffffff;     /* Blanc */
}

/* ---------------------------------------------------------------------------
   2. STYLES GÉNÉRAUX & CONTENEUR ÉLARGI
   --------------------------------------------------------------------------- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {
    background: var(--rose-50) !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Extension maximale de la zone principale pour étirer le tableau au maximum */
[data-testid="stMainBlockContainer"] {
    max-width: 1200px !important;
    padding: 2.5rem 1.5rem 1.5rem !important;
}

h1, h2, h3, h4 {
    color: var(--ink) !important;
    font-family: 'Manrope', sans-serif !important;
}

p, label, [data-testid="stMarkdownContainer"] {
    color: var(--ink) !important;
}

[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}

/* ---------------------------------------------------------------------------
   3. BRANDING & TITRES
   --------------------------------------------------------------------------- */
.auth-brand {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    color: var(--ink);
    font: 800 2.2rem 'Manrope', sans-serif;
    margin: 0 0 2rem;
}

.brand-mark {
    display: grid;
    place-items: center;
    width: 4rem;
    height: 4rem;
    border-radius: 16px;
    color: var(--white);
    background: var(--rose-600);
    font-size: 2rem;
    box-shadow: 0 8px 18px rgba(158,83,103,.16);
}

.app-brand {
    display: flex;
    align-items: center;
    gap: .7rem;
    color: var(--ink);
    font: 800 1.35rem 'Manrope', sans-serif;
    margin: .35rem 0 1.2rem;
}

.app-brand-mark {
    display: grid;
    place-items: center;
    width: 2.6rem;
    height: 2.6rem;
    border-radius: 11px;
    color: var(--white);
    background: var(--rose-600);
    font-size: 1.4rem;
    box-shadow: 0 5px 12px rgba(158,83,103,.14);
}

.auth-heading, .dashboard-heading {
    text-align: center;
    margin: 0 0 1.4rem;
}

.dashboard-title, .auth-title {
    font-size: clamp(1.5rem, 2.5vw, 2.1rem) !important;
    line-height: 1.15;
    margin: 0 0 .5rem;
}

.auth-kicker {
    color: var(--rose-600) !important;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .14em;
    margin: 0 0 .75rem;
}

.auth-subtitle {
    color: var(--muted) !important;
    font-size: .96rem;
    line-height: 1.6;
    max-width: 42rem;
    margin: 0 auto 1.5rem;
}

/* ---------------------------------------------------------------------------
   4. FORMULAIRES ET TABS
   --------------------------------------------------------------------------- */
[data-testid="stForm"], [data-testid="stTabs"] {
    width: 100% !important;
    max-width: 560px !important;
    margin: 0 auto 1rem !important;
}

[data-testid="stTabs"] [role="tablist"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: .5rem !important;
    border-bottom: 2px solid var(--rose-100) !important;
    padding-bottom: 2px !important;
}

[data-testid="stTabs"] button[role="tab"] {
    display: inline-flex !important;
    justify-content: center !important;
    align-items: center !important;
    flex: 0 1 220px !important;
    min-width: 160px !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
    font-size: .92rem !important;
    padding: .7rem 1rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
    margin-bottom: -2px !important;
    white-space: nowrap !important;
}

[data-testid="stTabs"] button[role="tab"]:hover {
    color: var(--rose-600) !important;
    background: rgba(248, 238, 241, 0.55) !important;
    border-radius: 6px 6px 0 0 !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--rose-600) !important;
    border-bottom: 2px solid var(--rose-600) !important;
    background: transparent !important;
}

[data-testid="stTabs"] [role="tabpanel"] {
    padding-top: 1rem;
}

/* ---------------------------------------------------------------------------
   5. FONDS BLANCS ET SUPPRESSION DU BLOC "PRESS ENTER TO SUBMIT FORM"
   --------------------------------------------------------------------------- */
/* Masquer l'indicateur de soumission et les instructions en incrustation */
[data-testid="InputInstructions"],
div[data-testid="stTextInput"] small,
div[data-baseweb="input"] + div {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    color: var(--ink) !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
}

/* Annulation des fonds gris et préservation du fond blanc */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input:-webkit-autofill,
[data-testid="stTextInput"] input:-webkit-autofill:hover, 
[data-testid="stTextInput"] input:-webkit-autofill:focus,
[data-testid="stNumberInput"] input:-webkit-autofill,
[data-testid="stNumberInput"] input:-webkit-autofill:hover,
[data-testid="stNumberInput"] input:-webkit-autofill:focus {
    background-color: var(--white) !important;
    -webkit-text-fill-color: var(--ink) !important;
    -webkit-box-shadow: 0 0 0px 1000px var(--white) inset !important;
    color: var(--ink) !important;
    font-size: .88rem !important;
    border: none !important;
}

/* Conteneur du champ de saisie */
[data-testid="stTextInput"] > div,
[data-testid="stNumberInput"] > div,
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"],
[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
    background-color: var(--white) !important;
    border: 1px solid var(--rose-200) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Icône de l'œil */
[data-testid="stTextInput"] div[data-baseweb="input"] button,
[data-testid="stTextInput"] button[aria-label="Show password"],
[data-testid="stTextInput"] button[aria-label="Hide password"] {
    background-color: var(--white) !important;
    border: none !important;
    padding-right: 10px !important;
}

[data-testid="stTextInput"] svg {
    fill: var(--ink) !important;
    color: var(--ink) !important;
}

/* Focus */
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    border-color: var(--rose-600) !important;
    box-shadow: 0 0 0 3px rgba(158,83,103,.12) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    font-size: 13px !important;
    min-height: 32px !important;
    background-color: var(--white) !important;
    border: 1px solid var(--rose-200) !important;
    border-radius: 10px !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: var(--ink) !important;
    background-color: transparent !important;
}

/* Placeholder */
[data-testid="stTextInput"] input::placeholder {
    color: #aa999f !important;
    opacity: 1 !important;
}

/* ---------------------------------------------------------------------------
   6. BULLES D'AIDE ET HELPER TEXTS À FOND BLANC
   --------------------------------------------------------------------------- */
/* Bulles d'aide (Help / Tooltips Streamlit) */
[data-testid="stTooltipIcon"] > div,
[role="tooltip"],
div[data-baseweb="popover"],
div[data-baseweb="tooltip"] {
    background-color: var(--white) !important;
    color: var(--ink) !important;
    border: 1px solid var(--rose-200) !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
}

div[data-baseweb="tooltip"] * {
    color: var(--ink) !important;
    background-color: var(--white) !important;
}

/* Texte d'explication d'aide en dessous des champs */
[data-testid="stInputHelp"], .stMarkdown small {
    color: var(--muted) !important;
    font-size: .78rem !important;
}

/* ---------------------------------------------------------------------------
   7. BOUTONS D'ACTION
   --------------------------------------------------------------------------- */
.stButton > button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {
    width: min(100%, 220px) !important;
    min-height: 2.7rem !important;
    padding: .65rem 1.2rem !important;
    border-radius: 9px !important;
    border: 1px solid var(--rose-600) !important;
    background-color: var(--rose-600) !important;
    color: var(--white) !important;
    font-weight: 700 !important;
    box-shadow: 0 5px 12px rgba(158,83,103,.12) !important;
    transition: background .18s ease, border-color .18s ease;
}

.stButton > button *,
[data-testid="stFormSubmitButton"] button *,
[data-testid="stDownloadButton"] button * {
    color: var(--white) !important;
}

[data-testid="stFormSubmitButton"] {
    display: flex;
    justify-content: center;
    width: 100%;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
    background-color: #874459 !important;
    border-color: #874459 !important;
}

/* ---------------------------------------------------------------------------
   8. BARRE LATÉRALE (SIDEBAR)
   --------------------------------------------------------------------------- */
[data-testid="stSidebar"] {
    background: var(--rose-100) !important;
    border-right: 1px solid var(--rose-200);
}

/* ---------------------------------------------------------------------------
   9. TABLEAU OPTIMISÉ (ÉLARGISSEMENT & POLICES COMPACTES SUR UNE LIGNE)
   --------------------------------------------------------------------------- */
[data-testid="stTable"] {
    width: 100% !important;
    overflow-x: auto !important;
}

[data-testid="stTable"] table {
    width: 100% !important;
    background: var(--white) !important;
    border: 1px solid var(--rose-200);
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 8px;
    overflow: hidden;
    table-layout: auto !important;
}

/* En-tête du tableau : police fine, compacte et centrée sans coupure */
[data-testid="stTable"] th {
    background: var(--rose-100) !important;
    color: var(--rose-600) !important;
    font-size: .65rem !important;
    font-weight: 700 !important;
    padding: 6px 6px !important;
    border-bottom: 1px solid var(--rose-200) !important;
    white-space: nowrap !important;
    text-align: center !important;
}

/* Cellules du tableau : police très réduite, alignement propre sans retour à la ligne */
[data-testid="stTable"] td {
    background: var(--white) !important;
    color: var(--ink) !important;
    font-size: .68rem !important;
    padding: 5px 6px !important;
    border-bottom: 1px solid #f1e3e7 !important;
    white-space: nowrap !important;
    text-align: center !important;
}

/* ---------------------------------------------------------------------------
   10. RESPONSIVE DESIGN
   --------------------------------------------------------------------------- */
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1.5rem 1rem !important;
    }
}
</style>
"""


# ---------------------------------------------------------------------------
# Point d'entrée de l'application
# ---------------------------------------------------------------------------
def main() -> None:
    """Configure la page, vérifie la configuration, puis affiche soit le
    tableau de bord, soit l'écran de connexion selon l'état de la session.
    """
    # Doit être le tout premier appel Streamlit
    logo_path = Path(__file__).resolve().parent / "favicon.svg"
    st.set_page_config(
        page_title="OncoScan",
        page_icon=str(logo_path) if logo_path.is_file() else "O",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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