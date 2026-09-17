from __future__ import annotations

#---------------------------#
# Imports et configuration  #
#---------------------------#
import logging
import os
import re
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

#---------------------------#
# Variables d'environnement #
#---------------------------#
APP_VERSION = "2.0.0"
MAX_HISTORY_ROWS = 50
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 60
MIN_PASSWORD_LENGTH = 8
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,30}$")

#-----------------#
# Nom de features #
#-----------------#
FEATURES = [
    "perimeter",
    "concave_points",
    "texture",
    "smoothness",
    "symmetry"
]

#----------------------#
# Bornes des variables #
#----------------------#
FEATURE_BOUNDS = {
    "perimeter": (40.0, 250.0, 90.0, 1.0),
    "concave_points": (0.0, 0.30, 0.05, 0.001),
    "texture": (5.0, 50.0, 17.0, 0.1),
    "smoothness": (0.01, 0.30, 0.09, 0.001),
    "symmetry": (0.05, 0.70, 0.18, 0.001),
}

FEATURE_HELP = {
    "perimeter": "Périmètre total de la tumeur observé",
    "concave_points": "Nombre de points concaves présents sur le contour tumoral.",
    "texture": "Variation du niveau de gris dans la zone tumorale.",
    "smoothness": "Variation locale de la longueur du contour tumoral.",
    "symmetry": "Degré d'asymétrie de la forme tumorale."
}


#---------------------------#
#  Paramètres de connexion  #
#---------------------------#
@dataclass(frozen=True)
class Settings:
    api_url: str
    api_key: str | None
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str


@st.cache_resource
def get_settings() -> Settings:
    """Lit les variables d'environnement de l'application.
    Cette fonction est mise en cache car les paramètres ne changent pas
    pendant la durée d'un lancement de l'interface.
    """
    return Settings(
        api_url=os.getenv("API_URL", "http://api:8000/predict"),
        api_key=os.getenv("API_KEY") or None,
        db_host=os.getenv("DB_HOST", "postgres"),
        db_port=int(os.getenv("DB_PORT", "5432")),
        db_name=os.getenv("DB_NAME", "oncoscan"),
        db_user=os.getenv("DB_USER", "oncoscan"),
        db_password=os.getenv("DB_PASSWORD", ""),
    )


#---------------------------------#
#  Pool de connexions PostgreSQL  #
#---------------------------------#
@st.cache_resource
def get_connection_pool(settings: Settings) -> pool.ThreadedConnectionPool:
    if not settings.db_password:
        raise RuntimeError("DB_PASSWORD n'est pas configuree.")
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
    """Garantit commit, rollback et retour de la connexion au pool."""
    connection = get_connection_pool(settings).getconn()
    try:
        with connection.cursor() as cursor:
            yield cursor
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        get_connection_pool(settings).putconn(connection)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def valid_username(username: str) -> bool:
    return bool(USERNAME_PATTERN.fullmatch(username))


def valid_password(password: str) -> bool:
    return (
        len(password) >= MIN_PASSWORD_LENGTH
        and any(character.isalpha() for character in password)
        and any(character.isdigit() for character in password)
    )


def initialise_session() -> None:
    defaults = {
        "authenticated": False,
        "username": None,
        "user_id": None,
        "failed_attempts": 0,
        "locked_until": 0.0,
        "last_prediction": None,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def login(settings: Settings, username: str, password: str) -> tuple[bool, str]:
    if time.time() < st.session_state.locked_until:
        remaining = max(1, int(st.session_state.locked_until - time.time()))

        return False, f"Trop de tentatives. Reessayez dans {remaining} seconde(s)."
    try:
        with db_cursor(settings) as cursor:
            cursor.execute(
                "SELECT id, password_hash FROM users WHERE username = %s",
                (username,),
            )
            row = cursor.fetchone()

    except (psycopg2.Error, RuntimeError):
        logging.getLogger(__name__).exception("Echec de lecture de l'utilisateur")
        return False, "Le service de comptes est temporairement indisponible."

    if row and verify_password(password, row[1]):
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.user_id = row[0]
        st.session_state.failed_attempts = 0
        st.session_state.locked_until = 0.0
        return True, "Connexion reussie."

    st.session_state.failed_attempts += 1
    if st.session_state.failed_attempts >= MAX_LOGIN_ATTEMPTS:
        st.session_state.locked_until = time.time() + LOCKOUT_SECONDS
        st.session_state.failed_attempts = 0
    return False, "Identifiant ou mot de passe incorrect."


def register_user(settings: Settings, username: str, password: str) -> tuple[bool, str]:
    try:
        with db_cursor(settings) as cursor:
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, hash_password(password)),
            )
        return True, "Compte cree. Vous pouvez maintenant vous connecter."
    except psycopg2.errors.UniqueViolation:
        return False, "Cet identifiant existe deja."
    except (psycopg2.Error, RuntimeError):
        logging.getLogger(__name__).exception("Echec de creation du compte")
        return False, "Le service de comptes est temporairement indisponible."


#-------------------------------#
#  Appel à l'API de prédiction  #
#-------------------------------#
def request_prediction(settings: Settings, inputs: dict[str, float]) -> dict[str, Any]:
    """Envoie les mesures du patient à l'API FastAPI et renvoie le résultat."""
    # Besoin d'une clé API en prod avant d'éffectuer une prédiciton 
    headers = {"X-API-Key": settings.api_key} if settings.api_key else {}
    response = requests.post(
        settings.api_url,
        json=inputs,
        headers=headers,
        timeout=(3, 15),
    )
    if response.status_code != 200:
        if response.status_code in (401, 403):
            raise RuntimeError("Le service de prediction a refuse la requete.")
        if response.status_code == 422:
            raise ValueError("Les valeurs saisies sont hors des limites acceptees.")
        if response.status_code == 503:
            raise RuntimeError("Le modele de prediction est indisponible.")
        raise RuntimeError(
            f"Le service de prediction a repondu {response.status_code}."
        )
    return response.json()


def save_prediction(
    settings: Settings,
    patient_id: str,
    inputs: dict[str, float],
    result: dict[str, Any],
) -> None:
    probability = result.get("probability_malignant", result.get("probability"))
    probability_value = (
        round(float(probability), 4) if probability is not None else None
    )
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
                st.session_state.user_id,
                patient_id,
                *values,
                result.get("prediction", "B"),
                probability_value,
            ),
        )


def get_history(settings: Settings) -> pd.DataFrame:
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
            (st.session_state.user_id, MAX_HISTORY_ROWS),
        )
        rows = cursor.fetchall()
    columns = [
        "Date et heure",
        "Diagnostic",
        "Probabilite (%)",
        *[feature.replace("_", " ").title() for feature in FEATURES],
    ]
    return pd.DataFrame(rows, columns=columns)


def render_prediction(result: dict[str, Any]) -> None:
    prediction = result.get("prédiction")
    if prediction is None:
        prediction = "M" if result.get("label") == "Maligne" else "B"

    probability = result.get("probability_malignant", result.get("probability"))
    probability_text = (
        f"{float(probability) * 100:.2f} %" if probability is not None else "non disponible"
    )

    if prediction == "M":
        st.error("Résultat du modèle : classe maligne")
        st.warning(
            f"Probabilité estimée : {probability_text}. Ce resultat doit être confirmé par un professionnel de santé."
        )
    else:
        st.success("Résultat du modele : classe benigne")
        st.info(f"Probabilité estimée : {probability_text}.")
    st.caption(f"Version du modèle : {result.get('model_version', 'inconnue')}")


def render_login(settings: Settings) -> None:
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
            '<p class="auth-subtitle">Analysez et rétrouvez les résultats en un seul endroit.</p>'
            "</div>",
            unsafe_allow_html=True,
        )
        login_tab, register_tab = st.tabs(["Connexion", "Créer un compte"])
        with login_tab:
            with st.form("login_form"):
                username = st.text_input("Identifiant", placeholder="Votre identifiant")
                password = st.text_input(
                    "Mot de passe", type="password", placeholder="Votre mot de passe"
                )
                submitted = st.form_submit_button(
                    "Se connecter", type="primary", use_container_width=False
                )
            if submitted:
                if not username.strip() or not password:
                    st.warning("Saisissez votre identifiant et votre mot de passe.")
                else:
                    success, message = login(
                        settings, username.strip().lower(), password
                    )
                    (st.success if success else st.error)(message)
                    if success:
                        st.rerun()

        with register_tab:
            with st.form("register_form"):
                username = st.text_input(
                    "Nouvel identifiant", placeholder="ex. Dr martin"
                )
                password = st.text_input(
                    "Mot de passe", type="password", placeholder="8 caractères minimum"
                )
                confirmation = st.text_input(
                    "Confirmez le mot de passe", type="password"
                )
                st.caption(
                    "3 à 30 caractères. Le mot de passe doit contenir une lettre et un chiffre."
                )
                submitted = st.form_submit_button(
                    "Créer le compte", use_container_width=False
                )
            if submitted:
                username = username.strip().lower()
                if not valid_username(username):
                    st.warning("Identifiant ou mot de passe invalide.")
                elif password != confirmation:
                    st.warning("Les mots de passe ne correspondent pas.")
                elif not valid_password(password):
                    st.warning(
                        "Le mot de passe doit contenir une lettre, un chiffre et au moins 8 caracteres."
                    )
                else:
                    success, message = register_user(settings, username, password)
                    (st.success if success else st.error)(message)


#------------------------#
#  Formulaire d'analyse  #
#------------------------#
def render_analysis(settings: Settings) -> None:
    """Affiche le formulaire de saisie des mesures et lance la prédiction."""

    st.subheader("Nouvelle analyse")
    st.caption("Entrez les informations cliniques et obtenez une prédiction.")

    with st.form("prediction_form"):
        patient_id = st.text_input(
            "Identifiant patient",
            placeholder="ex. PAT-001",
            help="Identifiant patient requis pour l'historique.",
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
                        save_prediction(settings, patient_id.strip(), inputs, result)
                    except (psycopg2.Error, RuntimeError):
                        logging.getLogger(__name__).exception(
                            "Echec de sauvegarde"
                        )
                        st.warning(
                            "Analyse terminée, mais l'historique n'a pas pu être sauvegarde."
                        )
                except requests.exceptions.Timeout:
                    st.error("Le service de prédiction met trop de temps a répondre.")
                except requests.exceptions.RequestException:
                    logging.getLogger(__name__).exception(
                        "Echec de communication avec l'API"
                    )
                    st.error("Impossible de contacter le service API.")
                except (RuntimeError, ValueError) as error:
                    st.error(str(error))
    if st.session_state.last_prediction:
        render_prediction(st.session_state.last_prediction)


#---------------------------#
#  Historique des analyses  #
#---------------------------#
def render_history(settings: Settings) -> None:
    """Récupère et affiche les prédictions enregistrées pour l'utilisateur."""
    st.subheader("Historique")
    try:
        history = get_history(settings)
    except (psycopg2.Error, RuntimeError):
        logging.getLogger(__name__).exception("Echec de lecture de l'historique")
        st.error("Impossible de charger l'historique pour le moment.")
        return
    if history.empty:
        st.info("Aucune analyse enregistrée pour le moment.")
        return
    history["Diagnostic"] = history["Diagnostic"].map({"M": "Maligne", "B": "Benigne"})
    choice = st.selectbox(
        "Filtrer", ["Toutes", "Maligne", "Benigne"], key="history_filter"
        )
    
    if choice != "Toutes":
        history = history[history["Diagnostic"] == choice]
    show_details = st.checkbox("Affichez les mesures cliniques", key="history_details")
    visible_columns = ["Date et heure", "Diagnostic", "Probabilite (%)"]

    if show_details:
        visible_columns += [
            column for column in history.columns if column not in visible_columns
        ]

    visible_history = history[visible_columns].copy()
    visible_history["Date et heure"] = pd.to_datetime(
        visible_history["Date et heure"], utc=True
        ).dt.strftime("%d/%m/%Y %H:%M")

    visible_history["Probabilite (%)"] = visible_history["Probabilite (%)"].map(
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
                    ("border-bottom", "1px solid #ead6dc"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#fffafb"),
                    ("color", "#30242a"),
                    ("border-bottom", "1px solid #f1e3e7"),
                ],
            },
        ]
    ).set_properties(
        **{"text-align": "left", "padding": "0.55rem", "font-size": "0.82rem"}
    )
    st.table(table)
    st.download_button(
        "Telechargez l'historique (CSV)",
        data=visible_history.to_csv(index=False).encode("utf-8-sig"),
        file_name="oncoscan_historique.csv",
        mime="text/csv",
        use_container_width=False,
    )


#---------------------------#
# Tableau de bord principal #
#---------------------------#
def render_dashboard(settings: Settings) -> None:
    """Affiche la navigation latérale et les deux onglets principaux.

    Les onglets "Nouvelle analyse" et "Historique" occupent la majeure partie
    de l'espace de travail pour optimiser la lecture et la saisie.
    """
    with st.sidebar:
        st.markdown(
            '<div class="app-brand"><span class="app-brand-mark">O</span><span>OncoScan</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Session : {st.session_state.username}")
        st.divider()
        if st.button("Se déconnecter", use_container_width=True):
            for key in ("authenticated", "username", "user_id", "last_prediction"):
                st.session_state.pop(key, None)
            st.rerun()
        st.caption("Les résultats sont indicatifs et ne remplacent pas un avis médical.")

    st.markdown(
        '<div class="dashboard-heading">'
        '<p class="auth-kicker">ESPACE PROFESSIONNEL</p>'
        '<h2 class="dashboard-title">Votre espace clinique</h2>'
        "</div>",
        unsafe_allow_html=True,
    )

    analysis_tab, history_tab = st.tabs(["Nouvelle analyse", "Historique"])
    with analysis_tab:
        render_analysis(settings)
    with history_tab:
        render_history(settings)


#---------------------------------#
# Point d'entrée de l'application #
#---------------------------------#
def main() -> None:
    """Configure la page Streamlit, initialise les sessions, puis affiche
    soit le tableau de bord, soit l'écran de connexion selon l'état.
    """
    logo_path = Path(__file__).resolve().parent / "favicon.svg"
    st.set_page_config(
        page_title="OncoScan",
        page_icon=str(logo_path) if logo_path.is_file() else "O",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

        /* --- VARIABLES GLOBALES --- */
        :root {
            --rose-50: #fffafb;
            --rose-100: #f8eef1;
            --rose-200: #ead6dc;
            --rose-400: #c98596;
            --rose-600: #9e5367;
            --ink: #30242a;
            --muted: #76656b;
            --white: #ffffff;
        }

        /* --- TYPOGRAPHIE & APPARENCE GÉNÉRALE --- */
        html, body, [class*="css"], .stApp,
        [data-testid="stAppViewContainer"], [data-testid="stMain"],
        [data-testid="stHeader"] {
            color: var(--ink) !important;
            background: var(--rose-50) !important;
            font-family: 'DM Sans', sans-serif !important;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1120px !important;
            padding: 3.5rem 2rem 2rem !important;
        }

        h1, h2, h3, h4 { color: var(--ink) !important; font-family: 'Manrope', sans-serif !important; }
        p, label, [data-testid="stMarkdownContainer"] { color: var(--ink) !important; }
        [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

        /* --- BRANDING & HEADERS --- */
        .auth-brand { display: flex; align-items: center; justify-content: center; gap: 1rem; color: var(--ink); font: 800 2.5rem 'Manrope', sans-serif; margin: 0 0 2.5rem; }
        .brand-mark { display: grid; place-items: center; width: 4.3rem; height: 4.3rem; border-radius: 16px; color: var(--white); background: var(--rose-600); font-size: 2.35rem; box-shadow: 0 8px 18px rgba(158,83,103,.16); }
        .app-brand { display: flex; align-items: center; gap: .7rem; color: var(--ink); font: 800 1.45rem 'Manrope', sans-serif; margin: .35rem 0 1.4rem; }
        .app-brand-mark { display: grid; place-items: center; width: 2.65rem; height: 2.65rem; border-radius: 11px; color: var(--white); background: var(--rose-600); font-size: 1.45rem; box-shadow: 0 5px 12px rgba(158,83,103,.14); }
        
        .auth-heading { text-align: center; }
        .dashboard-heading { text-align: center; margin: 0 0 2rem; }
        .dashboard-title { font-size: clamp(1.7rem, 3vw, 2.45rem) !important; line-height: 1.15; margin: 0 0 .65rem; }
        .dashboard-heading .auth-subtitle { display: block; width: 100%; text-align: center; }
        .auth-kicker { color: var(--rose-600) !important; font-size: .72rem; font-weight: 700; letter-spacing: .14em; margin: 0 0 .8rem; }
        .auth-title { font-size: clamp(1.65rem, 3vw, 2.35rem) !important; line-height: 1.15; margin: 0 0 .7rem; }
        .auth-subtitle { color: var(--muted) !important; font-size: .98rem; line-height: 1.6; max-width: 44rem; margin: 0 auto 2rem; }
        .app-footer { width: 100%; padding: 1rem 0 0; text-align: center; color: var(--muted); font-size: .78rem; }

        /* --- CADRAGE DES FORMULAIRES ET DES ONGLETS --- */
        [data-testid="stForm"]"aza {
            width: 100% !important;
            max-width: 440px !important; /* Largeur compacte optimale pour l'UX d'authentification */
            margin: 0 auto !important;
        }

        [data-testid="stTabs"] {
            width: 100% !important;
            max-width: none !important;
            margin: 0 auto !important;
        }

        [data-testid="stForm"] { padding-top: 1.25rem; }

        /* --- UX OPTIMISÉE POUR LES ONGLETS (stTabs) --- */
        [data-testid="stTabs"] [role="tablist"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: 100% !important;
            gap: 0.5rem !important;
            border-bottom: 2px solid var(--rose-100) !important;
            padding-bottom: 2px !important;
        }

        [data-testid="stTabs"] button[role="tab"] {
            display: inline-flex !important;
            justify-content: center !important;
            align-items: center !important;
            flex: 1 !important;
            color: var(--muted) !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 0.75rem 1rem !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            background: transparent !important;
            transition: all 0.2s ease !important;
            margin-bottom: -2px !important;
            white-space: nowrap !important;
        }

        [data-testid="stTabs"] button[role="tab"]:hover {
            color: var(--rose-600) !important;
            background: rgba(248, 238, 241, 0.5) !important;
            border-radius: 6px 6px 0 0 !important;
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--rose-600) !important;
            border-bottom: 2px solid var(--rose-600) !important;
            background: transparent !important;
        }

        [data-testid="stTabs"] [role="tabpanel"] { padding-top: 1.25rem; }

        /* --- CHAMPS DE SAISIE (INPUTS) --- */
        [data-testid="stTextInput"] label, [data-testid="stNumberInput"] label { color: var(--ink) !important; font-size: .82rem; font-weight: 600; }
        
        [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            background: var(--white) !important;
            border: 1px solid var(--rose-200) !important;
            border-radius: 8px !important;
            min-height: 2.7rem;
        }

        [data-testid="stTextInput"] [data-baseweb="input"], [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stTextInput"] [data-baseweb="base-input"], [data-testid="stNumberInput"] [data-baseweb="base-input"] {
            background: var(--white) !important;
            border: 2px solid var(--rose-400) !important;
            border-radius: 9px !important;
            min-height: 2.7rem;
            overflow: hidden;
        }

        [data-testid="stTextInput"] [data-baseweb="input"] input, [data-testid="stNumberInput"] [data-baseweb="input"] input {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            background: transparent !important;
            border: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
        }

        [data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
        [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
            border-color: var(--rose-600) !important;
            box-shadow: 0 0 0 3px rgba(158,83,103,.16) !important;
        }

        /* MOT DE PASSE ET BOUTON AFFICHER/MASQUER */
        [data-testid="stTextInput"]:has(input[type="password"]) [data-baseweb="input"] {
            min-height: 2.7rem;
            border: 2px solid var(--rose-400) !important;
            border-radius: 9px !important;
            overflow: hidden;
            display: flex !important;
            align-items: stretch !important;
            background: var(--white) !important;
        }

        [data-testid="stTextInput"]:has(input[type="password"]) [data-baseweb="input"] input {
            height: 2.55rem !important;
            min-height: 2.55rem !important;
            flex: 1 1 auto !important;
            border: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
        }

        [data-testid="stTextInput"]:has(input[type="password"]) [data-baseweb="input"] button {
            width: 2.8rem !important;
            min-width: 2.8rem !important;
            height: 2.55rem !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
            border-left: 1px solid var(--rose-200) !important;
            border-radius: 0 !important;
            background: var(--rose-100) !important;
            color: var(--rose-600) !important;
            display: grid !important;
            place-items: center !important;
        }

        [data-testid="stTextInput"]:has(input[type="password"]) [data-baseweb="input"] button:hover {
            background: var(--rose-200) !important;
            color: var(--rose-600) !important;
        }

        [data-testid="stTextInput"]:has(input[type="password"]) [data-baseweb="input"] button svg {
            width: 1.15rem !important;
            height: 1.15rem !important;
            fill: currentColor !important;
            color: currentColor !important;
        }

        [data-testid="stTextInput"] div, [data-testid="stNumberInput"] div { background-color: var(--white) !important; }
        
        [data-testid="stTextInput"]:has(input[type="password"]) [data-baseweb="input"] button,
        [data-testid="stTextInput"]:has(input[type="password"]) [data-baseweb="input"] button > * {
            background: var(--rose-100) !important;
        }

        /* AUTOFILL SAFARI / CHROME */
        [data-testid="stTextInput"] input:-webkit-autofill,
        [data-testid="stTextInput"] input:-webkit-autofill:hover,
        [data-testid="stTextInput"] input:-webkit-autofill:focus,
        [data-testid="stTextInput"] input:-webkit-autofill:active {
            -webkit-text-fill-color: var(--ink) !important;
            -webkit-box-shadow: 0 0 0 1000px var(--white) inset !important;
            box-shadow: 0 0 0 1000px var(--white) inset !important;
            background-color: var(--white) !important;
            transition: background-color 9999s ease-in-out 0s;
        }

        [data-testid="stTextInput"] input::placeholder { color: #aa999f !important; opacity: 1 !important; }
        [data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus { border-color: var(--rose-600) !important; box-shadow: 0 0 0 3px rgba(158,83,103,.16) !important; }
        [data-testid="stTextInput"]:has(input[type="password"]) [data-baseweb="input"]:focus-within { border-color: var(--rose-600) !important; box-shadow: 0 0 0 3px rgba(158,83,103,.16) !important; }

        /* --- BOUTONS & SOUMISSION --- */
        .stButton > button, [data-testid="stFormSubmitButton"] button {
            width: 100% !important; /* Le bouton prend toute la largeur du bloc 440px pour l'auth */
            min-height: 2.65rem;
            padding: .55rem 1.25rem !important;
            border-radius: 8px !important;
            border: 1px solid var(--rose-600) !important;
            background: var(--rose-600) !important;
            color: var(--white) !important;
            font-weight: 700 !important;
            box-shadow: 0 5px 12px rgba(158,83,103,.14) !important;
            transition: background .18s ease, border-color .18s ease;
        }

        [data-testid="stFormSubmitButton"] { display: flex; justify-content: center; width: 100%; }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { background: #874459 !important; border-color: #874459 !important; }

        /* --- SIDEBAR --- */
        [data-testid="stSidebar"] { background: var(--rose-100) !important; border-right: 1px solid var(--rose-200); }
        [data-testid="stSidebar"] .stButton > button { 
            width: 100% !important; 
            background: var(--white) !important; 
            color: var(--rose-600) !important; 
            border-color: var(--rose-400) !important; 
            box-shadow: none !important; 
            transition: background .18s ease, color .18s ease, border-color .18s ease, transform .18s ease !important; 
        }
        [data-testid="stSidebar"] .stButton > button:hover { 
            background: var(--rose-600) !important; 
            color: var(--white) !important; 
            border-color: var(--rose-600) !important; 
            transform: translateY(-1px); 
            box-shadow: 0 5px 12px rgba(158,83,103,.16) !important; 
        }
        [data-testid="stSidebar"] .stButton > button:active { transform: translateY(0); }

        /* --- SELECTBOX, CHECKBOX & COMPOSANTS DIVERS --- */
        [data-testid="stSelectbox"] label { color: var(--ink) !important; font-size: .82rem; font-weight: 600; }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div { min-height: 2.45rem; background: var(--white) !important; color: var(--ink) !important; border: 1px solid var(--rose-200) !important; border-radius: 8px !important; box-shadow: none !important; }
        [data-testid="stSelectbox"] [data-baseweb="select"] span { color: var(--ink) !important; }
        [data-testid="stSelectbox"] [data-baseweb="select"] svg { fill: var(--rose-600) !important; }
        
        [data-baseweb="popover"], [data-baseweb="menu"] { background: var(--white) !important; border: 1px solid var(--rose-200) !important; border-radius: 8px !important; box-shadow: 0 8px 24px rgba(158,83,103,.14) !important; }
        [data-baseweb="menu"] [role="option"] { background: var(--white) !important; color: var(--ink) !important; }
        [data-baseweb="menu"] [role="option"]:hover, [data-baseweb="menu"] [aria-selected="true"] { background: var(--rose-100) !important; color: var(--rose-600) !important; }

        [data-testid="stCheckbox"] label { color: var(--ink) !important; }
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] { background: var(--white) !important; border: 1px solid var(--rose-400) !important; border-radius: 5px !important; }
        [data-testid="stCheckbox"] [data-baseweb="checkbox"][aria-checked="true"] { background: var(--rose-600) !important; border-color: var(--rose-600) !important; }
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] svg { fill: var(--white) !important; color: var(--white) !important; }

        [data-testid="stDownloadButton"] { display: flex !important; justify-content: center !important; }
        [data-testid="stDownloadButton"] button { width: auto !important; min-width: 0 !important; background: var(--white) !important; color: var(--rose-600) !important; border: 1px solid var(--rose-400) !important; box-shadow: none !important; transition: background .18s ease, color .18s ease, border-color .18s ease, transform .18s ease !important; }
        [data-testid="stDownloadButton"] button:hover { background: var(--rose-600) !important; color: var(--white) !important; border-color: var(--rose-600) !important; transform: translateY(-1px); box-shadow: 0 5px 12px rgba(158,83,103,.16) !important; }

        /* --- TABLEAUX & DATAFRAMES --- */
        [data-testid="stTable"] { width: 100%; overflow-x: auto; }
        [data-testid="stTable"] table { width: 100%; min-width: 640px; background: var(--white) !important; border: 1px solid var(--rose-200); border-collapse: separate; border-spacing: 0; border-radius: 8px; overflow: hidden; }
        [data-testid="stTable"] th { background: var(--rose-100) !important; color: var(--rose-600) !important; font-size: .76rem !important; font-weight: 700 !important; border-bottom: 1px solid var(--rose-200) !important; }
        [data-testid="stTable"] td { background: var(--white) !important; color: var(--ink) !important; font-size: .82rem !important; border-bottom: 1px solid #f1e3e7 !important; }
        [data-testid="stTable"] tr:last-child td { border-bottom: 0 !important; }

        [data-testid="stAlert"] { border-radius: 8px !important; }
        [data-testid="stDataFrame"] { border: 1px solid var(--rose-200); border-radius: 8px; overflow: hidden; background: var(--white) !important; }
        [data-testid="stDataFrame"] [role="columnheader"] { background: var(--rose-100) !important; color: var(--rose-600) !important; }
        [data-testid="stDataFrame"] [role="gridcell"] { background: var(--white) !important; color: var(--ink) !important; }

        /* --- RESPONSIVE MOBILE --- */
        @media (max-width: 640px) {
            [data-testid="stMainBlockContainer"] { padding: 2rem 1rem !important; }
            .auth-brand { font-size: 2rem; margin-bottom: 2rem; }
            .brand-mark { width: 3.4rem; height: 3.4rem; font-size: 1.8rem; }
            [data-testid="stTabs"] button[role="tab"] { font-size: .85rem !important; padding: .6rem .5rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    initialise_session()

    settings = get_settings()

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