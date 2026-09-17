-- PostgreSQL Schema - OncoScan v2.0.0

-- 1. TABLE DES UTILISATEURS (Praticiens / Médecins)
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(30) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_users_username UNIQUE (username),
    CONSTRAINT chk_username_length CHECK (char_length(username) BETWEEN 3 AND 30)
);

-- 2. TABLE DES PRÉDICTIONS (Historique des analyses cytologiques)
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,

    -- Clé étrangère vers le praticien
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,

    -- Identifiant du patient pour suivi dans l'interface
    patient_id VARCHAR(50) NOT NULL,

    -- Les 5 variables cellulaires réelles (OncoScan v2)
    perimeter DOUBLE PRECISION NOT NULL
        CONSTRAINT chk_perimeter CHECK (perimeter > 40.0 AND perimeter <= 250.0),
    concave_points DOUBLE PRECISION NOT NULL
        CONSTRAINT chk_concave_points CHECK (concave_points >= 0.0 AND concave_points <= 0.3),
    texture DOUBLE PRECISION NOT NULL
        CONSTRAINT chk_texture CHECK (texture > 5.0 AND texture <= 50.0),
    smoothness DOUBLE PRECISION NOT NULL
        CONSTRAINT chk_smoothness CHECK (smoothness > 0.01 AND smoothness <= 0.3),
    symmetry DOUBLE PRECISION NOT NULL
        CONSTRAINT chk_symmetry CHECK (symmetry > 0.05 AND symmetry <= 0.7),

    -- Résultat du modèle ('M' = Maligne, 'B' = Bénigne)
    prediction CHAR(1) NOT NULL
        CONSTRAINT chk_prediction_format CHECK (prediction IN ('M', 'B')),

    -- Probabilité de malignité [0.0 - 1.0]
    probability_malignant DOUBLE PRECISION
        CONSTRAINT chk_probability_range CHECK (
            probability_malignant IS NULL OR probability_malignant BETWEEN 0.0 AND 1.0
        ),

    -- Validation/Feedback par le médecin spécialiste (Maligne, Bénigne ou Non confirmé)
    doctor_feedback VARCHAR(15) NOT NULL DEFAULT 'UNCONFIRMED'
        CONSTRAINT chk_doctor_feedback CHECK (doctor_feedback IN ('UNCONFIRMED', 'M', 'B')),

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 3. INDEX DE PRODUCTION OPTIMISÉS
-- Recherche rapide de l'historique d'un praticien trié par date
CREATE INDEX IF NOT EXISTS idx_predictions_user_date 
    ON predictions (user_id, created_at DESC);

-- Filtre par diagnostic ou alerte dans l'historique
CREATE INDEX IF NOT EXISTS idx_predictions_label 
    ON predictions (prediction, created_at DESC);

-- Recherche rapide par identifiant patient
CREATE INDEX IF NOT EXISTS idx_predictions_patient 
    ON predictions (patient_id);