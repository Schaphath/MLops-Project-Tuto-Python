#=======================#
#   Import librairies   #
#=======================#
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


#=============================#
#   Fonction d'entrainement   #
#=============================#
def compare_models(
    df: pd.DataFrame,
    target_col: str = "diagnosis",
    selected_features: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_models: bool = True,
    output_dir: str = "Save_models",
):
    if selected_features is None:
        selected_features = [
            "perimeter",
            "concave_points",
            "texture",
            "smoothness",
            "symmetry",
        ]

    available_cols = [c for c in selected_features if c in df.columns]
    if len(available_cols) != len(selected_features):
        print(
            f" /!\\ Attention : Colonnes introuvables. Utilisation des colonnes disponibles : {available_cols}"
        )
    X = df[available_cols]
    y = df[target_col]

    if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
        y = y.map({"M": 1, "B": 0}).astype(int)

    print("Distribution des classes :")
    print(f"Bénin (B=0) : {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"Malin (M=1) : {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    print(f"Train set : {X_train.shape[0]} échantillons")
    print(f"Test set  : {X_test.shape[0]} échantillons\n")

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    priors = [n_neg / len(y_train), n_pos / len(y_train)]

    base_models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Gaussian Naive Bayes": GaussianNB(priors=priors),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=1, class_weight="balanced"
            ),
            n_estimators=300,
            learning_rate=0.05,
            random_state=random_state,
        ),
    }

    pipelines = {
        name: Pipeline([("scaler", MinMaxScaler()), ("model", model)])
        for name, model in base_models.items()
    }

    print("Entraînement des modèles...\n")
    print("-" * 60)

    results = {}
    trained_pipelines = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        recall = recall_score(y_test, y_pred)

        results[name] = recall
        trained_pipelines[name] = pipeline
        print(f"{name:<25} | Recall: {recall:.4f}")

    print("-" * 60)

    results_df = pd.DataFrame(
        {"Model": list(results.keys()), "Recall": list(results.values())}
    ).sort_values("Recall", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(results_df["Recall"])

    ax.barh(
        results_df["Model"],
        results_df["Recall"],
        color=colors,
        edgecolor="black",
    )
    for i, recall in enumerate(results_df["Recall"]):
        ax.text(
            recall + 0.01, i, f"{recall:.3f}", va="center", fontweight="bold"
        )

    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_title(
        "Comparaison des modèles (Recall sur le test)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()

    best_model_name = results_df.iloc[-1]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    y_pred_best = best_pipeline.predict(X_test)

    print(f"\nMeilleur modèle : {best_model_name} (Recall: {results_df.iloc[-1]['Recall']:.4f})")

    print(f"\n{'='*60}")
    print(f"RAPPORT DETAILLE - {best_model_name}")
    print(f"{'='*60}\n")

    rapport_meilleur_model = classification_report(
        y_test,
        y_pred_best,
        target_names=["Bénin (0)", "Malin (1)"],
        digits=3,
        zero_division=0,
    )
    print(rapport_meilleur_model)

    cm = confusion_matrix(y_test, y_pred_best)
    tn, fp, fn, tp = cm.ravel()

    print(f"TN = {tn}  Vrais Négatifs -- Bénins bien classés")
    print(f"FP = {fp}  Faux Positifs -- Bénins classés malins (fausse alarme)")
    print(f"FN = {fn}  Faux Négatifs -- Malins manqués *DANGEREUX*")
    print(f"TP = {tp}  Vrais Positifs -- Malins bien détectés\n")

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Bénin (0)", "Malin (1)"]
        )
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(
        f"Matrice de confusion -- {best_model_name}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    #==================================================#
    #    Courbe d'apprentissage du meilleur modèle     #
    #==================================================#
    print("Génération de la courbe d'apprentissage...\n")

    lc_pipeline = clone(pipelines[best_model_name])

    # Appel standard et conforme à Scikit-Learn 1.8.0
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=lc_pipeline,
        X=X_train,
        y=y_train,
        cv=5,
        scoring="recall",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig_learning, ax_learning = plt.subplots(figsize=(9, 5))
    ax_learning.plot(
        train_sizes,
        train_mean,
        "o-",
        color="blue",
        label="Score entraînement",
    )
    ax_learning.plot(
        train_sizes, val_mean, "o-", color="red", label="Score validation"
    )
    ax_learning.set_xlabel(
        "Nombre d'échantillons d'entraînement", fontweight="bold"
    )
    ax_learning.set_ylabel("Recall", fontweight="bold")
    ax_learning.set_title(
        f"Courbe d'apprentissage -- {best_model_name}",
        fontsize=13,
        fontweight="bold",
    )
    ax_learning.legend(loc="lower right")
    ax_learning.grid(True, alpha=0.3, linestyle="--")
    ax_learning.set_ylim(0, 1.05)
    plt.tight_layout()

    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.1:
        print(
            f"/!\\ Écart train/validation = {gap:.3f} : le modèle sur-apprend peut-être un peu."
        )
    else:
        print(
            f"OK : écart train/validation = {gap:.3f}, pas de sur-apprentissage visible."
        )

    #====================================================#
    #    Sauvegarde du Pipeline complet avec joblib      #
    #====================================================#
    pipeline_filename = None

    if save_models:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        clean_name = best_model_name.lower().replace(" ", "_")
        pipeline_filename = output_path / f"{clean_name}_pipeline.joblib"

        joblib.dump(best_pipeline, pipeline_filename)

        print(
            f"\nPipeline complet (Scaler + Modèle) sauvegardé via Joblib : {pipeline_filename}"
        )

    return (
        results_df,
        fig,
        rapport_meilleur_model,
        fig_learning,
        fig_cm,
        pipeline_filename,
    )


# =========================#
#   Lancer l'entraînement  #
# =========================#
if __name__ == "__main__":
    df = pd.read_csv("Data/process/cancer_select.csv", sep=",")

    selected_features = [
        "perimeter",
        "concave_points",
        "texture",
        "smoothness",
        "symmetry",
    ]

    (
        results,
        fig,
        rapport,
        fig_learning,
        fig_cm,
        pipeline_filename,
    ) = compare_models(df, selected_features=selected_features)

    plt.show()