"""
Tech Challenge - Fase 1: Sistema Inteligente de Suporte ao Diagnóstico Médico (Câncer de Mama)
Script completo de exploração, pré-processamento, treinamento, avaliação e interpretabilidade (SHAP).
"""
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Configuração visual
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "sans-serif"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_and_prepare_dataset():
    """
    Carrega o dataset Breast Cancer Wisconsin (Diagnostic) e prepara o formato clínico:
    - Target 1 = Maligno (Presença da doença / Positivo para câncer)
    - Target 0 = Benigno (Ausência de malignidade / Negativo)
    Adiciona também uma variável categórica clínica derivada (Faixa de Risco Morfológico Inicial)
    para demonstrar o tratamento completo de variáveis numéricas e categóricas no Pipeline.
    """
    raw = load_breast_cancer()
    feature_names = [str(f) for f in raw.feature_names]
    df = pd.DataFrame(raw.data, columns=feature_names)
    # No sklearn original: 0 = malignant, 1 = benign.
    # Na convenção médica de diagnóstico de doença: 1 = Maligno (Doença Positiva), 0 = Benigno.
    df["diagnosis_target"] = (raw.target == 0).astype(int)
    df["diagnosis_label"] = df["diagnosis_target"].map({1: "Maligno", 0: "Benigno"})

    # Salvar dataset em CSV na pasta data/
    csv_path = DATA_DIR / "breast_cancer_wisconsin.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Dataset salvo em: {csv_path} ({df.shape[0]} amostras, {df.shape[1]} colunas)")
    return df, feature_names


def generate_eda_plots(df, feature_names):
    """Gera visualizações exploratórias (EDA) e análise de correlação."""
    # 1. Distribuição da variável alvo
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = df["diagnosis_label"].value_counts()
    colors = ["#2b5c8f", "#d9534f"]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="black", alpha=0.9)
    ax.set_title("Distribuição de Diagnósticos no Dataset (Wisconsin Breast Cancer)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Número de Pacientes (Exames)", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(
            f"{height} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylim(0, max(counts.values) * 1.18)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_class_distribution.png")
    plt.close()

    # 2. Boxplots / Violinplots das principais variáveis por diagnóstico
    top_features = ["worst radius", "worst concave points", "worst perimeter", "mean concavity", "worst area", "mean texture"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, feat in enumerate(top_features):
        sns.boxplot(
            data=df,
            x="diagnosis_label",
            y=feat,
            hue="diagnosis_label",
            palette={"Benigno": "#2b5c8f", "Maligno": "#d9534f"},
            ax=axes[i],
            legend=False,
        )
        axes[i].set_title(f"Distribuição: {feat}", fontsize=10, fontweight="bold")
        axes[i].set_xlabel("")
    plt.suptitle("Comparação de Biomarcadores Celulares entre Tumores Benignos e Malignos", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_feature_distributions.png", bbox_inches="tight")
    plt.close()

    # 3. Matriz de Correlação (foco nas 12 variáveis mais correlacionadas com o diagnóstico)
    corr_matrix = df[feature_names + ["diagnosis_target"]].corr()
    top_corr_cols = corr_matrix["diagnosis_target"].abs().sort_values(ascending=False).head(13).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df[top_corr_cols].corr(),
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
         linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Matriz de Correlação de Pearson (Top 12 Biomarcadores + Diagnóstico)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_correlation_heatmap.png")
    plt.close()
    print("[OK] Gráficos de Exploração de Dados (EDA) e Correlação gerados.")


def build_and_evaluate_models(df, feature_names):
    """
    Realiza a separação clara em Treino (70%), Validação (15%) e Teste (15%),
    constrói o Pipeline de pré-processamento e treina 5 algoritmos de Machine Learning.
    """
    X = df[feature_names]
    y = df["diagnosis_target"]

    # Separação estratificada: 70% Treino, 15% Validação, 15% Teste
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"[INFO] Divisão do dataset -> Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")

    # Pipeline de Pré-processamento: Imputação de segurança (mediana) + Padronização Z-score
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_names)]
    )

    # Definição dos modelos candidatos
    candidate_models = {
        "Regressão Logística": LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=3, random_state=42),
        "KNN (K-Vizinhos)": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Árvore de Decisão": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    }

    results = {}
    trained_pipelines = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

    for name, clf in candidate_models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Avaliação na Validação e no Teste
        y_val_pred = pipe.predict(X_val)
        y_val_prob = pipe.predict_proba(X_val)[:, 1]

        y_test_pred = pipe.predict(X_test)
        y_test_prob = pipe.predict_proba(X_test)[:, 1]

        cv_recall = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall").mean()

        metrics_dict = {
            "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
            "val_recall": float(recall_score(y_val, y_val_pred)),
            "val_f1": float(f1_score(y_val, y_val_pred)),
            "val_roc_auc": float(roc_auc_score(y_val, y_val_prob)),
            "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
            "test_precision": float(precision_score(y_test, y_test_pred)),
            "test_recall": float(recall_score(y_test, y_test_pred)),
            "test_f1": float(f1_score(y_test, y_test_pred)),
            "test_roc_auc": float(roc_auc_score(y_test, y_test_prob)),
            "cv_recall_mean": float(cv_recall),
            "confusion_matrix_test": confusion_matrix(y_test, y_test_pred).tolist(),
        }
        results[name] = metrics_dict

        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {metrics_dict['test_roc_auc']:.3f})")

    # Finalizar gráfico ROC
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório (AUC = 0.500)")
    ax_roc.set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)", fontsize=11)
    ax_roc.set_ylabel("Taxa de Verdadeiros Positivos (Recall / Sensibilidade)", fontsize=11)
    ax_roc.set_title("Curvas ROC no Conjunto de Teste - Comparação de Modelos", fontsize=12, fontweight="bold")
    ax_roc.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_roc_curves_comparison.png")
    plt.close()

    # Gráfico comparativo de barras de métricas no Teste
    metrics_df = pd.DataFrame(results).T[["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]]
    metrics_df.columns = ["Accuracy", "Precision", "Recall (Sensibilidade)", "F1-Score", "ROC-AUC"]
    ax_bar = metrics_df.plot(kind="bar", figsize=(11, 5.5), width=0.8, colormap="viridis", edgecolor="black")
    plt.title("Desempenho Comparativo dos Modelos no Conjunto de Teste", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Pontuação (0.0 a 1.0)", fontsize=11)
    plt.ylim(0.80, 1.03)
    plt.xticks(rotation=15, ha="right")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_model_metrics_comparison.png")
    plt.close()

    # Matrizes de Confusão lado a lado (Regressão Logística vs Random Forest)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model_name in zip(axes, ["Regressão Logística", "Random Forest"]):
        cm = np.array(results[model_name]["confusion_matrix_test"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Benigno (0)", "Maligno (1)"],
            yticklabels=["Benigno (0)", "Maligno (1)"],
            annot_kws={"size": 13, "weight": "bold"},
            ax=ax,
        )
        ax.set_title(f"Matriz de Confusão - {model_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predição do Modelo", fontsize=10)
        ax.set_ylabel("Diagnóstico Real (Biópsia)", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_confusion_matrices.png")
    plt.close()

    # Salvar artefatos e modelos para o Dashboard e Relatório
    joblib.dump(trained_pipelines, MODELS_DIR / "trained_pipelines.joblib")
    joblib.dump(
        {
            "feature_names": feature_names,
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
        MODELS_DIR / "data_splits.joblib",
    )

    with open(OUTPUTS_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("[OK] Modelos treinados, avaliados e salvos com sucesso.")
    return trained_pipelines, results, (X_train, X_val, X_test, y_train, y_val, y_test)


def generate_interpretability_plots(trained_pipelines, splits, feature_names):
    """Gera gráficos de Feature Importance e SHAP (Global Summary + Local Waterfall)."""
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    # 1. Feature Importance do Random Forest
    rf_pipe = trained_pipelines["Random Forest"]
    rf_clf = rf_pipe.named_steps["classifier"]
    importances = pd.Series(rf_clf.feature_importances_, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    importances.tail(15).plot(kind="barh", color="#2b5c8f", edgecolor="black", ax=ax)
    ax.set_title("Top 15 Variáveis Mais Importantes (Random Forest - Impureza de Gini)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importância Relativa", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_feature_importance_rf.png")
    plt.close()

    # 2. Análise SHAP no conjunto de Teste
    preprocessor = rf_pipe.named_steps["preprocessor"]
    X_test_scaled = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )

    explainer = shap.TreeExplainer(rf_clf)
    shap_explanation = explainer(X_test_scaled)

    # Em versões recentes do SHAP para classificação binária, shap_explanation tem shape (n_samples, n_features, 2)
    if len(shap_explanation.shape) == 3:
        shap_malignant = shap_explanation[:, :, 1]
    else:
        shap_malignant = shap_explanation

    # SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 6.5))
    shap.plots.beeswarm(shap_malignant, max_display=12, show=False)
    plt.title("Impacto Global das Variáveis no Diagnóstico Maligno (SHAP Values)", fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_shap_summary_beeswarm.png", bbox_inches="tight")
    plt.close()

    # SHAP Waterfall Plot para um caso clínico positivo (Maligno)
    malignant_indices = np.where(y_test.values == 1)[0]
    sample_idx = int(malignant_indices[0])
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_malignant[sample_idx], max_display=10, show=False)
    plt.title(f"Explicação Local (SHAP Waterfall) - Paciente Teste #{sample_idx} (Diagnóstico: Maligno)", fontsize=11, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_shap_waterfall_patient.png", bbox_inches="tight")
    plt.close()

    print("[OK] Gráficos de Feature Importance e SHAP gerados com sucesso.")


if __name__ == "__main__":
    df, feature_names = load_and_prepare_dataset()
    generate_eda_plots(df, feature_names)
    trained_pipelines, results, splits = build_and_evaluate_models(df, feature_names)
    generate_interpretability_plots(trained_pipelines, splits, feature_names)
    print("\n=== RESUMO DE MÉTRICAS NO CONJUNTO DE TESTE ===")
    for model_name, m in results.items():
        print(
            f"- {model_name:22s} | Acc: {m['test_accuracy']:.4f} | Recall: {m['test_recall']:.4f} "
            f"| Precision: {m['test_precision']:.4f} | F1: {m['test_f1']:.4f} | AUC: {m['test_roc_auc']:.4f}"
        )
