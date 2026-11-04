"""
Dashboard Interativo de Suporte ao Diagnóstico Médico - Câncer de Mama
POSTECH - Tech Challenge (Fase 1)
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "breast_cancer_wisconsin.csv"
MODELS_PATH = BASE_DIR / "models" / "trained_pipelines.joblib"
SPLITS_PATH = BASE_DIR / "models" / "data_splits.joblib"
METRICS_PATH = BASE_DIR / "outputs" / "metrics_summary.json"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

st.set_page_config(
    page_title="CDSS - Triagem Oncológica (Tech Challenge Fase 1)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return df, metrics


@st.cache_resource
def load_models_and_splits():
    pipelines = joblib.load(MODELS_PATH)
    splits = joblib.load(SPLITS_PATH)
    return pipelines, splits


df, metrics = load_data()
pipelines, splits = load_models_and_splits()
feature_names = splits["feature_names"]
X_test = splits["X_test"]
y_test = splits["y_test"]

# Sidebar
st.sidebar.title("🏥 Sistema CDSS Oncológico")
st.sidebar.markdown("**POSTECH - Tech Challenge | Fase 1**")
st.sidebar.markdown("---")
st.sidebar.info(
    "🩺 **Aviso Clínico Importante:**\n\n"
    "Este sistema de Inteligência Artificial atua exclusivamente como **suporte à triagem e segunda opinião**. "
    "**A decisão final do diagnóstico cabe sempre ao médico responsável.**"
)

st.sidebar.subheader("⚙️ Configurações de Triagem")
selected_model_name = st.sidebar.selectbox(
    "Algoritmo de Classificação:",
    list(pipelines.keys()),
    index=0,
)
threshold = st.sidebar.slider(
    "Limiar de Sensibilidade Clínica (Threshold para Maligno):",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05,
    help="Reduzir o limiar aumenta o Recall (Sensibilidade), evitando falsos negativos na triagem.",
)

st.title("🔬 Sistema Inteligente de Suporte ao Diagnóstico de Câncer de Mama")
st.markdown(
    "Análise automatizada de biópsias por Punção Aspirativa por Agulha Fina (PAAF / FNA) "
    "utilizando **Machine Learning** e **Explicabilidade Clínica (SHAP)**."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🧑‍⚕️ Triagem de Paciente & Explicabilidade (SHAP)",
        "📊 Exploração de Dados (EDA) & Correlação",
        "🤖 Desempenho e Comparação de Modelos",
        "📋 Discussão Clínica e Arquitetura",
    ]
)

with tab1:
    st.subheader("Simulador de Triagem Clínica e Segunda Opinião Médica")
    col_sel, col_custom = st.columns([1.2, 1])

    with col_sel:
        malignant_test_indices = y_test[y_test == 1].index.tolist()
        benign_test_indices = y_test[y_test == 0].index.tolist()

        preset_options = {
            f"Caso Clínico #{malignant_test_indices[0]} (Biópsia Real: Maligno - Alta Gravidade)": malignant_test_indices[0],
            f"Caso Clínico #{malignant_test_indices[1]} (Biópsia Real: Maligno - Moderado)": malignant_test_indices[1],
            f"Caso Clínico #{benign_test_indices[0]} (Biópsia Real: Benigno - Baixo Risco)": benign_test_indices[0],
            f"Caso Clínico #{benign_test_indices[1]} (Biópsia Real: Benigno - Limítrofe)": benign_test_indices[1],
        }
        chosen_preset = st.selectbox("Selecione um Prontuário do Conjunto de Teste:", list(preset_options.keys()))
        patient_idx = preset_options[chosen_preset]
        patient_row = X_test.loc[[patient_idx]].copy()
        real_label = "Maligno (Positivo)" if y_test.loc[patient_idx] == 1 else "Benigno (Negativo)"

    with col_custom:
        st.markdown("**Ajuste Fino de Biomarcadores Principais (Simulação Interativa):**")
        c1, c2 = st.columns(2)
        with c1:
            val_worst_radius = st.number_input(
                "Worst Radius (mm)",
                value=float(patient_row["worst radius"].iloc[0]),
                min_value=5.0,
                max_value=40.0,
                step=0.5,
            )
            val_worst_concave = st.number_input(
                "Worst Concave Points",
                value=float(patient_row["worst concave points"].iloc[0]),
                min_value=0.0,
                max_value=0.35,
                step=0.01,
                format="%.4f",
            )
        with c2:
            val_worst_perimeter = st.number_input(
                "Worst Perimeter (mm)",
                value=float(patient_row["worst perimeter"].iloc[0]),
                min_value=40.0,
                max_value=260.0,
                step=2.0,
            )
            val_mean_texture = st.number_input(
                "Mean Texture",
                value=float(patient_row["mean texture"].iloc[0]),
                min_value=5.0,
                max_value=45.0,
                step=0.5,
            )

        patient_row["worst radius"] = val_worst_radius
        patient_row["worst concave points"] = val_worst_concave
        patient_row["worst perimeter"] = val_worst_perimeter
        patient_row["mean texture"] = val_mean_texture

    # Predição com o modelo selecionado
    pipe = pipelines[selected_model_name]
    prob_malignant = float(pipe.predict_proba(patient_row)[0, 1])
    pred_is_malignant = prob_malignant >= threshold

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Modelo Ativo", selected_model_name)
    m2.metric("Probabilidade de Malignidade", f"{prob_malignant * 100:.1f}%")
    m3.metric(
        "Sugestão da Triagem IA",
        "🚨 ALTO RISCO (Maligno)" if pred_is_malignant else "✅ BAIXO RISCO (Benigno)",
        delta="Encaminhar Oncologia" if pred_is_malignant else "Acompanhamento Padrão",
        delta_color="inverse" if pred_is_malignant else "normal",
    )
    m4.metric("Laudo Histopatológico Real", real_label)

    st.markdown("#### 🔍 Explicabilidade da Predição para o Médico (SHAP Local & Biomarcadores)")
    col_shap1, col_shap2 = st.columns([1.2, 1])
    with col_shap1:
        # Cálculo em tempo real da contribuição para este paciente usando o Random Forest TreeExplainer
        rf_pipe = pipelines["Random Forest"]
        rf_prep = rf_pipe.named_steps["preprocessor"]
        rf_clf = rf_pipe.named_steps["classifier"]
        X_patient_scaled = pd.DataFrame(rf_prep.transform(patient_row), columns=feature_names)
        explainer = shap.TreeExplainer(rf_clf)
        shap_vals = explainer(X_patient_scaled)
        if len(shap_vals.shape) == 3:
            sv = shap_vals[0, :, 1].values
        else:
            sv = shap_vals[0, :].values

        contrib_df = (
            pd.DataFrame({"Biomarcador": feature_names, "Impacto SHAP": sv, "Valor Medido": patient_row.iloc[0].values})
            .assign(AbsImpact=lambda d: d["Impacto SHAP"].abs())
            .sort_values("AbsImpact", ascending=True)
            .tail(10)
        )
        contrib_df["Direção"] = np.where(contrib_df["Impacto SHAP"] > 0, "Aumenta Risco Maligno (+)", "Reduz Risco / Favorece Benigno (-)")
        fig_local = px.bar(
            contrib_df,
            x="Impacto SHAP",
            y="Biomarcador",
            color="Direção",
            orientation="h",
            hover_data=["Valor Medido"],
            color_discrete_map={
                "Aumenta Risco Maligno (+)": "#d9534f",
                "Reduz Risco / Favorece Benigno (-)": "#2b5c8f",
            },
            title=f"Top 10 Biomarcadores que Explicam o Diagnóstico do Caso #{patient_idx}",
        )
        fig_local.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_local, use_container_width=True)

    with col_shap2:
        st.image(
            str(FIGURES_DIR / "08_shap_summary_beeswarm.png"),
            caption="Visão Global SHAP (Beeswarm): Como cada variável afeta o risco em toda a população.",
            use_container_width=True,
        )

with tab2:
    st.subheader("📊 Exploração da Base de Dados e Análise de Correlação")
    c_info1, c_info2, c_info3, c_info4 = st.columns(4)
    c_info1.metric("Total de Exames (Pacientes)", len(df))
    c_info2.metric("Biomarcadores Avaliados", len(feature_names))
    c_info3.metric("Casos Benignos (0)", f"{int((df['diagnosis_target'] == 0).sum())} (62.7%)")
    c_info4.metric("Casos Malignos (1)", f"{int((df['diagnosis_target'] == 1).sum())} (37.3%)")

    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.image(str(FIGURES_DIR / "01_class_distribution.png"), use_container_width=True)
    with col_eda2:
        st.image(str(FIGURES_DIR / "03_correlation_heatmap.png"), use_container_width=True)

    st.image(str(FIGURES_DIR / "02_feature_distributions.png"), use_container_width=True)

    with st.expander("🔎 Visualizar Amostra dos Dados e Estatísticas Descritivas"):
        st.dataframe(df.head(15), use_container_width=True)
        st.dataframe(df[feature_names].describe().T, use_container_width=True)

with tab3:
    st.subheader("🤖 Avaliação Comparativa dos Modelos de Machine Learning")
    st.markdown(
        "Os dados foram divididos de forma estratificada em **Treino (70% - 398 exames)**, "
        "**Validação (15% - 85 exames)** e **Teste (15% - 86 exames)**."
    )

    rows = []
    for m_name, m_vals in metrics.items():
        rows.append(
            {
                "Modelo": m_name,
                "Accuracy (Teste)": f"{m_vals['test_accuracy']*100:.2f}%",
                "Recall / Sensibilidade (Teste)": f"{m_vals['test_recall']*100:.2f}%",
                "Precision (Teste)": f"{m_vals['test_precision']*100:.2f}%",
                "F1-Score (Teste)": f"{m_vals['test_f1']*100:.2f}%",
                "ROC-AUC (Teste)": f"{m_vals['test_roc_auc']:.4f}",
                "Recall Médio (5-Fold CV)": f"{m_vals['cv_recall_mean']*100:.2f}%",
            }
        )
    st.table(pd.DataFrame(rows))

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.image(str(FIGURES_DIR / "04_roc_curves_comparison.png"), use_container_width=True)
    with col_m2:
        st.image(str(FIGURES_DIR / "05_model_metrics_comparison.png"), use_container_width=True)

    col_m3, col_m4 = st.columns(2)
    with col_m3:
        st.image(str(FIGURES_DIR / "06_confusion_matrices.png"), use_container_width=True)
    with col_m4:
        st.image(str(FIGURES_DIR / "07_feature_importance_rf.png"), use_container_width=True)

with tab4:
    st.subheader("📋 Discussão Crítica: Escolha de Métricas e Aplicação Prática no Hospital")
    st.markdown(
        """
        ### 1. Por que o **Recall (Sensibilidade)** e o **F1-Score** são prioritários?
        Em um cenário de triagem oncológica hospitalar, os dois tipos de erro possuem custos humanos drasticamente diferentes:
        - **Falso Positivo (Erro Tipo I):** Um nódulo benigno é sinalizado como suspeito. O paciente passa por revisão prioritária pelo médico patologista ou exame complementar. Há custo operacional e ansiedade temporária, mas a vida do paciente está protegida.
        - **Falso Negativo (Erro Tipo II):** Um tumor **maligno** é classificado como benigno. Isso atrasa o início do tratamento oncológico, podendo levar à progressão metastática.

        Portanto, priorizamos a maximização do **Recall (Sensibilidade)** combinada a um alto **F1-Score** e **ROC-AUC**. No conjunto de teste:
        - A **Regressão Logística** atingiu **100.0% de Recall** (zero falsos negativos em 32 casos malignos reais) e **98.84% de Accuracy** (AUC = 0.9988).
        - O **Random Forest** atingiu **100.0% de Precision**, **96.88% de Recall** e **ROC-AUC = 1.0000**.

        ---
        ### 2. O modelo pode ser utilizado na prática clínica? Como?
        **Sim, como Sistema de Suporte à Decisão Clínica (CDSS) na triagem**, seguindo boas práticas médicas e regulatórias:
        1. **Priorização de Fila (Triagem Inteligente):** Exames com probabilidade de malignidade $\\ge 0.40$ recebem alerta de prioridade máxima no prontuário eletrônico para laudagem imediata pelo médico patologista/oncologista.
        2. **Explicabilidade Obrigatória (SHAP):** O sistema não entrega uma "caixa-preta"; junto ao score de risco, exibe quais variáveis celulares (como `worst radius`, `worst concave points` e `worst perimeter`) motivaram o alerta.
        3. **Soberania Médica (Human-in-the-Loop):** A Inteligência Artificial atua como segunda opinião automatizada para reduzir fadiga e erros de triagem, mas **o médico sempre detém a palavra final no diagnóstico e na conduta terapêutica**.
        """
    )
