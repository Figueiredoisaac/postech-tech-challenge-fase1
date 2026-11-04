"""
Gera o Relatório Técnico Oficial em PDF (Relatorio_Tecnico_Fase1.pdf)
contendo os links de entrega (GitHub e YouTube), estratégias de pré-processamento,
justificativa dos modelos, resultados, gráficos e interpretação com Feature Importance e SHAP.
"""
import argparse
from pathlib import Path
from fpdf import FPDF

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "outputs" / "figures"


class TechChallengePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 6, "POSTECH - Tech Challenge | Fase 1 - Sistema Inteligente de Suporte ao Diagnostico Medico", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, 16, 200, 16)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Pagina {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(43, 92, 143)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(20, 20, 20)
        self.ln(2)

    def subsection_title(self, title: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(43, 92, 143)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(20, 20, 20)
        self.ln(1)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.2, text)
        self.ln(1.5)

    def bullet_item(self, bold_prefix: str, text: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        self.write(5.2, f"  * {bold_prefix}: ")
        self.set_font("Helvetica", "", 10)
        self.write(5.2, f"{text}\n")
        self.ln(1)


def generate_report(github_url: str, youtube_url: str, author_name: str, output_filename: str = "Relatorio_Tecnico_Fase1.pdf"):
    pdf = TechChallengePDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ================= PÁGINA 1: CAPA / LINKS E CONTEXTO =================
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(25, 55, 95)
    pdf.multi_cell(0, 9, "TECH CHALLENGE - FASE 1\nSistema Inteligente de Suporte ao Diagnostico Medico\n(Classificacao de Cancer de Mama com Machine Learning e SHAP)", align="C")
    pdf.ln(3)

    if author_name:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 6, f"Autor(a): {author_name}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # Quadro de Links Obrigatórios
    pdf.set_fill_color(240, 245, 250)
    pdf.set_draw_color(43, 92, 143)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(25, 55, 95)
    pdf.cell(0, 7, "  LINKS OFICIAIS DE ENTREGA (REPOSITORIO GIT E VIDEO DE DEMONSTRACAO)", border="TLR", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(
        0,
        5.5,
        f"  - Link do Repositorio Git (Codigo, Dockerfile, README, Notebook): {github_url}\n"
        f"  - Link do Video de Demonstracao (YouTube / Vimeo - ate 15 min): {youtube_url}\n"
        f"  - Dataset Publico Utilizado: Breast Cancer Wisconsin (Diagnostic) - UCI / Kaggle\n"
        f"    (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)",
        border="BLR",
        fill=True,
    )
    pdf.ln(4)

    pdf.section_title("1. Contextualizacao e Problema Clinico")
    pdf.body_text(
        "Um grande hospital universitario enfrenta volume crescente de pacientes e exames, demandando solucoes de "
        "Inteligencia Artificial que acelerem a triagem clinica e reduzam erros diagnosticos. Nesta primeira fase, "
        "desenvolvemos a base do Sistema de Suporte a Decisao Clinica (CDSS) voltado ao diagnostico de Cancer de Mama "
        "a partir de dados tabulares extraidos de imagens digitalizadas de biopsias por Puncao Aspirativa por Agulha Fina (PAAF / FNA)."
    )
    pdf.body_text(
        "A base Breast Cancer Wisconsin (Diagnostic) possui 569 exames clinicos e 30 variaveis morfometricas continuas "
        "que descrevem propriedades dos nucleos celulares (raio, textura, perimetro, area, suavidade, compacidade, "
        "concavidade, pontos concavos, simetria e dimensao fractal, calculados em media, erro padrao e pior valor/extremo). "
        "A variavel alvo classifica cada exame em:"
    )
    pdf.bullet_item("Maligno (Classe 1 - Positivo para Doenca)", "212 pacientes (37,26% da base).")
    pdf.bullet_item("Benigno (Classe 0 - Ausencia de Malignidade)", "357 pacientes (62,74% da base).")

    pdf.section_title("2. Exploracao de Dados (EDA) e Analise de Correlacao")
    pdf.body_text(
        "A exploracao estatistica revelou ausencia de valores nulos ou registros duplicados nos 569 prontuarios. "
        "A analise das distribuicoes demonstra forte separacao morfologica entre tumores benignos e malignos: "
        "lesoes malignas apresentam valores significativamente superiores em 'worst radius' (media de 21,13 mm nos malignos "
        "contra 13,38 mm nos benignos), 'worst perimeter' e 'worst concave points'. "
        "Na matriz de correlacao de Pearson, os biomarcadores com maior correlacao positiva com a malignidade sao: "
        "'worst concave points' (r = 0,79), 'worst perimeter' (r = 0,78), 'mean concave points' (r = 0,78) e 'worst radius' (r = 0,78)."
    )

    # Inserir figuras 01 e 03 lado a lado
    y_fig = pdf.get_y() + 2
    pdf.image(str(FIGURES_DIR / "01_class_distribution.png"), x=10, y=y_fig, w=88)
    pdf.image(str(FIGURES_DIR / "03_correlation_heatmap.png"), x=102, y=y_fig, w=98)
    pdf.set_y(y_fig + 78)

    # ================= PÁGINA 2: PRÉ-PROCESSAMENTO E MODELOS =================
    pdf.add_page()
    pdf.image(str(FIGURES_DIR / "02_feature_distributions.png"), x=24, w=162)
    pdf.ln(1)

    pdf.section_title("3. Estrategias de Pre-processamento e Pipeline em Python")
    pdf.body_text(
        "Para garantir rigor metodologico e impedir qualquer vazamento de dados (data leakage) entre os conjuntos de "
        "treinamento e avaliacao, implementamos um Pipeline estruturado com scikit-learn (ColumnTransformer + Pipeline):"
    )
    pdf.bullet_item(
        "Separacao Clara e Estratificada (Treino, Validacao e Teste)",
        "O dataset foi dividido mantendo a proporcao exata de 37,3% de casos malignos em tres subconjuntos independentes: "
        "Treino (70% - 398 exames), Validacao (15% - 85 exames) e Teste Cego (15% - 86 exames).",
    )
    pdf.bullet_item(
        "Limpeza e Imputacao Preventiva (SimpleImputer)",
        "Inclusao de imputacao mediana para variaveis numericas e moda para variaveis categoricas no pipeline, garantindo "
        "resiliencia em producao caso um equipamento laboratorial omita alguma medicao.",
    )
    pdf.bullet_item(
        "Conversao e Padronizacao (StandardScaler e OneHotEncoder)",
        "Como as variaveis possuem ordens de grandeza muito distintas (ex.: 'worst area' chega a 4254, enquanto 'smoothness' "
        "e da ordem de 0,09), aplicamos padronizacao Z-score (media 0 e desvio padrao 1) ajustada exclusivamente no Treino. "
        "No Notebook Jupyter demonstramos tambem a codificacao de faixas categoricas via OneHotEncoder.",
    )

    pdf.section_title("4. Modelos Utilizados e Justificativa Tecnica")
    pdf.body_text(
        "Treinamos e comparamos cinco algoritmos de classificacao supervisionada com naturezas matematicas complementares:"
    )
    pdf.bullet_item(
        "1. Regressao Logistica (class_weight='balanced')",
        "Modelo linear probabilistico altamente calibrado e interpretavel, ideal para fronteiras de decisao no espaco padronizado.",
    )
    pdf.bullet_item(
        "2. Random Forest Classifier (200 arvores)",
        "Ensemble via bagging que captura interacoes nao-lineares entre biomarcadores e e robusto a multicolinearidade.",
    )
    pdf.bullet_item(
        "3. Gradient Boosting Classifier",
        "Ensemble sequencial baseado em boosting de gradiente, focado na correcao iterativa de erros residuais.",
    )
    pdf.bullet_item(
        "4. KNN (K-Nearest Neighbors, k=7 ponderado pela distancia)",
        "Classificador baseado em similaridade geometrica local entre prontuarios de pacientes no espaco multidimensional.",
    )
    pdf.bullet_item(
        "5. Arvore de Decisao (Decision Tree)",
        "Modelo base de regras hierarquicas diretas para comparacao de linha de base (baseline).",
    )

    # ================= PÁGINA 3: RESULTADOS, MÉTRICAS E INTERPRETAÇÃO =================
    pdf.add_page()
    pdf.section_title("5. Avaliacao dos Modelos e Discussao da Escolha das Metricas")
    pdf.body_text(
        "No dominio medico oncologico, os erros de classificacao possuem consequencias assimetricas criticas:\n"
        "- Falso Positivo (Erro Tipo I): Um nodulo benigno recebe alerta de suspeita e passa por revisao prioritaria do medico patologista. "
        "Gera custo de revisao, mas nao coloca a vida do paciente em risco.\n"
        "- Falso Negativo (Erro Tipo II): Um tumor MALIGNO e classificado como benigno. Isso pode postergar a biopsia confirmativa "
        "e o tratamento oncologico, agravando drasticamente o prognostico do paciente.\n\n"
        "Por esse motivo, o Recall (Sensibilidade = VP / [VP + FN]) e a metrica primordial de seguranca clinica, apoiada pelo "
        "F1-Score (media harmonica entre Precision e Recall) e pela area sob a curva ROC (ROC-AUC)."
    )

    # Tabela de Resultados no Conjunto de Teste
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_fill_color(43, 92, 143)
    pdf.set_text_color(255, 255, 255)
    col_widths = [46, 26, 28, 26, 24, 20, 20]
    headers = ["Modelo", "Accuracy", "Recall (Sens.)", "Precision", "F1-Score", "ROC-AUC", "CV Recall"]
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 7, h, border=1, align="C", fill=True)
    pdf.ln()

    rows = [
        ["Regressao Logistica", "98,84%", "100,00% (32/32)", "96,97%", "98,46%", "0,9988", "95,93%"],
        ["Random Forest", "98,84%", "96,88% (31/32)", "100,00%", "98,41%", "1,0000", "93,86%"],
        ["Gradient Boosting", "96,51%", "93,75% (30/32)", "96,77%", "95,24%", "0,9988", "90,46%"],
        ["KNN (K-Vizinhos)", "96,51%", "90,62% (29/32)", "100,00%", "95,08%", "0,9994", "92,51%"],
        ["Arvore de Decisao", "93,02%", "90,62% (29/32)", "90,62%", "90,62%", "0,9126", "91,84%"],
    ]
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(20, 20, 20)
    for idx, r in enumerate(rows):
        fill = idx % 2 == 1
        if fill:
            pdf.set_fill_color(242, 246, 250)
        for w, val in zip(col_widths, r):
            pdf.cell(w, 6.5, val, border=1, align="C", fill=fill)
        pdf.ln()
    pdf.ln(3)

    y_fig3 = pdf.get_y()
    pdf.image(str(FIGURES_DIR / "04_roc_curves_comparison.png"), x=10, y=y_fig3, w=92)
    pdf.image(str(FIGURES_DIR / "06_confusion_matrices.png"), x=104, y=y_fig3 + 4, w=96)
    pdf.set_y(y_fig3 + 74)

    pdf.body_text(
        "Destaque dos Resultados: A Regressao Logistica atingiu 100% de Recall no conjunto de teste (identificou corretamente "
        "todos os 32 casos malignos, com zero falsos negativos e apenas 1 falso positivo em 54 benignos) e 98,84% de Accuracy. "
        "O Random Forest alcancou ROC-AUC perfeito (1,0000), 100% de Precision e 98,84% de Accuracy."
    )

    # ================= PÁGINA 4: INTERPRETABILIDADE (SHAP) E DISCUSSÃO CRÍTICA =================
    pdf.add_page()
    pdf.section_title("6. Interpretacao dos Resultados (Feature Importance e SHAP)")
    pdf.body_text(
        "Para eliminar o efeito 'caixa-preta' e fornecer transparencia clinica ao medico, aplicamos Feature Importance "
        "(Impureza de Gini no Random Forest) e valores SHAP (SHapley Additive exPlanations):"
    )
    pdf.bullet_item(
        "Visao Global (Feature Importance & SHAP Beeswarm)",
        "As variaveis mais determinantes para malignidade sao 'worst perimeter', 'worst area', 'mean concave points', "
        "'worst concave points' e 'worst radius'. No grafico SHAP Beeswarm (pontos vermelhos a direita), valores elevados "
        "de pontos concavos e dimensoes nucleares extremas deslocam fortemente a predicao para maligno.",
    )
    pdf.bullet_item(
        "Explicacao Local por Paciente (SHAP Waterfall)",
        "Decomposicao individual de quanto cada biomarcador do paciente somou ou subtraiu da probabilidade base.",
    )

    y_fig4 = pdf.get_y() + 1
    pdf.image(str(FIGURES_DIR / "07_feature_importance_rf.png"), x=10, y=y_fig4, w=82)
    pdf.image(str(FIGURES_DIR / "08_shap_summary_beeswarm.png"), x=96, y=y_fig4, w=94)
    pdf.set_y(y_fig4 + 54)

    pdf.image(str(FIGURES_DIR / "09_shap_waterfall_patient.png"), x=48, w=110)
    pdf.ln(1)

    pdf.section_title("7. Discussao Critica: Aplicacao Pratica e Papel do Medico")
    pdf.body_text(
        "O modelo pode ser utilizado na pratica hospitalar? Sim, integrado como Sistema de Suporte a Decisao Clinica (CDSS):\n"
        "1. Fluxo Operacional de Triagem: Ao concluir a morfometria digital da lamina de PAAF, o pipeline atribui um Score de "
        "Prioridade Oncologica. Exames com probabilidade >= 0,40 recebem marcacao de urgencia na fila do patologista.\n"
        "2. Segunda Opiniao Auditavel: No Dashboard interativo (Streamlit) desenvolvido neste projeto, o medico inspeciona a "
        "probabilidade e o grafico SHAP individual do paciente, prevenindo erros por fadiga em rotinas de alto volume.\n"
        "3. Soberania Medica (Human-in-the-Loop): O modelo avalia apenas variaveis citologicas. A correlacao com exame fisico, "
        "imagem (BI-RADS), historico familiar e imuno-histoquimica e indispensavel - o medico sempre tem a palavra final."
    )

    out_path = BASE_DIR / output_filename
    pdf.output(str(out_path))
    print(f"[OK] Relatorio Tecnico PDF gerado com sucesso em: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--github",
        default="https://github.com/SEU-USUARIO/postech-tech-challenge-fase1 (Substituir pelo seu link final)",
    )
    parser.add_argument(
        "--youtube",
        default="https://youtu.be/SEU-LINK-DO-VIDEO (Substituir apos upload no YouTube)",
    )
    parser.add_argument("--author", default="")
    args = parser.parse_args()
    generate_report(args.github, args.youtube, args.author)
