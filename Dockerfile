# Imagem base oficial do Python
FROM python:3.11-slim

# Evita geração de arquivos .pyc e ativa logs em tempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependências básicas de sistema necessárias para compilação científica
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o código-fonte do projeto
COPY . .

# Executa o pipeline de treinamento/avaliação para garantir todos os modelos e figuras atualizados
RUN python src/train_pipeline.py

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Comando padrão: inicia o Dashboard Interativo de Suporte ao Diagnóstico Médico
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
