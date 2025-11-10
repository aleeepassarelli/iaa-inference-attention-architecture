FROM python:3.10-slim

LABEL maintainer="EAT-Lab <contact@eat-lab.org>"
LABEL description="EAT-Lab Framework - Semantic & Causal Analysis"

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Instalar pacote
RUN pip install -e .

# Expor porta para Jupyter
EXPOSE 8888

# Comando padrão
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
