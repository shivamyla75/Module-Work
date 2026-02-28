# ============================================================
# Dockerfile — HIGGS Boson Big Data ML Project
# ============================================================
# Multi-stage build:
#   Stage 1: base  — Java + Python dependencies
#   Stage 2: final — project files + Jupyter config
# ============================================================

FROM python:3.10.13-slim AS base

LABEL maintainer="HIGGS ML Project"
LABEL description="PySpark + scikit-learn pipeline for HIGGS Boson classification"

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    curl \
    wget \
    procps \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Java environment ----
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ---- Python environment ----
WORKDIR /app

COPY environment.yml /tmp/environment.yml

# Install pip packages from environment.yml pip section
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pyspark==3.5.1 \
        pandas==2.1.4 \
        numpy==1.26.4 \
        scipy==1.13.0 \
        scikit-learn==1.4.2 \
        matplotlib==3.8.4 \
        seaborn==0.13.2 \
        pyarrow==15.0.2 \
        fastparquet==2024.2.0 \
        requests==2.31.0 \
        pyyaml==6.0.1 \
        jupyter==1.0.0 \
        notebook==7.1.3 \
        jupyterlab==4.1.6 \
        nbconvert==7.16.3 \
        nbformat==5.10.4 \
        findspark==2.0.1 \
        pytest==8.1.1 \
        pytest-cov==5.0.0 \
        tqdm==4.66.2

# ---- Spark configuration ----
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV SPARK_LOCAL_IP=127.0.0.1

# ============================================================
FROM base AS final

# ---- Project files ----
COPY . /app/

# ---- Directory structure ----
RUN mkdir -p /app/data/{schemas,samples,models,train,val,test,higgs_parquet} && \
    chmod -R 755 /app/data && \
    chmod +x /app/scripts/setup_environment.sh

# ---- Jupyter configuration ----
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> ~/.jupyter/jupyter_lab_config.py

# ---- Ports ----
# 8888: JupyterLab
# 4040: Spark UI
# 18080: Spark History Server
EXPOSE 8888 4040 18080

# ---- Health check ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1

# ---- Entrypoint ----
WORKDIR /app

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/app/notebooks"]
