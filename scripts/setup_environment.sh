#!/usr/bin/env bash
# =============================================================
# setup_environment.sh — HIGGS Big Data ML Project
# =============================================================
# Creates conda environment, installs all dependencies,
# configures Spark, and validates the setup.
# =============================================================

set -euo pipefail

ENV_NAME="higgs_bigml"
PYTHON_VERSION="3.10"

echo "=============================================="
echo "  HIGGS Big Data ML — Environment Setup"
echo "=============================================="

# ----------------------------------------------------------
# 1. Check prerequisites
# ----------------------------------------------------------
echo "[1/7] Checking prerequisites..."

command -v conda  >/dev/null 2>&1 || { echo "ERROR: conda not found. Install Miniconda first."; exit 1; }
command -v java   >/dev/null 2>&1 || { echo "ERROR: Java not found. Install Java 11+."; exit 1; }

JAVA_VER=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d. -f1)
if [ "$JAVA_VER" -lt 11 ]; then
    echo "ERROR: Java 11+ required. Found version $JAVA_VER."
    exit 1
fi
echo "    Java version: OK ($JAVA_VER)"

# ----------------------------------------------------------
# 2. Create conda environment
# ----------------------------------------------------------
echo "[2/7] Creating conda environment '$ENV_NAME'..."

if conda env list | grep -q "^$ENV_NAME "; then
    echo "    Environment already exists — skipping creation."
else
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# ----------------------------------------------------------
# 3. Install Python dependencies
# ----------------------------------------------------------
echo "[3/7] Installing Python packages..."

conda run -n "$ENV_NAME" pip install \
    pyspark==3.5.1 \
    pandas==2.1.4 \
    numpy==1.26.4 \
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
    findspark==2.0.1 \
    pytest==8.1.1 \
    pytest-cov==5.0.0 \
    black==24.3.0 \
    isort==5.13.2

echo "    Python packages installed."

# ----------------------------------------------------------
# 4. Set SPARK_HOME and JAVA_HOME
# ----------------------------------------------------------
echo "[4/7] Configuring environment variables..."

CONDA_PREFIX=$(conda run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)")
PYSPARK_PATH=$(conda run -n "$ENV_NAME" python -c "import pyspark; print(pyspark.__file__.rsplit('/',1)[0])")

cat >> ~/.bashrc << EOF

# ---- HIGGS Big Data ML Project ----
export SPARK_HOME="$PYSPARK_PATH"
export PYSPARK_PYTHON="\$(conda run -n $ENV_NAME which python)"
export PYSPARK_DRIVER_PYTHON="\$(conda run -n $ENV_NAME which python)"
export PATH="\$SPARK_HOME/bin:\$PATH"
export JAVA_HOME="\$(dirname \$(dirname \$(readlink -f \$(which java))))"
# ------------------------------------
EOF

echo "    Environment variables set in ~/.bashrc"

# ----------------------------------------------------------
# 5. Create data directories
# ----------------------------------------------------------
echo "[5/7] Creating data directories..."
mkdir -p data/{schemas,samples,models,train,val,test}
mkdir -p data/higgs_parquet
echo "    Data directories created."

# ----------------------------------------------------------
# 6. Validate Spark installation
# ----------------------------------------------------------
echo "[6/7] Validating Spark..."

conda run -n "$ENV_NAME" python - << 'PYEOF'
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("test").master("local[2]").getOrCreate()
df = spark.range(100)
assert df.count() == 100
spark.stop()
print("    Spark validation: PASSED")
PYEOF

# ----------------------------------------------------------
# 7. Final instructions
# ----------------------------------------------------------
echo "[7/7] Setup complete!"
echo ""
echo "=============================================="
echo "  To activate environment:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  To launch Jupyter:"
echo "    jupyter lab --port=8888"
echo ""
echo "  To run the full pipeline:"
echo "    python scripts/run_pipeline.py"
echo ""
echo "  Spark UI (when running):"
echo "    http://localhost:4040"
echo "=============================================="
