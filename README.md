# HIGGS Boson Detection — Big Data ML Pipeline

> **Module:** Big Data Machine Learning | **University:** Coventry University
> **Dataset:** [UCI HIGGS Dataset](https://archive.ics.uci.edu/dataset/280/higgs) (~8GB, 11M rows, 29 features)
> **Task:** Binary Classification — Higgs Signal (1) vs Background (0)

---

## Quick Setup (One Line)

### macOS / Linux
```bash
python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install pyspark==3.5.1 pandas==2.1.4 numpy==1.26.4 scipy==1.13.0 scikit-learn==1.4.2 matplotlib==3.8.4 seaborn==0.13.2 pyarrow==15.0.2 fastparquet==2024.2.0 requests==2.31.0 pyyaml==6.0.1 jupyter==1.0.0 jupyterlab==4.1.6 nbconvert==7.16.3 nbformat==5.10.4 findspark==2.0.1 pytest==8.1.1 pytest-cov==5.0.0 tqdm==4.66.2
```

### Windows (Command Prompt)
```cmd
python -m venv venv && venv\Scripts\activate && pip install --upgrade pip && pip install pyspark==3.5.1 pandas==2.1.4 numpy==1.26.4 scipy==1.13.0 scikit-learn==1.4.2 matplotlib==3.8.4 seaborn==0.13.2 pyarrow==15.0.2 fastparquet==2024.2.0 requests==2.31.0 pyyaml==6.0.1 jupyter==1.0.0 jupyterlab==4.1.6 nbconvert==7.16.3 nbformat==5.10.4 findspark==2.0.1 pytest==8.1.1 pytest-cov==5.0.0 tqdm==4.66.2
```

### Windows (PowerShell)
```powershell
python -m venv venv; venv\Scripts\Activate.ps1; pip install --upgrade pip; pip install pyspark==3.5.1 pandas==2.1.4 numpy==1.26.4 scipy==1.13.0 scikit-learn==1.4.2 matplotlib==3.8.4 seaborn==0.13.2 pyarrow==15.0.2 fastparquet==2024.2.0 requests==2.31.0 pyyaml==6.0.1 jupyter==1.0.0 jupyterlab==4.1.6 nbconvert==7.16.3 nbformat==5.10.4 findspark==2.0.1 pytest==8.1.1 pytest-cov==5.0.0 tqdm==4.66.2
```

> **Prerequisite:** Java 11 or 17 must be installed before running PySpark.
> Check with `java -version`. Download from https://adoptium.net if needed.

---

## After Setup — Run the Project

```bash
# 1. Activate the virtual environment (every new terminal session)
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 2. Launch JupyterLab
jupyter lab

# 3. OR run the full pipeline automatically
python scripts/run_pipeline.py

# 4. OR run a single notebook only (e.g. notebook 3)
python scripts/run_pipeline.py --notebook 3
```

---

## Deactivate / Reactivate venv

```bash
deactivate                        # Exit the virtual environment

source venv/bin/activate          # Re-enter (macOS / Linux)
venv\Scripts\activate             # Re-enter (Windows)
```

---

## Project Overview

This project applies **Big Data Machine Learning** techniques to the HIGGS Boson dataset, implementing a full end-to-end pipeline from raw data ingestion through to distributed model training, evaluation, and Tableau visualisation.

### Problem Statement
Identifying Higgs Boson signals in particle physics collision data is a challenging classification problem where even small improvements in detection accuracy have significant scientific value. With 11 million collision events, this dataset requires distributed computing infrastructure to process efficiently.

### Algorithms Implemented
| Algorithm | Framework | AUC-ROC (approx) |
|-----------|-----------|-----------------|
| Logistic Regression | PySpark MLlib | ~0.77 |
| Random Forest | PySpark MLlib | ~0.81 |
| **Gradient Boosted Trees** | **PySpark MLlib** | **~0.85** |
| Linear SVM | PySpark MLlib | ~0.76 |
| Logistic Regression (baseline) | scikit-learn (500K) | ~0.75 |
| Random Forest (baseline) | scikit-learn (500K) | ~0.79 |

---

## Project Structure

```
higgs_project/
├── notebooks/
│   ├── 1_data_ingestion.ipynb        # Download, validate, Parquet storage
│   ├── 2_feature_engineering.ipynb   # EDA, custom transformer, splits
│   ├── 3_model_training.ipynb        # 4 MLlib models + sklearn baseline
│   └── 4_evaluation.ipynb            # Metrics, ROC, CI, scalability
├── tableau/
│   ├── dashboard1.twbx               # Data quality & pipeline monitoring
│   ├── dashboard2.twbx               # Model performance & feature importance
│   ├── dashboard3.twbx               # Business insights
│   ├── dashboard4.twbx               # Scalability & cost analysis
│   └── README_tableau.md             # Tableau setup instructions
├── scripts/
│   ├── setup_environment.sh          # Conda env + dependencies
│   ├── run_pipeline.py               # End-to-end pipeline runner
│   └── performance_profiler.py       # Spark UI profiling tool
├── config/
│   ├── spark_config.yaml             # Documented Spark settings
│   └── tableau_config.json           # Dashboard data source config
├── data/
│   ├── schemas/                      # Parquet schema definitions
│   └── samples/                      # 50K row sample + plots
├── tests/
│   └── test_pipeline.py              # 20+ unit & integration tests
├── .gitignore
├── environment.yml                   # Conda environment specification
├── Dockerfile                        # Containerised deployment
└── README.md
```

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.10.13 |
| Java | 11 | 17 |
| RAM | 16 GB | 32 GB |
| Disk | 15 GB | 25 GB |
| CPU Cores | 4 | 8+ |

---

## Dataset Details

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/280/higgs |
| Size | ~8 GB (uncompressed CSV) |
| Rows | 11,000,000 |
| Columns | 29 (1 label + 28 features) |
| Task | Binary classification |
| Class balance | ~53% background, ~47% signal |

**Feature groups:**
- **Low-level (cols 1–21):** Raw detector measurements — lepton momenta, jet properties, missing energy
- **High-level (cols 22–28):** Derived invariant masses (m_bb, m_wwbb, m_jj, etc.) computed by physicists

**License:** Creative Commons — free for academic use. Cite: Baldi et al. (2014), *Nature Communications*.

---

## Technical Highlights

### PySpark Engineering
- **Explicit schema** avoids costly inference scan on 8GB file
- **Parquet + Snappy** compression: ~3x size reduction, columnar reads
- **Partition by label** aligns with ML train/val/test query patterns
- **AQE enabled** for dynamic partition coalescing
- **Kryo serializer** ~10x faster than Java default for ML vectors

### Custom Transformer
`PhysicsFeatureEngineer` adds 5 domain-specific features:
- `lepton_to_jet_pT_ratio` — momentum transfer proxy
- `total_visible_pT` — total jet energy
- `bjet_multiplicity` — b-tagging count
- `mass_ratio_bb_wwbb` — Higgs decay signature ratio
- `delta_eta_leading_jets` — VBF topology indicator

### Distributed Training
- All 4 MLlib models use `CrossValidator` with `parallelism=4`
- Hyperparameter grids designed within computational constraints
- Models serialised via MLlib's native save/load API
- Sklearn models serialised via Pickle for comparison

---

## Running Tests

```bash
# Activate venv first
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# Run all tests
pytest tests/test_pipeline.py -v

# With coverage report
pytest tests/test_pipeline.py --cov=scripts --cov-report=html -v

# Run a specific test class only
pytest tests/test_pipeline.py::TestModelTraining -v
```

---

## Performance Profiling

```bash
# Generate mock Spark performance profile (no live Spark needed)
python scripts/performance_profiler.py --mock

# Profile a live running Spark job
python scripts/performance_profiler.py --live

# Profile a completed job by app ID
python scripts/performance_profiler.py --app-id application_1234567890_0001
```

---

## Docker (Alternative — Fully Reproducible)

```bash
# Build image
docker build -t higgs-bigml .

# Run container
docker run -p 8888:8888 -p 4040:4040 higgs-bigml

# Open JupyterLab at http://localhost:8888
# Spark UI available at http://localhost:4040
```

---

## Tableau Dashboards

See `tableau/README_tableau.md` for detailed step-by-step build instructions.

Data sources generated by the pipeline (`data/samples/`):

| CSV File | Dashboard |
|----------|-----------|
| `higgs_sample_50k.csv` | Dashboard 1 — Data Quality |
| `test_evaluation_results.csv` | Dashboard 2 — Model Performance |
| `business_cost_results.csv` | Dashboard 3 — Business Insights |
| `scaling_results.csv` | Dashboard 4 — Scalability |

---

## Key Results Summary

- **Best model:** GBT with AUC-ROC ≈ 0.85
- **Scalability:** Near-linear strong scaling from 50→200 partitions; sub-linear at 400 (shuffle bottleneck)
- **Stability:** GBT AUC varies <0.005 under 20% data perturbation — highly stable
- **Business insight:** Optimal threshold = 0.42 minimises physics experiment cost (3:1 FN:FP ratio)
- **MLlib vs sklearn:** MLlib processes full 11M rows; sklearn capped at 500K — MLlib achieves higher AUC on full data

---

## References

- Baldi, P., Sadowski, P., & Whiteson, D. (2014). Searching for exotic particles in high-energy physics with deep learning. *Nature Communications*, 5, 4308.
- Zaharia, M., et al. (2016). Apache Spark: A unified engine for big data processing. *Communications of the ACM*, 59(11), 56–65.
- Meng, X., et al. (2016). MLlib: Machine learning in Apache Spark. *Journal of Machine Learning Research*, 17(34), 1–7.

---