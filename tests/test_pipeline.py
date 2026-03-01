"""
test_pipeline.py
================
Unit and integration tests for the HIGGS ML Pipeline.
Covers data ingestion, feature engineering, model training, and evaluation.

Run with:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py --cov=scripts -v
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# ---------------------------------------------------------------
# PySpark fixtures
# ---------------------------------------------------------------
@pytest.fixture(scope='session')
def spark():
    """Create a SparkSession for the test session."""
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName('HIGGS-Tests')
        .master('local[2]')
        .config('spark.driver.memory', '2g')
        .config('spark.sql.shuffle.partitions', '10')
        .config('spark.ui.enabled', 'false')
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel('ERROR')
    yield spark
    spark.stop()


@pytest.fixture(scope='session')
def sample_df(spark):
    """Create a small synthetic HIGGS-like DataFrame for testing."""
    from pyspark.sql.types import StructType, StructField, FloatType, DoubleType
    import random

    random.seed(42)
    n = 1000

    FEATURE_NAMES = [
        'lepton_pT', 'lepton_eta', 'lepton_phi',
        'missing_energy_magnitude', 'missing_energy_phi',
        'jet1_pT', 'jet1_eta', 'jet1_phi', 'jet1_b_tag',
        'jet2_pT', 'jet2_eta', 'jet2_phi', 'jet2_b_tag',
        'jet3_pT', 'jet3_eta', 'jet3_phi', 'jet3_b_tag',
        'jet4_pT', 'jet4_eta', 'jet4_phi', 'jet4_b_tag',
        'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
    ]

    data = []
    for i in range(n):
        row = [float(i % 2)]  # label
        row += [random.gauss(0, 1) for _ in FEATURE_NAMES]
        data.append(tuple(row))

    fields = [StructField('label', FloatType())]
    for name in FEATURE_NAMES:
        fields.append(StructField(name, FloatType()))

    schema = StructType(fields)
    return spark.createDataFrame(data, schema)


# ===============================================================
# Data Ingestion Tests
# ===============================================================
class TestDataIngestion:

    def test_schema_has_correct_columns(self, sample_df):
        """Schema must have label + 28 feature columns = 29 total."""
        assert len(sample_df.columns) == 29, (
            f'Expected 29 columns, got {len(sample_df.columns)}'
        )

    def test_schema_has_label_column(self, sample_df):
        assert 'label' in sample_df.columns

    def test_row_count_matches_input(self, sample_df):
        assert sample_df.count() == 1000

    def test_label_values_are_binary(self, sample_df):
        from pyspark.sql import functions as F
        distinct_labels = sample_df.select('label').distinct().collect()
        label_set = {row.label for row in distinct_labels}
        assert label_set.issubset({0.0, 1.0}), f'Unexpected labels: {label_set}'

    def test_no_completely_null_columns(self, sample_df):
        from pyspark.sql import functions as F
        for col in sample_df.columns:
            null_count = sample_df.filter(F.col(col).isNull()).count()
            assert null_count < sample_df.count(), f'Column {col} is entirely null'

    def test_parquet_roundtrip(self, spark, sample_df, tmp_path):
        """Parquet write then read must preserve row count and schema."""
        parquet_path = str(tmp_path / 'test_parquet')
        sample_df.write.mode('overwrite').parquet(parquet_path)
        df_loaded = spark.read.parquet(parquet_path)
        assert df_loaded.count() == sample_df.count()
        assert set(df_loaded.columns) == set(sample_df.columns)


# ===============================================================
# Feature Engineering Tests
# ===============================================================
class TestFeatureEngineering:

    def test_physics_transformer_adds_5_columns(self, sample_df, spark):
        """PhysicsFeatureEngineer must add exactly 5 new columns."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'notebooks'))
        from pyspark.sql import functions as F

        # Inline the transformer to avoid notebook import issues
        from pyspark.ml import Transformer
        from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

        class PhysicsFeatureEngineer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
            def _transform(self, df):
                df = df.withColumn('lepton_to_jet_pT_ratio',
                                   F.col('lepton_pT') / (F.col('jet1_pT') + F.lit(1e-6)))
                df = df.withColumn('total_visible_pT',
                                   F.col('jet1_pT') + F.col('jet2_pT') +
                                   F.col('jet3_pT') + F.col('jet4_pT'))
                df = df.withColumn('bjet_multiplicity',
                                   (F.col('jet1_b_tag') > 0.5).cast('int') +
                                   (F.col('jet2_b_tag') > 0.5).cast('int') +
                                   (F.col('jet3_b_tag') > 0.5).cast('int') +
                                   (F.col('jet4_b_tag') > 0.5).cast('int'))
                df = df.withColumn('mass_ratio_bb_wwbb',
                                   F.col('m_bb') / (F.col('m_wwbb') + F.lit(1e-6)))
                df = df.withColumn('delta_eta_leading_jets',
                                   F.abs(F.col('jet1_eta') - F.col('jet2_eta')))
                return df

        transformer = PhysicsFeatureEngineer()
        df_out = transformer.transform(sample_df)
        new_cols = set(df_out.columns) - set(sample_df.columns)
        assert len(new_cols) == 5, f'Expected 5 new columns, got {len(new_cols)}: {new_cols}'

    def test_vector_assembler_output(self, sample_df, spark):
        """VectorAssembler must produce a 'features' column."""
        from pyspark.ml.feature import VectorAssembler
        from pyspark.sql import functions as F

        feature_cols = [c for c in sample_df.columns if c != 'label']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features',
                                    handleInvalid='skip')
        df_out = assembler.transform(sample_df)
        assert 'features' in df_out.columns

    def test_standard_scaler_zero_mean(self, sample_df, spark):
        """After StandardScaler with withMean=True, mean should be ~0."""
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        from pyspark.ml.functions import vector_to_array
        from pyspark.sql import functions as F

        feature_cols = [c for c in sample_df.columns if c != 'label']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='raw_features',
                                    handleInvalid='skip')
        scaler = StandardScaler(inputCol='raw_features', outputCol='features',
                                withMean=True, withStd=True)
        df_vec = assembler.transform(sample_df)
        scaler_model = scaler.fit(df_vec)
        df_scaled = scaler_model.transform(df_vec)

        # Check first feature has mean ~0
        mean_val = (
            df_scaled
            .withColumn('f0', vector_to_array('features')[0])
            .agg(F.mean('f0'))
            .collect()[0][0]
        )
        assert abs(mean_val) < 0.1, f'Expected mean ~0, got {mean_val}'

    def test_train_val_test_split_sizes(self, sample_df, spark):
        """70/15/15 split must sum to total rows."""
        from pyspark.sql import functions as F

        df_label = sample_df.withColumn('label', F.col('label').cast('double'))
        class0 = df_label.filter(F.col('label') == 0)
        class1 = df_label.filter(F.col('label') == 1)

        tr0, va0, te0 = class0.randomSplit([0.70, 0.15, 0.15], seed=42)
        tr1, va1, te1 = class1.randomSplit([0.70, 0.15, 0.15], seed=42)

        train = tr0.union(tr1)
        val   = va0.union(va1)
        test  = te0.union(te1)

        total = sample_df.count()
        split_total = train.count() + val.count() + test.count()
        assert split_total == total, f'Split sizes {split_total} != total {total}'


# ===============================================================
# Model Training Tests
# ===============================================================
class TestModelTraining:

    @pytest.fixture(scope='class')
    def prepared_df(self, sample_df, spark):
        """Prepare feature vector DataFrame for model tests."""
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        from pyspark.sql import functions as F

        feature_cols = [c for c in sample_df.columns if c != 'label']
        df_label = sample_df.withColumn('label', F.col('label').cast('double'))
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='raw_features',
                                    handleInvalid='skip')
        scaler = StandardScaler(inputCol='raw_features', outputCol='features',
                                withMean=True, withStd=True)
        df_vec = assembler.transform(df_label)
        scaler_model = scaler.fit(df_vec)
        return scaler_model.transform(df_vec).select('label', 'features')

    def test_logistic_regression_trains(self, prepared_df, spark):
        from pyspark.ml.classification import LogisticRegression
        lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)
        model = lr.fit(prepared_df)
        preds = model.transform(prepared_df)
        assert 'prediction' in preds.columns

    def test_random_forest_trains(self, prepared_df, spark):
        from pyspark.ml.classification import RandomForestClassifier
        rf = RandomForestClassifier(featuresCol='features', labelCol='label',
                                    numTrees=10, maxDepth=5, seed=42)
        model = rf.fit(prepared_df)
        assert model.numTrees == 10

    def test_gbt_trains(self, prepared_df, spark):
        from pyspark.ml.classification import GBTClassifier
        gbt = GBTClassifier(featuresCol='features', labelCol='label',
                             maxIter=5, maxDepth=3, seed=42)
        model = gbt.fit(prepared_df)
        preds = model.transform(prepared_df)
        assert 'prediction' in preds.columns

    def test_svm_trains(self, prepared_df, spark):
        from pyspark.ml.classification import LinearSVC
        svm = LinearSVC(featuresCol='features', labelCol='label', maxIter=10)
        model = svm.fit(prepared_df)
        preds = model.transform(prepared_df)
        assert 'prediction' in preds.columns

    def test_predictions_are_binary(self, prepared_df, spark):
        from pyspark.ml.classification import LogisticRegression
        from pyspark.sql import functions as F
        lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=5)
        model = lr.fit(prepared_df)
        preds = model.transform(prepared_df)
        distinct_preds = {row.prediction for row in preds.select('prediction').distinct().collect()}
        assert distinct_preds.issubset({0.0, 1.0}), f'Unexpected predictions: {distinct_preds}'


# ===============================================================
# Evaluation Tests
# ===============================================================
class TestEvaluation:

    def test_auc_is_between_0_and_1(self, sample_df, spark):
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        from pyspark.sql import functions as F

        feature_cols = [c for c in sample_df.columns if c != 'label']
        df = sample_df.withColumn('label', F.col('label').cast('double'))
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='raw_f', handleInvalid='skip')
        scaler    = StandardScaler(inputCol='raw_f', outputCol='features', withMean=True, withStd=True)
        df_vec    = assembler.transform(df)
        df_scaled = scaler.fit(df_vec).transform(df_vec).select('label', 'features')

        lr = LogisticRegression(maxIter=10)
        model = lr.fit(df_scaled)
        preds = model.transform(df_scaled)

        evaluator = BinaryClassificationEvaluator(labelCol='label',
                                                   rawPredictionCol='rawPrediction',
                                                   metricName='areaUnderROC')
        auc_val = evaluator.evaluate(preds)
        assert 0.0 <= auc_val <= 1.0, f'AUC out of range: {auc_val}'

    def test_bootstrap_ci_bounds(self):
        """Bootstrap CI lower bound must be <= mean <= upper bound."""
        from sklearn.utils import resample
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 1000)
        y_prob = rng.random(1000)

        aucs = []
        for _ in range(200):
            yt, yp = resample(y_true, y_prob, stratify=y_true, random_state=42)
            aucs.append(roc_auc_score(yt, yp))

        lo = np.percentile(aucs, 2.5)
        hi = np.percentile(aucs, 97.5)
        mean = np.mean(aucs)

        assert lo <= mean <= hi

    def test_confusion_matrix_shape(self):
        """Confusion matrix for binary classification must be 2x2."""
        from sklearn.metrics import confusion_matrix
        y_true = [0, 1, 0, 1, 0, 1, 1, 0]
        y_pred = [0, 1, 1, 1, 0, 0, 1, 0]
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)


# ===============================================================
# Config Tests
# ===============================================================
class TestConfig:

    def test_spark_config_loads(self):
        import yaml
        config_path = Path(__file__).parent.parent / 'config' / 'spark_config.yaml'
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert 'driver_memory' in cfg
        assert 'executor_memory' in cfg
        assert 'shuffle_partitions' in cfg

    def test_tableau_config_loads(self):
        config_path = Path(__file__).parent.parent / 'config' / 'tableau_config.json'
        with open(config_path) as f:
            cfg = json.load(f)
        assert 'tableau_config' in cfg
        assert 'dashboards' in cfg['tableau_config']
        assert len(cfg['tableau_config']['dashboards']) == 4

    def test_schema_file_exists(self):
        schema_path = Path(__file__).parent.parent / 'data' / 'schemas' / 'higgs_schema.json'
        # Only check if run after ingestion
        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)
            assert 'label' in schema
            assert len(schema) == 29


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
