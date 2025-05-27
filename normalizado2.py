from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import *
from pyspark.sql.functions import col

# 1. Configurações
seed = 42
target_col = "target"
features_cols = ["col1", "col2", "col3"]  # Suas features

# 2. Carregar dados
df_dev = spark.table("dados_desenvolvimento").fillna(0)
df_oot = spark.table("dados_oot").fillna(0)

# 3. Pipeline de transformação (SEM criar coluna prediction)
assembler = VectorAssembler(inputCols=features_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled")
pipeline = Pipeline(stages=[assembler, scaler]).fit(df_dev)

# 4. Preparar dados
def prepare_data(df):
    df_transformed = pipeline.transform(df)
    df_normalized = df_transformed.withColumn("features", col("features_scaled")).drop("features_scaled")
    df_original = df_transformed.withColumn("features", col("features_raw")).drop("features_raw")
    return df_normalized, df_original

train_norm, train_orig = prepare_data(df_dev)
oot_norm, oot_orig = prepare_data(df_oot)

# 5. Split para treino/teste
train_norm, test_norm = train_norm.randomSplit([0.8, 0.2], seed=seed)
train_orig, test_orig = train_orig.randomSplit([0.8, 0.2], seed=seed)

# 6. Lista de modelos (todos usarão predictionCol="prediction")
models = [
    ("LogisticRegression", LogisticRegression(featuresCol="features", labelCol=target_col)),
    ("DecisionTreeClassifier", DecisionTreeClassifier(featuresCol="features", labelCol=target_col, seed=seed)),
    ("RandomForestClassifier", RandomForestClassifier(featuresCol="features", labelCol=target_col, seed=seed)),
    ("GBTClassifier", GBTClassifier(featuresCol="features", labelCol=target_col, seed=seed)),
    ("LinearSVC", LinearSVC(featuresCol="features", labelCol=target_col)),
    ("MultilayerPerceptronClassifier", MultilayerPerceptronClassifier(
        featuresCol="features", labelCol=target_col, layers=[len(features_cols), 5, 2], seed=seed)),
    ("FMClassifier", FMClassifier(featuresCol="features", labelCol=target_col)),
    ("LightGBMClassifier", LightGBMClassifier(featuresCol="features", labelCol=target_col, seed=seed)),
    ("XGBoostClassifier", XGBoostClassifier(featuresCol="features", labelCol=target_col, seed=seed))
]

# 7. Mapeamento de modelos para datasets
model_data_mapping = {
    "LogisticRegression": "normalized",
    "DecisionTreeClassifier": "original",
    "RandomForestClassifier": "original",
    "GBTClassifier": "original",
    "LinearSVC": "normalized",
    "MultilayerPerceptronClassifier": "normalized",
    "FMClassifier": "normalized",
    "LightGBMClassifier": "original",
    "XGBoostClassifier": "original"
}

# 8. Treinamento e avaliação
results = []

for model_name, model in models:
    try:
        # Selecionar dataset apropriado
        data_type = model_data_mapping[model_name]
        train_df = train_norm if data_type == "normalized" else train_orig
        test_df = test_norm if data_type == "normalized" else test_orig
        oot_df = oot_norm if data_type == "normalized" else oot_orig
        
        # Treinar modelo
        fitted_model = model.fit(train_df.select("features", target_col))
        
        # Avaliar no OOT (mantendo apenas prediction)
        oot_pred = fitted_model.transform(oot_df.select("features", target_col))
        
        # Calcular métricas (exemplo com AUC)
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        evaluator = BinaryClassificationEvaluator(labelCol=target_col)
        auc = evaluator.evaluate(oot_pred)
        
        results.append({
            "model_name": model_name,
            "model": fitted_model,
            "auc": auc,
            "predictions": oot_pred.select(target_col, "prediction")
        })
        
        print(f"{model_name} - AUC OOT: {auc:.4f}")
        
    except Exception as e:
        print(f"Erro no {model_name}: {str(e)}")

# 9. Selecionar o melhor modelo
best_model = max(results, key=lambda x: x["auc"])
print(f"\nMelhor modelo: {best_model['model_name']} (AUC: {best_model['auc']:.4f})")

# 10. Salvar o melhor modelo se necessário
# best_model["model"].write().overwrite().save("melhor_modelo")
