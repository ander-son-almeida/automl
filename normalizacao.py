from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
    LinearSVC,
    MultilayerPerceptronClassifier,
    FMClassifier
)
from synapse.ml.lightgbm import LightGBMClassifier
from sparkdl.xgboost import XGBoostClassifier
from pyspark.sql.functions import col

# 1. Configurações iniciais
seed = 42
target_col = "target"
features_cols = ["col1", "col2", "col3"]  # Substitua pelas suas colunas de features

# 2. Carregar e preparar dados
df_dev = spark.table("dados_desenvolvimento").fillna(0)
df_oot = spark.table("dados_oot").fillna(0)

# 3. Pipeline de transformação
assembler = VectorAssembler(inputCols=features_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled")
pipeline = Pipeline(stages=[assembler, scaler]).fit(df_dev)

# 4. Criar DataFrames transformados
def prepare_data(df):
    df_transformed = pipeline.transform(df)
    df_normalized = df_transformed.withColumnRenamed("features_scaled", "features")
    df_original = df_transformed.withColumn("features", col("features_raw"))
    return df_normalized, df_original

# Dados de desenvolvimento
train_norm, train_orig = prepare_data(df_dev)
test_norm, test_orig = prepare_data(df_dev)  # Nota: na prática faça o split corretamente

# Dados OOT
oot_norm, oot_orig = prepare_data(df_oot)

# 5. Lista completa de modelos
models = [
    ("LogisticRegression", LogisticRegression(featuresCol="features", labelCol=target_col)),
    ("DecisionTreeClassifier", DecisionTreeClassifier(featuresCol="features", labelCol=target_col, seed=seed)),
    ("RandomForestClassifier", RandomForestClassifier(featuresCol="features", labelCol=target_col, seed=seed)),
    ("GBTClassifier", GBTClassifier(featuresCol="features", labelCol=target_col, seed=seed)),
    ("LinearSVC", LinearSVC(featuresCol="features", labelCol=target_col)),
    ("MultilayerPerceptronClassifier", MultilayerPerceptronClassifier(
        featuresCol="features", 
        labelCol=target_col,
        layers=[len(features_cols), 5, 2],  # Ajuste conforme necessário
        seed=seed
    )),
    ("FMClassifier", FMClassifier(featuresCol="features", labelCol=target_col)),
    ("LightGBMClassifier", LightGBMClassifier(featuresCol="features", labelCol=target_col, seed=seed)),
    ("XGBoostClassifier", XGBoostClassifier(featuresCol="features", labelCol=target_col, seed=seed))
]

# 6. Mapeamento de modelos para datasets
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

# 7. Função de avaliação
def evaluate_model(model, train_df, test_df, oot_df):
    # Treinar
    fitted_model = model.fit(train_df.select("features", target_col))
    
    # Avaliar
    for data, data_type in [(test_df, "Teste"), (oot_df, "OOT")]:
        predictions = fitted_model.transform(data.select("features", target_col))
        print(f"\nAvaliação {model.__class__.__name__} - {data_type}")
        # Adicione suas métricas aqui
        # Exemplo:
        # evaluator = BinaryClassificationEvaluator(labelCol=target_col)
        # print(f"AUC: {evaluator.evaluate(predictions):.4f}")

# 8. Treinamento e avaliação
for model_name, model in models:
    try:
        data_type = model_data_mapping[model_name]
        
        if data_type == "normalized":
            train_df = train_norm
            test_df = test_norm
            oot_df = oot_norm
        else:
            train_df = train_orig
            test_df = test_orig
            oot_df = oot_orig
        
        print(f"\nTreinando {model_name} com dados {data_type}")
        evaluate_model(model, train_df, test_df, oot_df)
        
    except Exception as e:
        print(f"\nErro no modelo {model_name}: {str(e)}")