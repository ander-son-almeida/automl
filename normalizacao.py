from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import *
from pyspark.sql.functions import col
from synapse.ml.lightgbm import LightGBMClassifier
from sparkdl.xgboost import XGBoostClassifier

# 1. Configurações iniciais
seed = 42
variavel_resposta = "target"
features_cols = ["col1", "col2", "col3"]  # Substitua pelas suas colunas de features

# 2. Carregar dados
df_dev = spark.table("dados_desenvolvimento")  # Seu DataFrame de desenvolvimento
df_oot = spark.table("dados_oot")             # Seu DataFrame OOT (Out-of-Time)

# 3. Tratar missing values (preencher com zero)
df_dev = df_dev.fillna(0)
df_oot = df_oot.fillna(0)

# 4. Pipeline de preparação de dados
assembler = VectorAssembler(inputCols=features_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
prep_pipeline = Pipeline(stages=[assembler, scaler])

# 5. Treinar pipeline apenas nos dados de desenvolvimento
prep_model = prep_pipeline.fit(df_dev)

# 6. Aplicar transformações em todos os datasets
df_dev_prep = prep_model.transform(df_dev)
df_oot_prep = prep_model.transform(df_oot)

# 7. Criar versões originais (não normalizadas)
df_dev_original = df_dev_prep.withColumn("features", col("features_raw"))
df_oot_original = df_oot_prep.withColumn("features", col("features_raw"))

# 8. Split para treino/teste (apenas dados de desenvolvimento)
train_data, test_data = df_dev_prep.randomSplit([0.8, 0.2], seed=seed)

# 9. Lista de modelos
models = [
    ("LogisticRegression", LogisticRegression(featuresCol='features', labelCol=variavel_resposta)),
    ("DecisionTree", DecisionTreeClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
    ("RandomForest", RandomForestClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
    ("GBT", GBTClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
    ("LinearSVC", LinearSVC(featuresCol='features', labelCol=variavel_resposta)),
    ("MultilayerPerceptron", MultilayerPerceptronClassifier(
        featuresCol='features', labelCol=variavel_resposta, 
        layers=[len(features_cols), 5, 2], seed=seed)),
    ("FMClassifier", FMClassifier(featuresCol='features', labelCol=variavel_resposta)),
    ("LightGBM", LightGBMClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
    ("XGBoost", XGBoostClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed))
]

# 10. Modelos que usam dados normalizados
normalized_models = [
    "LogisticRegression",
    "LinearSVC",
    "MultilayerPerceptron",
    "FMClassifier"
]

# 11. Loop de treinamento e avaliação
for model_name, model in models:
    try:
        # Selecionar dataset apropriado
        if model_name in normalized_models:
            train_df = train_data.select("features", variavel_resposta)
            test_df = test_data.select("features", variavel_resposta)
            oot_df = df_oot_prep.select("features", variavel_resposta)
        else:
            train_df = df_dev_original.select("features", variavel_resposta)
            test_df = df_oot_original.select("features", variavel_resposta)
            oot_df = df_oot_original.select("features", variavel_resposta)
        
        # Treinar modelo
        trained_model = model.fit(train_df)
        
        # Avaliar nos dados de teste
        test_predictions = trained_model.transform(test_df)
        print(f"\nAvaliação {model_name} - Teste:")
        # Adicione suas métricas de avaliação aqui
        
        # Avaliar nos dados OOT
        oot_predictions = trained_model.transform(oot_df)
        print(f"\nAvaliação {model_name} - OOT:")
        # Adicione suas métricas de avaliação aqui
        
    except Exception as e:
        print(f"\nErro no modelo {model_name}: {str(e)}")

print("\nProcesso de modelagem concluído!")
