from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
    GBTClassifier, LinearSVC, NaiveBayes, MultilayerPerceptronClassifier,
    FMClassifier, OneVsRest
)
from sparkdl.xgboost import XGBoostClassifier
from synapse.ml.lightgbm import LightGBMClassifier

# 1. Tratar missing values (preencher com zero)
dataset = dataset.fillna(0)

# 2. Criar pipeline de preparação
assembler = VectorAssembler(inputCols=features_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", 
                       withStd=True, withMean=True)

prep_pipeline = Pipeline(stages=[assembler, scaler])
prep_model = prep_pipeline.fit(dataset)

# 3. Criar versões normalizada e original do dataset
dataset_normalized = prep_model.transform(dataset)
dataset_original = dataset_normalized.withColumn("features", dataset_normalized["features_raw"])

# 4. Lista de modelos (nomes originais)
models = [
    ("LogisticRegression", LogisticRegression(featuresCol='features', labelCol=variavel_resposta)),
    ("DecisionTreeClassifier", DecisionTreeClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
    ("RandomForestClassifier", RandomForestClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
    ("GBTClassifier", GBTClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
    ("LinearSVC", LinearSVC(featuresCol='features', labelCol=variavel_resposta)),
    ("NaiveBayes", NaiveBayes(featuresCol='features', labelCol=variavel_resposta)),
    ("MultilayerPerceptronClassifier", MultilayerPerceptronClassifier(
        featuresCol='features', labelCol=variavel_resposta, layers=[len(features_cols), 5, 2], seed=seed)),
    ("FMClassifier", FMClassifier(featuresCol='features', labelCol=variavel_resposta)),
    ("LightGBMClassifier", LightGBMClassifier(
        featuresCol='features', labelCol=variavel_resposta, predictionCol="prediction", seed=seed)),
    ("XGBoostClassifier", XGBoostClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
    ("OneVsRest", OneVsRest(classifier=LogisticRegression(featuresCol='features', labelCol=variavel_resposta)))
]

# 5. Lista de modelos que DEVEM usar dados normalizados
models_using_normalized = [
    "LogisticRegression",
    "LinearSVC",
    "MultilayerPerceptronClassifier",
    "FMClassifier",
    "OneVsRest"
]

# 6. Loop de treinamento
for model_name, model in models:
    try:
        # Decide qual dataset usar
        current_df = dataset_normalized if model_name in models_using_normalized else dataset_original
        
        # Treina o modelo
        fitted_model = model.fit(current_df)
        predictions = fitted_model.transform(current_df)
        
        # Avaliação do modelo aqui...
        print(f"Modelo {model_name} treinado com sucesso")
        
    except Exception as e:
        print(f"Erro ao treinar {model_name}: {str(e)}")
