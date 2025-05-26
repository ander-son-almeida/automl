from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
    GBTClassifier, LinearSVC, NaiveBayes, MultilayerPerceptronClassifier,
    FMClassifier, OneVsRest
)
from sparkdl.xgboost import XGBoostClassifier
from synapse.ml.lightgbm import LightGBMClassifier

# Primeiro, tratar valores missing (preencher com zero)
dataset = dataset.fillna(0)

# Criar o vetor de features (já com missing tratados)
assembler = VectorAssembler(inputCols=features_cols, outputCol="features_raw")

# Criar o scaler para normalização
scaler = StandardScaler(inputCol="features_raw", outputCol="features", 
                        withStd=True, withMean=True)

# Pipeline para preparação dos dados
prep_pipeline = Pipeline(stages=[assembler, scaler])
prep_model = prep_pipeline.fit(dataset)
dataset_normalized = prep_model.transform(dataset)

# Para modelos que não precisam de normalização, usar o dataset original
dataset_original = prep_pipeline.fit(dataset).transform(dataset)
dataset_original = dataset_original.withColumn("features", dataset_original["features_raw"])

models = [
    # Modelos que usam dados normalizados
    ("LogisticRegression", LogisticRegression(featuresCol='features', labelCol=variavel_resposta)),
    ("LinearSVC", LinearSVC(featuresCol='features', labelCol=variavel_resposta)),
    ("MultilayerPerceptronClassifier", MultilayerPerceptronClassifier(
        featuresCol='features', 
        labelCol=variavel_resposta, 
        layers=[len(features_cols), 5, 2], 
        seed=seed)),
    ("FMClassifier", FMClassifier(featuresCol='features', labelCol=variavel_resposta)),
    ("OneVsRest", OneVsRest(classifier=LogisticRegression(featuresCol='features', labelCol=variavel_resposta))),
    
    # Modelos que usam dados originais (não normalizados)
    ("DecisionTreeClassifier_original", DecisionTreeClassifier(
        featuresCol='features', labelCol=variavel_resposta, seed=seed)),
    ("RandomForestClassifier_original", RandomForestClassifier(
        featuresCol='features', labelCol=variavel_resposta, seed=seed)),
    ("GBTClassifier_original", GBTClassifier(
        featuresCol='features', labelCol=variavel_resposta, seed=seed)),
    ("NaiveBayes_original", NaiveBayes(
        featuresCol='features', labelCol=variavel_resposta)),
    ("LightGBMClassifier_original", LightGBMClassifier(
        featuresCol='features', labelCol=variavel_resposta, predictionCol="prediction", seed=seed)),
    ("XGBoostClassifier_original", XGBoostClassifier(
        featuresCol='features', labelCol=variavel_resposta, seed=seed))
]

# Agora no loop de avaliação você precisará usar o dataset apropriado para cada modelo
for model_name, model in models:
    try:
        if "_original" in model_name:
            # Usar dataset original para modelos que não precisam de normalização
            fitted_model = model.fit(dataset_original)
            predictions = fitted_model.transform(dataset_original)
        else:
            # Usar dataset normalizado para outros modelos
            fitted_model = model.fit(dataset_normalized)
            predictions = fitted_model.transform(dataset_normalized)
        
        # Avaliação do modelo aqui...
        
    except Exception as e:
        print(f"Erro ao treinar {model_name}: {str(e)}")
