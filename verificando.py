from pyspark.sql.functions import col, isnan, when, count

# Função para verificar se há NaN nas colunas de features
def has_nan_features(df, feature_columns):
    # Conta os NaN em cada coluna de features
    nan_counts = df.select(
        [count(when(isnan(col(c)) | col(c).isNull(), c)).alias(c) for c in feature_columns]
    ).collect()[0]
    
    # Verifica se alguma coluna tem NaN
    return any(nan_counts[c] > 0 for c in feature_columns)

# Verifica se há NaN nas features
if has_nan_features(train_df, features_cols):
    print("⚠️ Dataset contém valores NaN/nulos. Aplicando apenas LightGBM e XGBoost (suportam missing values).")
    models = [
        ("LightGBMClassifier", LightGBMClassifier(featuresCol='features', labelCol=variavel_resposta, predictionCol="prediction", seed=seed)),
        ("XGBoostClassifier", XGBoostClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed))
    ]
else:
    print("✅ Dataset sem valores NaN/nulos. Aplicando todos os modelos.")
    models = [
        ("LogisticRegression", LogisticRegression(featuresCol='features', labelCol=variavel_resposta)),
        ("DecisionTreeClassifier", DecisionTreeClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
        ("RandomForestClassifier", RandomForestClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
        ("GBTClassifier", GBTClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
        ("LinearSVC", LinearSVC(featuresCol='features', labelCol=variavel_resposta)),
        ("NaiveBayes", NaiveBayes(featuresCol='features', labelCol=variavel_resposta)),
        ("MultilayerPerceptronClassifier", MultilayerPerceptronClassifier(featuresCol='features', 
                                                                          labelCol=variavel_resposta, 
                                                                          layers=[len(features_cols), 5, 2], 
                                                                          seed=seed)),
        ("FMClassifier", FMClassifier(featuresCol='features', labelCol=variavel_resposta)),
        ("LightGBMClassifier", LightGBMClassifier(featuresCol='features', labelCol=variavel_resposta, predictionCol="prediction", seed=seed)),
        ("XGBoostClassifier", XGBoostClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
        ("OneVsRest", OneVsRest(classifier=LogisticRegression(featuresCol='features', labelCol=variavel_resposta)))
    ]
