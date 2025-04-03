import os
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml import PipelineModel
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
    GBTClassifier, LinearSVC, NaiveBayes, MultilayerPerceptronClassifier,
    FMClassifier)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from synapse.ml.lightgbm import LightGBMClassifier
from pyspark.ml.classification import OneVsRest
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import optuna

def calcular_ks(df, variavel_resposta, coluna_prob="probability"):
    # Calcula totais de bons e maus
    totais = df.agg(
        F.sum(F.when(F.col(variavel_resposta) == 0, 1).otherwise(0)).alias("total_bons"),
        F.sum(F.when(F.col(variavel_resposta) == 1, 1).otherwise(0)).alias("total_maus")
    ).collect()[0]
          
    total_bons = totais["total_bons"]
    total_maus = totais["total_maus"]
    
    # Ordena por probabilidade decrescente
    janela = Window.orderBy(F.desc(coluna_prob))
    
    # Calcula as distribuições acumuladas
    df_ks = df.withColumn("num_linha", F.row_number().over(janela)) \
              .withColumn("acum_bons", F.sum(F.when(F.col(variavel_resposta) == 0, 1).otherwise(0)).over(janela) / total_bons) \
              .withColumn("acum_maus", F.sum(F.when(F.col(variavel_resposta) == 1, 1).otherwise(0)).over(janela) / total_maus) \
              .withColumn("ks", F.col("acum_maus") - F.col("acum_bons"))
              
    # Encontra o máximo KS
    linha_ks = df_ks.orderBy(F.desc("ks")).first()
    valor_ks = linha_ks["ks"]
    
    # Converte para pandas para o gráfico
    df_ks_pandas = df_ks.select(coluna_prob, "acum_bons", "acum_maus", "ks").orderBy(coluna_prob).toPandas()
    
    return valor_ks, df_ks_pandas

def cross_validate_model(model, df, df_oot, variavel_resposta, metricas_disponiveis, n_folds=5, seed=42):
    # Configurar avaliadores
    evaluators = []
    for metric in metricas_disponiveis:
        if metric in ["accuracy", "weightedPrecision", "weightedRecall", "f1"]:
            evaluator = MulticlassClassificationEvaluator(
                labelCol=variavel_resposta, 
                predictionCol="prediction",
                metricName=metric
            )
            evaluators.append((metric, evaluator))
        elif metric == "AUC":
            evaluator = BinaryClassificationEvaluator(
                labelCol=variavel_resposta,
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )
            evaluators.append((metric, evaluator))
    
    # Criar grade de parâmetros básica
    paramGrid = ParamGridBuilder().build()
    
    # Configurar CrossValidator
    cv = CrossValidator(
        estimator=model,
        estimatorParamMaps=paramGrid,
        evaluator=evaluators[0][1],  # Usa o primeiro avaliador como referência
        numFolds=n_folds,
        seed=seed,
        collectSubModels=False
    )
    
    # Executar validação cruzada
    cvModel = cv.fit(df)
    
    # 1. Calcular métricas médias de CV
    cv_metrics = {}
    for metric, evaluator in evaluators:
        if metric == "KS":
            ks_values = []
            for i, (train, test) in enumerate(cvModel.getEstimator().split(df)):
                fitted_model = model.fit(train)
                predictions = fitted_model.transform(test)
                ks, _ = calcular_ks(predictions, variavel_resposta)
                ks_values.append(ks)
            cv_metrics[f"CV_{metric}"] = np.mean(ks_values)
        else:
            cv_metrics[f"CV_{metric}"] = np.mean(cvModel.avgMetrics)
    
    # 2. Avaliar no conjunto OOT (se fornecido)
    oot_metrics = {}
    if df_oot is not None:
        oot_predictions = cvModel.bestModel.transform(df_oot)
        
        for metric, evaluator in evaluators:
            if metric in ["accuracy", "weightedPrecision", "weightedRecall", "f1"]:
                oot_metrics[f"OOT_{metric}"] = evaluator.evaluate(oot_predictions)
            elif metric == "AUC":
                oot_metrics[f"OOT_{metric}"] = evaluator.evaluate(oot_predictions)
            elif metric == "KS":
                ks, _ = calcular_ks(oot_predictions, variavel_resposta)
                oot_metrics[f"OOT_{metric}"] = ks
    
    # Combinar todas as métricas
    return {**cv_metrics, **oot_metrics}

def optimize_model_with_cv(model_name, df, variavel_resposta, metricas_disponiveis, metrica_vencedora, seed, n_trials=10, n_folds=3):
    def objective(trial):
        # Definir modelo com hiperparâmetros sugeridos pelo Optuna
        if model_name == "LogisticRegression":
            model = LogisticRegression(
                featuresCol='features',
                labelCol=variavel_resposta,
                regParam=trial.suggest_float("regParam", 0.01, 10.0, log=True),
                elasticNetParam=trial.suggest_float("elasticNetParam", 0.0, 1.0)
            )
        elif model_name == "DecisionTreeClassifier":
            model = DecisionTreeClassifier(
                featuresCol='features',
                labelCol=variavel_resposta,
                maxDepth=trial.suggest_int("maxDepth", 2, 10),
                minInstancesPerNode=trial.suggest_int("minInstancesPerNode", 1, 10),
                seed=seed
            )
        elif model_name == "RandomForestClassifier":
            model = RandomForestClassifier(
                featuresCol='features',
                labelCol=variavel_resposta,
                numTrees=trial.suggest_int("numTrees", 10, 100),
                maxDepth=trial.suggest_int("maxDepth", 2, 10),
                seed=seed
            )
        elif model_name == "GBTClassifier":
            model = GBTClassifier(
                featuresCol='features',
                labelCol=variavel_resposta,
                maxIter=trial.suggest_int("maxIter", 10, 100),
                maxDepth=trial.suggest_int("maxDepth", 2, 10),
                seed=seed
            )
        elif model_name == "LinearSVC":
            model = LinearSVC(
                featuresCol='features',
                labelCol=variavel_resposta,
                regParam=trial.suggest_float("regParam", 0.01, 10.0, log=True),
                maxIter=trial.suggest_int("maxIter", 10, 100)
            )
        elif model_name == "NaiveBayes":
            model = NaiveBayes(
                featuresCol='features',
                labelCol=variavel_resposta,
                smoothing=trial.suggest_float("smoothing", 0.0, 10.0)
            )
        elif model_name == "MultilayerPerceptronClassifier":
            model = MultilayerPerceptronClassifier(
                featuresCol='features',
                labelCol=variavel_resposta,
                layers=[len(features_cols), trial.suggest_int("hiddenLayerSize", 2, 10), 2],
                maxIter=trial.suggest_int("maxIter", 10, 100),
                seed=seed
            )
        elif model_name == "FMClassifier":
            model = FMClassifier(
                featuresCol='features',
                labelCol=variavel_resposta,
                factorSize=trial.suggest_int("factorSize", 2, 10),
                regParam=trial.suggest_float("regParam", 0.01, 10.0, log=True)
            )
        elif model_name == "LightGBMClassifier":
            model = LightGBMClassifier(
                featuresCol='features',
                labelCol=variavel_resposta,
                numLeaves=trial.suggest_int("numLeaves", 10, 100),
                maxDepth=trial.suggest_int("maxDepth", 2, 10),
                learningRate=trial.suggest_float("learningRate", 0.01, 0.3, log=True),
                seed=seed
            )
        elif model_name == "OneVsRest":
            base_model = LogisticRegression(
                featuresCol='features',
                labelCol=variavel_resposta,
                regParam=trial.suggest_float("regParam", 0.01, 10.0, log=True),
                elasticNetParam=trial.suggest_float("elasticNetParam", 0.0, 1.0)
            )
            model = OneVsRest(classifier=base_model)
        
        # Avaliar com cross-validation
        metrics = cross_validate_model(
            model, 
            df, 
            None,  # Não usar OOT durante otimização
            variavel_resposta, 
            metricas_disponiveis, 
            n_folds=n_folds, 
            seed=seed
        )
        
        return metrics[f"CV_{metrica_vencedora}"]
    
    # Configurar estudo Optuna
    study = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    
    # Executar otimização
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def save_model_and_params(model, params, model_name, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, f"{model_name}_model")
    model.save(model_path)
    params_path = os.path.join(save_dir, f"{model_name}_params.yaml")
    with open(params_path, 'w') as file:
        yaml.dump(params, file)
    print(f"Modelo e hiperparâmetros salvos em: {save_dir}")

def create_model_with_params(model_name, params, features_cols, variavel_resposta, seed):
    if model_name == "LogisticRegression":
        return LogisticRegression(featuresCol='features', labelCol=variavel_resposta, **params)
    elif model_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed, **params)
    elif model_name == "RandomForestClassifier":
        return RandomForestClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed, **params)
    elif model_name == "GBTClassifier":
        return GBTClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed, **params)
    elif model_name == "LinearSVC":
        return LinearSVC(featuresCol='features', labelCol=variavel_resposta, **params)
    elif model_name == "NaiveBayes":
        return NaiveBayes(featuresCol='features', labelCol=variavel_resposta, **params)
    elif model_name == "MultilayerPerceptronClassifier":
        return MultilayerPerceptronClassifier(
            featuresCol='features', 
            labelCol=variavel_resposta, 
            layers=[len(features_cols), 5, 2], 
            seed=seed, 
            **params
        )
    elif model_name == "FMClassifier":
        return FMClassifier(featuresCol='features', labelCol=variavel_resposta, **params)
    elif model_name == "LightGBMClassifier":
        return LightGBMClassifier(
            featuresCol='features', 
            labelCol=variavel_resposta, 
            predictionCol="prediction", 
            seed=seed, 
            **params
        )
    elif model_name == "OneVsRest":
        return OneVsRest(classifier=LogisticRegression(
            featuresCol='features', 
            labelCol=variavel_resposta, 
            **params
        ))

def calculate_all_metrics(predictions, variavel_resposta, metricas_disponiveis):
    metrics = {}
    
    # Métricas multiclasse/binárias padrão
    evaluator = MulticlassClassificationEvaluator(labelCol=variavel_resposta, predictionCol="prediction")
    for metric in ["accuracy", "weightedPrecision", "weightedRecall", "f1"]:
        if metric in metricas_disponiveis:
            metrics[metric] = evaluator.evaluate(predictions, {evaluator.metricName: metric})
    
    # Métricas específicas para classificação binária
    if len(predictions.select(variavel_resposta).distinct().collect()) == 2:
        if "AUC" in metricas_disponiveis:
            auc_evaluator = BinaryClassificationEvaluator(
                labelCol=variavel_resposta, 
                rawPredictionCol="rawPrediction", 
                metricName="areaUnderROC"
            )
            metrics["AUC"] = auc_evaluator.evaluate(predictions)
        
        if "KS" in metricas_disponiveis:
            ks_value, _ = calcular_ks(predictions, variavel_resposta)
            metrics["KS"] = ks_value
    
    # Matriz de confusão
    metrics["ConfusionMatrix"] = predictions.groupBy(variavel_resposta, "prediction").count().orderBy(variavel_resposta, "prediction").collect()
    
    return metrics

def main(df, df_oot, features_cols, variavel_resposta, metricas_disponiveis, metrica_vencedora, seed, save_dir):
    spark = SparkSession.builder.appName("AutoML").getOrCreate()
    
    # Preparar os dados
    assembler = VectorAssembler(inputCols=features_cols, outputCol="features")
    df = assembler.transform(df)
    df_oot = assembler.transform(df_oot)
    
    # Definir todos os modelos
    models = [
        ("LogisticRegression", LogisticRegression(featuresCol='features', labelCol=variavel_resposta)),
        ("DecisionTreeClassifier", DecisionTreeClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
        ("RandomForestClassifier", RandomForestClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
        ("GBTClassifier", GBTClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed)),
        ("LinearSVC", LinearSVC(featuresCol='features', labelCol=variavel_resposta)),
        ("NaiveBayes", NaiveBayes(featuresCol='features', labelCol=variavel_resposta)),
        ("MultilayerPerceptronClassifier", MultilayerPerceptronClassifier(
            featuresCol='features', 
            labelCol=variavel_resposta, 
            layers=[len(features_cols), 5, 2], 
            seed=seed)),
        ("FMClassifier", FMClassifier(featuresCol='features', labelCol=variavel_resposta)),
        ("LightGBMClassifier", LightGBMClassifier(
            featuresCol='features', 
            labelCol=variavel_resposta, 
            predictionCol="prediction", 
            seed=seed)),
        ("OneVsRest", OneVsRest(classifier=LogisticRegression(
            featuresCol='features', 
            labelCol=variavel_resposta)))
    ]
    
    # Avaliar todos os modelos com cross-validation + OOT
    results = []
    for model_name, model in models:
        print(f"\nAvaliando {model_name} com cross-validation e OOT...")
        metrics = cross_validate_model(
            model, 
            df, 
            df_oot,
            variavel_resposta, 
            metricas_disponiveis, 
            n_folds=5, 
            seed=seed
        )
        results.append((model_name, metrics))
        print(f"{model_name} - Métricas:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    # Determinar o modelo vencedor com base na métrica OOT escolhida
    oot_metric_key = f"OOT_{metrica_vencedora}"
    winning_model_name, winning_metrics = max(results, key=lambda x: x[1].get(oot_metric_key, -1))
    
    print(f"\nModelo vencedor: {winning_model_name} com {oot_metric_key}: {winning_metrics[oot_metric_key]:.4f}")
    
    # Otimizar o modelo vencedor com cross-validation
    print(f"\nOtimizando {winning_model_name} com Optuna e cross-validation...")
    best_params = optimize_model_with_cv(
        winning_model_name, 
        df,  # Usar apenas dados de treino para otimização
        variavel_resposta, 
        metricas_disponiveis, 
        metrica_vencedora, 
        seed,
        n_trials=10,
        n_folds=3
    )
    print(f"Melhores hiperparâmetros para {winning_model_name}: {best_params}")
    
    # Treinar o modelo final com todos os dados de treino
    final_model = create_model_with_params(
        winning_model_name, 
        best_params, 
        features_cols, 
        variavel_resposta, 
        seed
    )
    fitted_final_model = final_model.fit(df)
    
    # Avaliar no conjunto OOT
    final_oot_predictions = fitted_final_model.transform(df_oot)
    final_oot_metrics = calculate_all_metrics(final_oot_predictions, variavel_resposta, metricas_disponiveis)
    
    # Salvar modelo e resultados
    save_model_and_params(fitted_final_model, best_params, winning_model_name, save_dir)
    
    # Adicionar resultados
    optimized_results = {
        "Modelo": f"{winning_model_name} (Otimizado)",
        **final_oot_metrics,
        "Hiperparâmetros": best_params
    }
    results.append(("Modelo Otimizado", optimized_results))
    
    # Converter resultados para DataFrame
    result_df = pd.DataFrame([{
        "Modelo": model_name,
        **metrics
    } for model_name, metrics in results])
    
    # Exibir o DataFrame de resultados
    print("\nResultados completos:")
    print(result_df.to_string())
    
    return result_df

# Exemplo de uso
if __name__ == "__main__":
    # Inicializar Spark
    spark = SparkSession.builder.appName("AutoML").getOrCreate()
    
    # Criar dados de exemplo (substitua por seus dados reais)
    data_train = [(1.0, 2.0, 3.0, 0), (4.0, 5.0, 6.0, 1), (7.0, 8.0, 9.0, 0), (10.0, 11.0, 12.0, 1)]
    data_oot = [(2.0, 3.0, 4.0, 0), (5.0, 6.0, 7.0, 1)]
    columns = ["feature1", "feature2", "feature3", "target"]
    
    df_train = spark.createDataFrame(data_train, columns)
    df_oot = spark.createDataFrame(data_oot, columns)
    
    # Configurações
    features_cols = ["feature1", "feature2", "feature3"]
    variavel_resposta = "target"
    metricas_disponiveis = ["accuracy", "AUC", "KS"]
    metrica_vencedora = "KS"
    seed = 42
    save_dir = "/dbfs/mnt/your_directory"
    
    # Executar
    result_df = main(
        df=df_train,
        df_oot=df_oot,
        features_cols=features_cols,
        variavel_resposta=variavel_resposta,
        metricas_disponiveis=metricas_disponiveis,
        metrica_vencedora=metrica_vencedora,
        seed=seed,
        save_dir=save_dir
    )
