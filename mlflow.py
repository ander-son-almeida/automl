import mlflow
import mlflow.spark

# Função principal
def main(df, features_cols, variavel_resposta, metricas_disponiveis, metrica_vencedora, seed, save_dir):
    spark = SparkSession.builder.appName("AutoML").getOrCreate()
    assembler = VectorAssembler(inputCols=features_cols, outputCol="features")
    df = assembler.transform(df)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)
    
    # Definir todos os modelos
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
        ("OneVsRest", OneVsRest(classifier=LogisticRegression(featuresCol='features', labelCol=variavel_resposta)))
    ]
    
    # Avaliar todos os modelos
    results = []
    for model_name, model in models:
        with mlflow.start_run():
            print(f"Avaliando {model_name}...")
            metrics, predictions = evaluate_model(model, train_df, test_df, variavel_resposta)
            results.append((model_name, metrics, mlflow.active_run().info.run_id))
            print(f"{model_name} - Métricas: {metrics}")
            
            # Log das métricas no MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log do modelo no MLflow
            mlflow.spark.log_model(model, model_name)
    
    # Determinar o modelo vencedor com base na métrica escolhida
    sorted_results = sorted(results, key=lambda x: x[1][metrica_vencedora], reverse=True)
    winning_model_name, winning_metrics, winning_run_id = sorted_results[0]
    print(f"Modelo vencedor: {winning_model_name} com {metrica_vencedora}: {winning_metrics[metrica_vencedora]}")
    
    # Otimizar o modelo vencedor
    print(f"Otimizando {winning_model_name} com Optuna...")
    best_params = optimize_model(winning_model_name, train_df, test_df, variavel_resposta, seed)
    print(f"Melhores hiperparâmetros para {winning_model_name}: {best_params}")
    
    # Treinar o modelo vencedor com os melhores hiperparâmetros
    if winning_model_name == "LogisticRegression":
        model = LogisticRegression(featuresCol='features', labelCol=variavel_resposta, **best_params)
    elif winning_model_name == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed, **best_params)
    elif winning_model_name == "RandomForestClassifier":
        model = RandomForestClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed, **best_params)
    elif winning_model_name == "GBTClassifier":
        model = GBTClassifier(featuresCol='features', labelCol=variavel_resposta, seed=seed, **best_params)
    elif winning_model_name == "LinearSVC":
        model = LinearSVC(featuresCol='features', labelCol=variavel_resposta, **best_params)
    elif winning_model_name == "NaiveBayes":
        model = NaiveBayes(featuresCol='features', labelCol=variavel_resposta, **best_params)
    elif winning_model_name == "MultilayerPerceptronClassifier":
        model = MultilayerPerceptronClassifier(featuresCol='features', labelCol=variavel_resposta, 
                                               layers=[len(features_cols), 5, 2], seed=seed, **best_params)
    elif winning_model_name == "FMClassifier":
        model = FMClassifier(featuresCol='features', labelCol=variavel_resposta, **best_params)
    elif winning_model_name == "LightGBMClassifier":
        model = LightGBMClassifier(featuresCol='features', labelCol=variavel_resposta, predictionCol="prediction", seed=seed, **best_params)
    elif winning_model_name == "OneVsRest":
        model = OneVsRest(classifier=LogisticRegression(featuresCol='features', labelCol=variavel_resposta, **best_params))
    
    # Avaliar o modelo otimizado
    with mlflow.start_run():
        print(f"Avaliando {winning_model_name} otimizado...")
        optimized_metrics, optimized_predictions = evaluate_model(model, train_df, test_df, variavel_resposta)
        print(f"{winning_model_name} (Otimizado) - Métricas: {optimized_metrics}")
        
        # Log das métricas no MLflow
        for metric_name, metric_value in optimized_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log do modelo otimizado no MLflow
        mlflow.spark.log_model(model, f"{winning_model_name}_optimized")
        
        # Salvar o modelo e os hiperparâmetros
        save_model_and_params(model, best_params, winning_model_name, save_dir)
        
        # Adicionar o modelo otimizado aos resultados
        optimized_results = {
            "Modelo": f"{winning_model_name} (Otimizado)",
            **optimized_metrics,
            "Hiperparâmetros": best_params,
            "Run ID": mlflow.active_run().info.run_id
        }
        results.append(("Modelo Otimizado", optimized_results))
    
    # Converter resultados para DataFrame
    result_df = pd.DataFrame([{
        "Modelo": model_name,
        **metrics,
        "Run ID": run_id
    } for model_name, metrics, run_id in results])
    
    # Exibir o DataFrame de resultados
    print("Resultados dos modelos:")
    print(result_df)
    
    # Plotar gráficos apenas para o modelo vencedor otimizado
    if "ConfusionMatrix" in optimized_metrics:
        plot_confusion_matrix(optimized_metrics["ConfusionMatrix"])
    if "AUC" in optimized_metrics:
        plot_roc_curve(optimized_predictions, variavel_resposta)
        plot_pr_curve(optimized_predictions, variavel_resposta)
        plot_probability_distribution(optimized_predictions, variavel_resposta)
        plot_calibration_curve(optimized_predictions, variavel_resposta)
        plot_ks_curve(optimized_predictions, variavel_resposta)
        plot_lift_curve(optimized_predictions, variavel_resposta)
    if winning_model_name in ["RandomForestClassifier", "GBTClassifier"]:
        plot_feature_importance(model, features_cols)
    
    return result_df

# Exemplo de uso
if __name__ == "__main__":
    data = [(1.0, 2.0, 3.0, 0), (4.0, 5.0, 6.0, 1), (7.0, 8.0, 9.0, 0), (10.0, 11.0, 12.0, 1)]
    columns = ["feature1", "feature2", "feature3", "target"]
    spark = SparkSession.builder.appName("Example").getOrCreate()
    df = spark.createDataFrame(data, columns)
    
    features_cols = ["feature1", "feature2", "feature3"]
    variavel_resposta = "target"
    metricas_disponiveis = ["accuracy", "weightedPrecision", "weightedRecall", "f1", "AUC", "KS"]
    metrica_vencedora = "accuracy"
    seed = 42
    save_dir = "/dbfs/mnt/your_directory"
    
    result_df = main(df, features_cols, variavel_resposta, metricas_disponiveis, metrica_vencedora, seed, save_dir)


import mlflow.spark

# Suponha que você tenha o run_id do modelo vencedor
run_id = result_df[result_df["Modelo"] == "Modelo Otimizado"]["Run ID"].values[0]

# Carregar o modelo
model = mlflow.spark.load_model(f"runs:/{run_id}/{winning_model_name}_optimized")



%pip install h2o pysparkling-water

from pysparkling import *
import h2o
from h2o.automl import H2OAutoML

# Inicializar H2O
hc = H2OContext.getOrCreate(spark)

# Carregar dados
df_spark = spark.table("tabela_classificacao")
h2o_frame = hc.asH2OFrame(df_spark)

# Configurar AutoML
aml = H2OAutoML(max_models=10, seed=42, max_runtime_secs=300, sort_metric="AUC")
aml.train(x=features, y="target", training_frame=h2o_frame)

# Melhor modelo
best_model = aml.leader
print(best_model)

# Salvar modelo
h2o.save_model(best_model, path="/dbfs/meus_modelos/h2o_automl_model", force=True)