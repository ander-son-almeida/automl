from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from databricks import automl
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np
import os
import shutil

# Inicializa a sessão Spark
spark = SparkSession.builder.appName("AutoML Binary Classification").getOrCreate()

# Exemplo de DataFrame
data = [
    (1, 2.0, 3.0, 0),
    (2, 4.0, 5.0, 1),
    (3, 6.0, 7.0, 0),
    (4, 8.0, 9.0, 1),
    (5, 10.0, 11.0, 0),
]

# Nomes das colunas
columns = ["feature1", "feature2", "target"]

# Cria o DataFrame
df = spark.createDataFrame(data, columns)

# Configura o diretório de saída personalizado
output_dir = "/dbfs/FileStore/automl_results"  # Defina o diretório de sua preferência
os.makedirs(output_dir, exist_ok=True)  # Cria o diretório se não existir

# Configura o AutoML para classificação binária
summary = automl.classify(
    dataset=df,
    target_col="target",
    primary_metric="accuracy",  # Métrica primária (pode ser alterada para "roc_auc", "f1", etc.)
    timeout_minutes=30,
    max_trials=10
)

# Função para calcular o KS
def calculate_ks(predictions):
    # Ordena as previsões pela probabilidade da classe positiva
    predictions = predictions.orderBy(col("probability").desc())
    
    # Calcula a taxa acumulada de verdadeiros positivos e falsos positivos
    predictions = predictions.withColumn("cumulative_tp", expr("sum(label) over (order by probability desc)"))
    predictions = predictions.withColumn("cumulative_fp", expr("sum(1 - label) over (order by probability desc)"))
    
    # Normaliza as taxas acumuladas
    total_tp = predictions.selectExpr("sum(label)").collect()[0][0]
    total_fp = predictions.selectExpr("sum(1 - label)").collect()[0][0]
    predictions = predictions.withColumn("cumulative_tp_rate", col("cumulative_tp") / total_tp)
    predictions = predictions.withColumn("cumulative_fp_rate", col("cumulative_fp") / total_fp)
    
    # Calcula o KS
    ks = predictions.withColumn("ks", col("cumulative_tp_rate") - col("cumulative_fp_rate")).agg({"ks": "max"}).collect()[0][0]
    
    return ks

# Lista para armazenar os resultados de cada modelo
results = []

# Avalia cada modelo treinado pelo AutoML
for trial in summary.trials:
    # Obtém o run_id do trial
    run_id = trial.run_id
    
    # Carrega o modelo treinado usando MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.spark.load_model(model_uri)
    
    # Obtém as previsões do modelo
    predictions = model.transform(df)
    
    # Converte as previsões para Pandas para facilitar o cálculo das métricas
    predictions_pd = predictions.select("target", "probability").toPandas()
    predictions_pd["probability_positive"] = predictions_pd["probability"].apply(lambda x: x[1])  # Probabilidade da classe positiva
    predictions_pd["predicted_label"] = predictions_pd["probability_positive"].apply(lambda x: 1 if x > 0.5 else 0)
    
    # Calcula métricas de avaliação
    evaluator = BinaryClassificationEvaluator(labelCol="target", rawPredictionCol="rawPrediction")
    roc_auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    pr_auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
    
    # Calcula o KS
    ks = calculate_ks(predictions)
    
    # Adiciona os resultados à lista
    results.append({
        "model_name": trial.model_name,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_score": f1_score,
        "ks": ks,
        "run_id": run_id,  # Salva o run_id para referência
        "predictions": predictions_pd  # Salva as previsões para plotar gráficos
    })

# Obtém o modelo vencedor
best_model_run_id = summary.best_trial.run_id
best_model_uri = f"runs:/{best_model_run_id}/model"
best_model = mlflow.spark.load_model(best_model_uri)

# Salva o modelo vencedor no MLflow
with mlflow.start_run():
    model_id = mlflow.spark.log_model(best_model, "best_model")
    print(f"Model ID do modelo vencedor: {model_id}")

# Atualiza o ID do modelo vencedor na tabela de resultados
for result in results:
    if result["run_id"] == best_model_run_id:
        result["model_id"] = model_id

# Cria um DataFrame com os resultados
results_df = pd.DataFrame(results)

# Exibe a tabela de resultados
print("Tabela de Resultados:")
print(results_df)

# Salva a tabela de resultados em um arquivo CSV no diretório de saída
results_csv_path = os.path.join(output_dir, "automl_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Tabela de resultados salva em: {results_csv_path}")

# Carrega o modelo vencedor posteriormente (exemplo)
loaded_model = mlflow.spark.load_model(f"runs:/{model_id}/best_model")
print("Modelo vencedor carregado com sucesso!")

# Função para gerar subplots dinâmicos
def plot_metrics(y_true, y_prob, model_name):
    # Lista de gráficos disponíveis
    plots = []
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plots.append({
        "type": "roc",
        "data": (fpr, tpr, roc_auc),
        "title": f"Curva ROC (AUC = {roc_auc:.2f})",
        "xlabel": "False Positive Rate",
        "ylabel": "True Positive Rate"
    })
    
    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plots.append({
        "type": "precision_recall",
        "data": (recall, precision, pr_auc),
        "title": f"Curva Precision-Recall (AUC = {pr_auc:.2f})",
        "xlabel": "Recall",
        "ylabel": "Precision"
    })
    
    # Distribuição de Probabilidades (KS)
    df_ks = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df_ks = df_ks.sort_values(by='y_prob', ascending=False)
    df_ks['cumulative_tp'] = df_ks['y_true'].cumsum() / df_ks['y_true'].sum()
    df_ks['cumulative_fp'] = (1 - df_ks['y_true']).cumsum() / (1 - df_ks['y_true']).sum()
    df_ks['ks'] = df_ks['cumulative_tp'] - df_ks['cumulative_fp']
    ks_value = df_ks['ks'].max()
    plots.append({
        "type": "ks",
        "data": (df_ks['y_prob'], df_ks['cumulative_tp'], df_ks['cumulative_fp'], ks_value),
        "title": f"Distribuição de Probabilidades (KS = {ks_value:.2f})",
        "xlabel": "Probabilidade da Classe Positiva",
        "ylabel": "Taxa Acumulada"
    })
    
    # Cria subplots dinâmicos
    n_plots = len(plots)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]  # Garante que axes seja uma lista mesmo com um único gráfico
    
    for i, plot in enumerate(plots):
        ax = axes[i]
        if plot["type"] == "roc":
            fpr, tpr, roc_auc = plot["data"]
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlabel(plot["xlabel"])
            ax.set_ylabel(plot["ylabel"])
            ax.set_title(plot["title"])
            ax.legend(loc="lower right")
        elif plot["type"] == "precision_recall":
            recall, precision, pr_auc = plot["data"]
            ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
            ax.set_xlabel(plot["xlabel"])
            ax.set_ylabel(plot["ylabel"])
            ax.set_title(plot["title"])
            ax.legend(loc="lower left")
        elif plot["type"] == "ks":
            y_prob, cumulative_tp, cumulative_fp, ks_value = plot["data"]
            ax.plot(y_prob, cumulative_tp, label='True Positive Rate', color='blue')
            ax.plot(y_prob, cumulative_fp, label='False Positive Rate', color='red')
            ax.fill_between(y_prob, cumulative_tp, cumulative_fp, color='gray', alpha=0.3, label=f'KS = {ks_value:.2f}')
            ax.set_xlabel(plot["xlabel"])
            ax.set_ylabel(plot["ylabel"])
            ax.set_title(plot["title"])
            ax.legend(loc="upper left")
    
    plt.tight_layout()
    plt.suptitle(f"Métricas do Modelo: {model_name}", y=1.05, fontsize=16)
    
    # Salva o gráfico no diretório de saída
    plot_path = os.path.join(output_dir, f"{model_name}_metrics.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Gráficos salvos em: {plot_path}")

# Plota os gráficos para cada modelo
for result in results:
    y_true = result["predictions"]["target"]
    y_prob = result["predictions"]["probability_positive"]
    model_name = result["model_name"]
    
    print(f"\nGráficos para o modelo: {model_name}")
    plot_metrics(y_true, y_prob, model_name)

# Move os notebooks gerados pelo AutoML para o diretório de saída
automl_output_dir = summary.experiment_dir  # Diretório padrão onde o AutoML salva os resultados
if os.path.exists(automl_output_dir):
    notebooks_dir = os.path.join(output_dir, "notebooks")
    os.makedirs(notebooks_dir, exist_ok=True)
    for item in os.listdir(automl_output_dir):
        item_path = os.path.join(automl_output_dir, item)
        if os.path.isdir(item_path) and item.endswith("_notebook"):
            shutil.move(item_path, notebooks_dir)
    print(f"Notebooks do AutoML movidos para: {notebooks_dir}")
else:
    print("Diretório de saída do AutoML não encontrado.")
