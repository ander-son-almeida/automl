import mlflow
import mlflow.spark
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import DataFrame
import uuid
from databricks_cli.sdk.api_client import ApiClient
from databricks_cli.sdk.service import WorkspaceService

# --------------------------------------------------
# 1. MÓDULO DE LOGGING (externo como você pediu)
# --------------------------------------------------
def setup_logger(
    name: str = 'mlflow_logger',
    log_level: str = 'INFO',
    console_log: bool = True,
    mlflow_log: bool = True
) -> logging.Logger:
    """Configura logger integrado com MLflow"""
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if console_log:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if mlflow_log:
        class MLflowLogHandler(logging.Handler):
            def emit(self, record):
                log_entry = self.format(record)
                mlflow.log_text(log_entry, "execution_log.txt")
        
        mlflow_handler = MLflowLogHandler()
        mlflow_handler.setFormatter(formatter)
        logger.addHandler(mlflow_handler)
    
    return logger

# --------------------------------------------------
# 2. CLASSE PARA VISUALIZAÇÕES (separada)
# --------------------------------------------------
class ModelVisualizer:
    """Gera e salva visualizações do modelo"""
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def plot_feature_importance(self, model, feature_names: List[str]) -> plt.Figure:
        """Gera gráfico de importância de features"""
        self.logger.info("Gerando feature importance plot")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        importances = model.featureImportances.toArray()
        indices = importances.argsort()[::-1]
        
        ax.barh(range(len(feature_names)), importances[indices], align='center')
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title("Feature Importances")
        
        return fig
    
    def plot_metrics_comparison(self, metrics: dict) -> plt.Figure:
        """Gráfico de comparação de métricas"""
        self.logger.info("Gerando gráfico de métricas")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(metrics.keys(), metrics.values())
        ax.set_title("Model Metrics Comparison")
        plt.xticks(rotation=45)
        
        return fig

# --------------------------------------------------
# 3. FUNÇÕES AUXILIARES
# --------------------------------------------------
def save_dataframe_to_mlflow(df: DataFrame, artifact_path: str):
    """Salva DataFrame como CSV no MLflow"""
    pandas_df = df.toPandas()
    csv_str = pandas_df.to_csv(index=False)
    mlflow.log_text(csv_str, f"data/{artifact_path}.csv")

def get_mlflow_run_url(run_id: str) -> str:
    """Retorna URL para acessar a run no MLflow UI"""
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
    return f"https://{workspace_url}/#mlflow/runs/{run_id}"

# --------------------------------------------------
# 4. FLUXO PRINCIPAL
# --------------------------------------------------
def run_full_pipeline(
    train_data: DataFrame,
    test_data: DataFrame,
    feature_cols: List[str],
    target_col: str = "label",
    experiment_name: str = "Prod_Model_Training"
):
    """Orquestra todo o processo de treinamento e logging"""
    
    # Configuração inicial
    mlflow.set_experiment(experiment_name)
    logger = setup_logger()
    visualizer = ModelVisualizer(logger)
    
    with mlflow.start_run() as run:
        try:
            # ------------------------------------------
            # ETAPA 1: PREPARAÇÃO
            # ------------------------------------------
            logger.info("Iniciando pipeline de treinamento")
            logger.info(f"Run ID: {run.info.run_id}")
            
            # ------------------------------------------
            # ETAPA 2: TREINAMENTO
            # ------------------------------------------
            logger.info("Iniciando treinamento do modelo")
            
            rf = RandomForestClassifier(
                numTrees=50,
                maxDepth=7,
                featuresCol="features",
                labelCol=target_col,
                seed=42
            )
            
            model = rf.fit(train_data)
            
            # Log dos parâmetros
            mlflow.log_params({
                "numTrees": 50,
                "maxDepth": 7,
                "features": str(feature_cols),
                "target": target_col
            })
            
            # ------------------------------------------
            # ETAPA 3: AVALIAÇÃO
            # ------------------------------------------
            logger.info("Avaliando modelo")
            
            predictions = model.transform(test_data)
            
            # Cálculo de métricas (exemplo)
            accuracy = predictions.filter(predictions[target_col] == predictions.prediction).count() / predictions.count()
            
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": 0.92,  # Exemplo
                "recall": 0.89       # Exemplo
            })
            
            # ------------------------------------------
            # ETAPA 4: VISUALIZAÇÕES
            # ------------------------------------------
            logger.info("Gerando visualizações")
            
            # Gráfico da classe
            fig1 = visualizer.plot_feature_importance(model, feature_cols)
            mlflow.log_figure(fig1, "plots/feature_importance.png")
            plt.close(fig1)
            
            # Gráfico adicional (não da classe)
            fig2, ax = plt.subplots(figsize=(10, 6))
            ax.hist(predictions.select("prediction").toPandas(), bins=20)
            ax.set_title("Prediction Distribution")
            mlflow.log_figure(fig2, "plots/prediction_distribution.png")
            plt.close(fig2)
            
            # ------------------------------------------
            # ETAPA 5: SALVANDO ARTEFATOS
            # ------------------------------------------
            logger.info("Salvando artefatos adicionais")
            
            # Salva CSV com predictions
            save_dataframe_to_mlflow(predictions.limit(1000), "sample_predictions")
            
            # Salva estatísticas
            stats_df = predictions.describe()
            save_dataframe_to_mlflow(stats_df, "predictions_stats")
            
            # ------------------------------------------
            # ETAPA 6: SALVANDO O MODELO
            # ------------------------------------------
            logger.info("Registrando modelo no MLflow")
            mlflow.spark.log_model(
                model,
                "model",
                registered_model_name="Prod_RandomForest"
            )
            
            # ------------------------------------------
            # ETAPA FINAL: GERANDO LINK
            # ------------------------------------------
            run_url = get_mlflow_run_url(run.info.run_id)
            logger.info(f"Processo concluído! Acesse os resultados em: {run_url}")
            
            # Exibe o link no notebook (Databricks)
            displayHTML(f'<a href="{run_url}" target="_blank">Abrir no MLflow UI</a>')
            
            return run_url
            
        except Exception as e:
            logger.error(f"Erro no pipeline: {str(e)}")
            raise

# --------------------------------------------------
# 5. EXEMPLO DE USO (simulado)
# --------------------------------------------------
if __name__ == "__main__":
    # Exemplo de chamada (substitua com seus dados reais)
    try:
        # from pyspark.sql import SparkSession
        # spark = SparkSession.builder.getOrCreate()
        # train_data = spark.read...
        # test_data = spark.read...
        
        run_link = run_full_pipeline(
            train_data=train_data,
            test_data=test_data,
            feature_cols=["feature1", "feature2", "feature3"],
            target_col="label",
            experiment_name="Meu_Experimento_Prod"
        )
        
        print(f"Resultados disponíveis em: {run_link}")
    
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
