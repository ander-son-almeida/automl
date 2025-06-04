from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from typing import List, Optional, Dict, Tuple
import numpy as np

def calculate_auto_psi(
    df: DataFrame,
    date_col: str,
    features: List[str],
    num_bins: int = 10,
    baseline_months: int = 3,
    min_months_required: int = 6,
    min_samples_per_bin: int = 50
) -> DataFrame:
    """
    Calcula PSI para múltiplas features, determinando automaticamente os meses disponíveis para cada feature.
    
    Args:
        df: DataFrame Spark com os dados
        date_col: Nome da coluna de data (dt_ref)
        features: Lista de features para cálculo do PSI
        num_bins: Número de bins para discretização
        baseline_months: Número de meses para baseline
        min_months_required: Mínimo de meses necessários para análise
        min_samples_per_bin: Mínimo de amostras por bin para considerar válido
    
    Returns:
        DataFrame com colunas: [feature, analysis_month, baseline_window, psi, samples_current, samples_baseline]
    """
    # 1. Pré-processamento básico
    df = df.withColumn("year_month", F.date_format(F.col(date_col), "yyyy-MM"))
    
    # 2. Função para calcular PSI para uma única feature
    def calculate_single_feature_psi(feature_df: DataFrame, feature_name: str) -> List[Dict]:
        # Determinar meses disponíveis para esta feature (não-nulos)
        feature_months = feature_df.filter(F.col(feature_name).isNotNull()) \
                                 .select("year_month").distinct() \
                                 .orderBy("year_month").rdd.flatMap(lambda x: x).collect()
        
        if len(feature_months) < min_months_required:
            print(f"Feature {feature_name} ignorada - apenas {len(feature_months)} meses disponíveis")
            return []
        
        # Discretização usando os primeiros N meses como baseline
        baseline_data = feature_df.filter(F.col("year_month").isin(feature_months[:baseline_months]))
        
        # Amostragem para definir bins se muitos dados
        sample_frac = min(0.2, 1e6 / baseline_data.count()) if baseline_data.count() > 1e6 else 1.0
        bins = baseline_data.sample(sample_frac).select(feature_name).rdd.flatMap(lambda x: x).histogram(num_bins)[1]
        
        # Calcular distribuição para todos os meses
        monthly_dist = feature_df.withColumn("bin", F.expr(f"width_bucket({feature_name}, array({','.join(map(str, bins))})")) \
                               .groupBy("year_month", "bin").agg(F.count("*").alias("count"))
        
        # Calcular PSI para cada mês após a baseline
        results = []
        for i in range(baseline_months, len(feature_months)):
            current_month = feature_months[i]
            baseline_window = feature_months[i-baseline_months:i]
            
            # Verificar se há amostras suficientes
            current_samples = monthly_dist.filter(F.col("year_month") == current_month).agg(F.sum("count")).collect()[0][0]
            baseline_samples = monthly_dist.filter(F.col("year_month").isin(baseline_window)).agg(F.sum("count")).collect()[0][0]
            
            if current_samples < min_samples_per_bin * num_bins or baseline_samples < min_samples_per_bin * num_bins * baseline_months:
                print(f"Feature {feature_name} mês {current_month} ignorada - amostras insuficientes")
                continue
            
            # Calcular distribuições
            baseline_pct = monthly_dist.filter(F.col("year_month").isin(baseline_window)) \
                                     .groupBy("bin").agg((F.sum("count")/baseline_samples).alias("pct"))
            
            current_pct = monthly_dist.filter(F.col("year_month") == current_month) \
                                   .groupBy("bin").agg((F.sum("count")/current_samples).alias("pct"))
            
            # Calcular PSI
            psi = baseline_pct.join(current_pct, "bin", "full") \
                            .fillna(1e-6, ["pct_x", "pct_y"]) \
                            .withColumn("psi_component", (F.col("pct_y") - F.col("pct_x")) * F.log(F.col("pct_y") / F.col("pct_x"))) \
                            .agg(F.sum("psi_component").alias("psi")).collect()[0]["psi"]
            
            results.append({
                "feature": feature_name,
                "analysis_month": current_month,
                "baseline_window": ",".join(baseline_window),
                "psi": float(psi),
                "samples_current": current_samples,
                "samples_baseline": baseline_samples
            })
        
        return results
    
    # 3. Processar cada feature em paralelo (otimizado para Spark)
    from multiprocessing import Pool
    import pandas as pd
    
    # Criar sub-dataframes para cada feature (otimizado)
    feature_dfs = {feature: df.select("year_month", feature).filter(F.col(feature).isNotNull()) for feature in features}
    
    # Calcular PSI para todas as features (paralelizado)
    with Pool(processes=min(4, len(features))) as pool:
        all_results = pool.starmap(calculate_single_feature_psi, [(feature_dfs[feature], feature) for feature in features])
    
    # 4. Consolidar resultados
    flat_results = [item for sublist in all_results for item in sublist]
    
    if not flat_results:
        return spark.createDataFrame([], schema="feature string, analysis_month string, baseline_window string, psi double, samples_current long, samples_baseline long")
    
    # 5. Criar DataFrame final com informações adicionais
    result_pd = pd.DataFrame(flat_results)
    result_df = spark.createDataFrame(result_pd)
    
    # Adicionar categorização do PSI
    result_df = result_df.withColumn("psi_category",
        F.when(F.col("psi") < 0.1, "Stable")
         .when(F.col("psi") < 0.2, "Minor Change")
         .otherwise("Significant Change")) \
        .orderBy("feature", "analysis_month")
    
    return result_df


# Exemplo de uso com detecção automática de meses
psi_results = calculate_auto_psi(
    df=df_gigante,
    date_col="dt_ref",
    features=["score_credito", "valor_transacao", "idade_cliente"],
    num_bins=10,
    baseline_months=3,
    min_months_required=6
)

# Visualizar resultados
psi_results.show(n=20, truncate=False)

# Salvar resultados organizados por feature
(psi_results
 .repartition("feature")
 .write
 .partitionBy("feature")
 .mode("overwrite")
 .parquet("path/to/psi_results_auto"))
