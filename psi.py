from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from typing import List
import sys

def calculate_psi_final(
    df: DataFrame,
    date_col: str,
    features: List[str],
    num_bins: int = 10,
    baseline_months: int = 3,
    min_months_required: int = 6,
    min_samples_per_bin: int = 50
) -> DataFrame:
    """
    Calcula PSI para múltiplas features, tratando datas no formato 'yyyy-MM-dd' (último dia do mês).
    
    Args:
        df: DataFrame Spark com os dados
        date_col: Nome da coluna de data no formato 'yyyy-MM-dd' (último dia do mês)
        features: Lista de features para cálculo do PSI
        num_bins: Número de bins para discretização
        baseline_months: Número de meses para baseline
        min_months_required: Mínimo de meses necessários para análise
        min_samples_per_bin: Mínimo de amostras por bin para considerar válido
    
    Returns:
        DataFrame com colunas: [feature, analysis_month, baseline_window, psi, samples_current, samples_baseline, psi_category]
    """
    # 1. Pré-processamento básico - converter para formato 'yyyy-MM' extraindo do 'yyyy-MM-dd'
    df = df.withColumn("year_month", F.date_format(F.col(date_col), "yyyy-MM"))
    
    # 2. Estrutura para armazenar resultados
    schema = ["feature string", "analysis_month string", "baseline_window string", 
              "psi double", "samples_current long", "samples_baseline long",
              "psi_category string"]
    empty_df = spark.createDataFrame([], ",".join(schema))
    results = empty_df
    
    for feature in features:
        try:
            print(f"\nProcessando feature: {feature}")
            
            # 3. Filtrar dados não-nulos para a feature atual (excluindo a coluna de data)
            feature_df = df.filter(F.col(feature).isNotNull()).select("year_month", feature)
            
            # 4. Obter meses disponíveis ordenados
            months = [row['year_month'] for row in feature_df.select("year_month").distinct().orderBy("year_month").collect()]
            
            if len(months) < min_months_required:
                print(f"Ignorando {feature} - apenas {len(months)} meses disponíveis")
                continue
            
            # 5. Calcular bins usando apenas os dados da feature (não inclui a coluna de data)
            baseline_df = feature_df.filter(F.col("year_month").isin(months[:baseline_months]))
            
            # Verificar se a feature é numérica
            if not isinstance(df.schema[feature].dataType, NumericType):
                print(f"Feature {feature} não é numérica. Convertendo para double...")
                baseline_df = baseline_df.withColumn(feature, F.col(feature).cast("double"))
            
            # Calcular bins apenas para a feature atual
            bins = baseline_df.select(feature).rdd.flatMap(lambda x: x).histogram(num_bins)[1]
            bin_expr = f"width_bucket({feature}, array({','.join(map(str, bins))}))"
            
            # 6. Calcular distribuição para todos os meses
            monthly_dist = feature_df.withColumn("bin", F.expr(bin_expr)) \
                                   .groupBy("year_month", "bin").agg(F.count("*").alias("count"))
            
            # 7. Calcular totais por mês
            monthly_totals = monthly_dist.groupBy("year_month").agg(F.sum("count").alias("total"))
            
            # 8. Juntar e calcular porcentagens
            dist_pct = monthly_dist.join(monthly_totals, "year_month") \
                                 .withColumn("pct", F.col("count") / F.col("total")) \
                                 .select("year_month", "bin", "pct", "count")
            
            # 9. Para cada mês após a baseline, calcular PSI
            for i in range(baseline_months, len(months)):
                current_month = months[i]
                baseline_window = months[i-baseline_months:i]
                
                # Verificar amostras suficientes
                current_samples = dist_pct.filter(F.col("year_month") == current_month) \
                                       .agg(F.sum("count")).collect()[0][0] or 0
                                       
                baseline_samples = dist_pct.filter(F.col("year_month").isin(baseline_window)) \
                                         .agg(F.sum("count")).collect()[0][0] or 0
                
                if (current_samples < min_samples_per_bin * num_bins or 
                    baseline_samples < min_samples_per_bin * num_bins * baseline_months):
                    print(f"Ignorando {feature} {current_month} - amostras insuficientes")
                    continue
                
                # Calcular distribuições
                baseline_pct = dist_pct.filter(F.col("year_month").isin(baseline_window)) \
                                     .groupBy("bin") \
                                     .agg((F.sum("count") / baseline_samples).alias("baseline_pct"))
                
                current_pct = dist_pct.filter(F.col("year_month") == current_month) \
                                    .groupBy("bin") \
                                    .agg((F.sum("count") / current_samples).alias("current_pct"))
                
                # Calcular PSI
                psi_df = baseline_pct.join(current_pct, "bin", "full") \
                                   .fillna(1e-6, ["baseline_pct", "current_pct"]) \
                                   .withColumn("psi_component", 
                                              (F.col("current_pct") - F.col("baseline_pct")) * 
                                              F.log(F.col("current_pct") / F.col("baseline_pct")))
                
                psi_row = psi_df.agg(
                    F.sum("psi_component").alias("psi"),
                    F.lit(current_samples).alias("samples_current"),
                    F.lit(baseline_samples).alias("samples_baseline")
                ).withColumn("feature", F.lit(feature)) \
                 .withColumn("analysis_month", F.lit(current_month)) \
                 .withColumn("baseline_window", F.lit(",".join(baseline_window)))
                
                # Adicionar categoria PSI
                psi_row = psi_row.withColumn("psi_category",
                    F.when(F.col("psi") < 0.1, "Stable")
                     .when(F.col("psi") < 0.2, "Minor Change")
                     .otherwise("Significant Change"))
                
                # Adicionar ao DataFrame de resultados
                results = results.unionByName(psi_row.select(*schema))
                
        except Exception as e:
            print(f"Erro ao processar feature {feature}: {str(e)}")
            continue
    
    # 10. Ordenar resultados finais
    return results.orderBy("feature", "analysis_month")
