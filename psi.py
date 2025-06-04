from pyspark.sql import functions as F
from pyspark.sql.window import Window
import numpy as np

def calculate_optimized_psi(df, date_col, value_col, num_bins=10, min_date=None, max_date=None):
    """
    Versão ultra-otimizada para cálculo de PSI com janela de 3 meses.
    
    Args:
        df: DataFrame Spark
        date_col: Nome da coluna de data
        value_col: Nome da coluna de valores
        num_bins: Número de bins (padrão 10)
        min_date/max_date: Opcional para filtrar período
    
    Returns:
        DataFrame com PSI para cada mês comparado aos 3 meses anteriores
    """
    # 1. Filtro inicial por datas (se fornecido)
    if min_date or max_date:
        date_filter = []
        if min_date:
            date_filter.append(F.col(date_col) >= min_date)
        if max_date:
            date_filter.append(F.col(date_col) <= max_date)
        df = df.filter(*date_filter)

    # 2. Discretização otimizada usando amostragem para definir bins
    sample_frac = 0.1 if df.count() > 1e6 else 1.0  # Amostra 10% se > 1M registros
    bins = df.sample(fraction=sample_frac).select(value_col).rdd.flatMap(lambda x: x).histogram(num_bins)[1]
    
    # 3. Pré-processamento em uma única passagem
    processed = df.withColumn("year_month", F.date_format(F.col(date_col), "yyyy-MM")) \
                 .withColumn("bin", F.expr(f"width_bucket({value_col}, array({','.join(map(str, bins))})")) \
                 .groupBy("year_month", "bin").agg(F.count("*").alias("count"))

    # 4. Persistir os dados processados
    processed.persist()

    # 5. Calcular totais por mês (otimizado)
    monthly_totals = processed.groupBy("year_month").agg(F.sum("count").alias("total"))
    
    # 6. Coletar meses ordenados (otimizado para evitar múltiplas ações)
    months = [r['year_month'] for r in processed.select("year_month").distinct().orderBy("year_month").collect()]

    # 7. Pré-calcular todas as distribuições mensais
    dist_df = processed.join(monthly_totals, "year_month") \
                      .withColumn("pct", F.col("count") / F.col("total")) \
                      .select("year_month", "bin", "pct")
    
    # 8. Calcular PSI para cada mês (a partir do 4º)
    results = []
    for i in range(3, len(months)):
        current_month = months[i]
        ref_months = months[i-3:i]
        
        # Calcular distribuição de referência (otimizado)
        ref_pct = dist_df.filter(F.col("year_month").isin(ref_months)) \
                        .groupBy("bin").agg(F.avg("pct").alias("ref_pct"))
        
        # Join e cálculo PSI (otimizado)
        psi = dist_df.filter(F.col("year_month") == current_month) \
                    .join(ref_pct, "bin", "full") \
                    .fillna(1e-6, ["pct", "ref_pct"]) \
                    .withColumn("psi_component", 
                              (F.col("pct") - F.col("ref_pct")) * F.log(F.col("pct") / F.col("ref_pct"))) \
                    .agg(F.sum("psi_component").alias("psi")) \
                    .collect()[0]["psi"]
        
        results.append((current_month, float(psi)))

    # 9. Limpar persistência
    processed.unpersist()
    
    return df.sql_ctx.createDataFrame(results, ["month", "psi"]).orderBy("month")
