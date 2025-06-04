from typing import List
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType

def calculate_psi_vectorized(
    df: DataFrame,
    date_col: str,
    features: List[str],
    num_bins: int = 10,
    baseline_months: int = 3,
    min_months_required: int = 6
) -> DataFrame:

    # Substituir date_col por mês-formatado, mas mantendo o mesmo nome da coluna original
    df = df.withColumn(date_col, F.date_format(F.col(date_col), "yyyy-MM")).persist()

    # Obter meses disponíveis
    months = [r[date_col] for r in df.select(date_col).distinct().orderBy(date_col).collect()]
    
    if len(months) < min_months_required:
        return spark.createDataFrame([], "feature string, month string, baseline string, psi double")

    baseline_window = months[:baseline_months]
    analysis_months = months[baseline_months:]

    # Amostra para calcular os bins
    baseline_data = df.filter(F.col(date_col).isin(baseline_window))
    sample_frac = min(0.01, 100000 / baseline_data.count())
    baseline_sample = baseline_data.sample(fraction=sample_frac)

    # Gerar bins por feature
    bins_dict = {}
    for feature in features:
        if not isinstance(df.schema[feature].dataType, NumericType):
            baseline_sample = baseline_sample.withColumn(feature, F.col(feature).cast("double"))
        quantiles = baseline_sample.select(feature).approxQuantile(
            feature, [i / num_bins for i in range(num_bins + 1)], 0.05
        )
        bins_dict[feature] = quantiles[1:-1]

    # Adicionar colunas de bin para cada feature
    for feature in features:
        splits = bins_dict[feature]
        bucket_expr = F.when(F.col(feature).isNull(), -1)
        for i, b in enumerate(splits):
            if i == 0:
                bucket_expr = bucket_expr.when(F.col(feature) <= b, i)
            else:
                bucket_expr = bucket_expr.when((F.col(feature) > splits[i-1]) & (F.col(feature) <= b), i)
        bucket_expr = bucket_expr.otherwise(len(splits))
        df = df.withColumn(f"{feature}_bin", bucket_expr)

    # Empilhar: (feature, bin, date_col)
    stacks = []
    for feature in features:
        stacks.append(
            df.select(
                F.lit(feature).alias("feature"),
                F.col(date_col),
                F.col(f"{feature}_bin").alias("bin")
            )
        )
    long_df = stacks[0].unionByName(*stacks[1:])

    # Contar por feature, mês e bin
    counts = long_df.groupBy("feature", date_col, "bin").agg(F.count("*").alias("count"))

    # Marcar baseline ou current
    counts = counts.withColumn(
        "type",
        F.when(F.col(date_col).isin(baseline_window), F.lit("baseline")).otherwise(F.lit("current"))
    )

    # Totais por feature e mês
    totals = counts.groupBy("feature", date_col, "type").agg(F.sum("count").alias("total"))
    counts = counts.join(totals, on=["feature", date_col, "type"])
    counts = counts.withColumn("pct", F.col("count") / F.col("total"))

    # Separar baseline e current
    baseline = counts.filter(F.col("type") == "baseline").select(
        "feature", "bin", date_col, F.col("pct").alias("baseline_pct")
    )
    current = counts.filter(F.col("type") == "current").select(
        "feature", "bin", date_col, F.col("pct").alias("current_pct")
    )

    # Média de baseline por feature/bin
    baseline_avg = baseline.groupBy("feature", "bin").agg(F.avg("baseline_pct").alias("baseline_pct"))

    # Calcular PSI
    psi_df = current.join(baseline_avg, on=["feature", "bin"])
    psi_df = psi_df.withColumn(
        "psi_term",
        (F.col("current_pct") - F.col("baseline_pct")) *
        F.log((F.col("current_pct") + 1e-6) / (F.col("baseline_pct") + 1e-6))
    )

    # Agregar PSI por feature e mês
    psi_final = psi_df.groupBy("feature", date_col).agg(F.sum("psi_term").alias("psi"))
    psi_final = psi_final.withColumnRenamed(date_col, "month") \
                         .withColumn("baseline", F.lit(",".join(baseline_window)))

    df.unpersist()
    return psi_final.orderBy("feature", "month")
