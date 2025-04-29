import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Inicializar sessão Spark
spark = SparkSession.builder.appName("ScorecardAnalysis").getOrCreate()

# ====================================================
# PASSO 1: Simular dados de exemplo (substitua pelos seus dados reais)
# ====================================================
data = {
    "id": range(1, 1001),
    "escore_bruto": [float(x) for x in np.random.normal(3, 1, 1000)],  # Exemplo: escores normalmente distribuídos
    "target": [0 if x < 0.7 else 1 for x in np.random.uniform(0, 1, 1000)]  # 70% bons (0), 30% maus (1)
}
df = spark.createDataFrame(pd.DataFrame(data))

# ====================================================
# PASSO 2: Criar 10 faixas de escore bruto (usando percentis)
# ====================================================
df = df.withColumn(
    "faixa",
    F.ntile(10).over(Window.orderBy("escore_bruto"))
)

# ====================================================
# PASSO 3: Calcular estatísticas por faixa (bins/métricas)
# ====================================================
df_stats = df.groupBy("faixa").agg(
    F.min("escore_bruto").alias("min_escore"),
    F.max("escore_bruto").alias("max_escore"),
    F.count("*").alias("total_clientes"),
    F.sum(F.when(F.col("target") == 0, 1).otherwise(0)).alias("bons"),
    F.sum(F.when(F.col("target") == 1, 1).otherwise(0)).alias("maus"),
    F.avg("escore_bruto").alias("media_escore")
).orderBy("faixa")

# Calcular PD (Probabilidade de Default) e Log-Odds por faixa
df_stats = df_stats.withColumn("pd", F.col("maus") / F.col("total_clientes")) \
                   .withColumn("log_odds", F.log((1 - F.col("pd")) / F.col("pd")))

# ====================================================
# PASSO 4: Regressão Linear para alinhamento do escore
# ====================================================
assembler = VectorAssembler(inputCols=["media_escore"], outputCol="features")
df_features = assembler.transform(df_stats)

lr = LinearRegression(featuresCol="features", labelCol="log_odds")
model = lr.fit(df_features)
df_stats = model.transform(df_features).withColumnRenamed("prediction", "escore_alinhado")

# Coeficientes para a fórmula final
inclinacao = model.coefficients[0]
intercepto = model.intercept
print(f"Fórmula: escore_alinhado = {intercepto:.2f} + {inclinacao:.2f} * escore_bruto")

# ====================================================
# PASSO 5: Aplicar alinhamento aos dados originais
# ====================================================
df = df.withColumn(
    "escore_alinhado",
    intercepto + inclinacao * F.col("escore_bruto")
)

# ====================================================
# PASSO 6: Calcular KS, Gini e AUC por faixa
# ====================================================
# Função para calcular KS
def calculate_ks(spark_df, faixa):
    df_faixa = spark_df.filter(F.col("faixa") == faixa)
    window = Window.orderBy("escore_alinhado")
    df_faixa = df_faixa.withColumn("cum_bons", F.sum(F.when(F.col("target") == 0, 1).otherwise(0)).over(window)) \
                      .withColumn("cum_maus", F.sum(F.when(F.col("target") == 1, 1).otherwise(0)).over(window)) \
                      .withColumn("cum_bons_norm", F.col("cum_bons") / F.sum("cum_bons").over(Window.partitionBy())) \
                      .withColumn("cum_maus_norm", F.col("cum_maus") / F.sum("cum_maus").over(Window.partitionBy()))
    ks = df_faixa.agg(F.max(F.col("cum_bons_norm") - F.col("cum_maus_norm")).alias("ks")).collect()[0]["ks"]
    return float(ks) if ks is not None else 0.0

# Função para calcular AUC/Gini (usando BinaryClassificationEvaluator)
def calculate_auc_gini(spark_df, faixa):
    df_faixa = spark_df.filter(F.col("faixa") == faixa)
    if df_faixa.count() == 0:
        return 0.0, 0.0
    evaluator = BinaryClassificationEvaluator(labelCol="target", rawPredictionCol="escore_alinhado", metricName="areaUnderROC")
    auc = evaluator.evaluate(df_faixa)
    gini = 2 * auc - 1
    return float(auc), float(gini)

# Calcular métricas para cada faixa
metrics = []
for faixa in range(1, 11):
    ks = calculate_ks(df, faixa)
    auc, gini = calculate_auc_gini(df, faixa)
    metrics.append({
        "faixa": faixa,
        "ks": ks,
        "auc": auc,
        "gini": gini
    })

# Juntar métricas com estatísticas
df_metrics = spark.createDataFrame(metrics)
df_final = df_stats.join(df_metrics, on="faixa", how="left")

# ====================================================
# PASSO 7: Formatar tabela final em Pandas
# ====================================================
# Converter para Pandas e formatar intervalos
pd_final = df_final.toPandas()
pd_final["intervalo_escore"] = pd_final.apply(
    lambda row: f"{row['min_escore']:.1f} - {row['max_escore']:.1f}", axis=1
)

# Selecionar colunas de interesse
pd_final = pd_final[[
    "intervalo_escore", "bons", "maus", "total_clientes", "pd",
    "ks", "auc", "gini"
]]

# Adicionar linha de médias
pd_final.loc["Média"] = pd_final.mean(numeric_only=True)
pd_final.at["Média", "intervalo_escore"] = "TOTAL"

# Exibir tabela
print(pd_final)

# Salvar em CSV (opcional)
pd_final.to_csv("resultado_metricas.csv", index=False)
