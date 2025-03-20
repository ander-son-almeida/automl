from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors

# Inicializar a sessão do Spark
spark = SparkSession.builder.appName("KS Calculation with DenseVector").getOrCreate()

# Exemplo de dados com DenseVector contendo duas probabilidades
data = [
    (0, Vectors.dense([0.9, 0.1])),  # label, probability (DenseVector com [P(0), P(1)])
    (1, Vectors.dense([0.6, 0.4])),
    (0, Vectors.dense([0.65, 0.35])),
    (1, Vectors.dense([0.2, 0.8])),
    (0, Vectors.dense([0.8, 0.2])),
    (1, Vectors.dense([0.1, 0.9])),
    (0, Vectors.dense([0.7, 0.3])),
    (1, Vectors.dense([0.3, 0.7])),
    (0, Vectors.dense([0.95, 0.05])),
    (1, Vectors.dense([0.4, 0.6]))
]

# Criar DataFrame
df = spark.createDataFrame(data, ["label", "probability"])

# Extrair a probabilidade da classe positiva (segundo elemento do DenseVector)
df = df.withColumn("probability_positive", F.col("probability").getItem(1))

# Ordenar as probabilidades em ordem decrescente
window_spec = Window.orderBy(F.desc("probability_positive"))
df = df.withColumn("rank", F.row_number().over(window_spec))

# Calcular o total de positivos e negativos
total_positives = df.filter(F.col("label") == 1).count()
total_negatives = df.filter(F.col("label") == 0).count()

# Calcular as distribuições cumulativas
df = df.withColumn(
    "cumulative_positives",
    F.sum(F.when(F.col("label") == 1, 1).otherwise(0)).over(window_spec)
).withColumn(
    "cumulative_negatives",
    F.sum(F.when(F.col("label") == 0, 1).otherwise(0)).over(window_spec)
)

# Calcular TPR e FPR
df = df.withColumn(
    "TPR",
    F.col("cumulative_positives") / total_positives
).withColumn(
    "FPR",
    F.col("cumulative_negatives") / total_negatives
)

# Calcular a diferença KS (TPR - FPR)
df = df.withColumn("KS", F.col("TPR") - F.col("FPR"))

# Encontrar o valor máximo de KS
ks_statistic = df.agg(F.max("KS")).collect()[0][0]

print(f"KS Statistic: {ks_statistic:.4f}")

# Parar a sessão do Spark
spark.stop()



from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, row_number
from pyspark.sql.window import Window
import numpy as np

# Criar sessão Spark
spark = SparkSession.builder.appName("KS Test for Binary Classification").getOrCreate()

# Exemplo de dados
data = [
    (0.8, 1),
    (0.3, 0),
    (0.6, 1),
    (0.4, 0),
    (0.7, 1),
    (0.2, 0)
]

# Criar DataFrame
df = spark.createDataFrame(data, ["probabilidade", "classe_real"])

# Separar as probabilidades por classe
df_class_1 = df.filter(col("classe_real") == 1).select("probabilidade")
df_class_0 = df.filter(col("classe_real") == 0).select("probabilidade")

# Calcular as CDFs para cada classe
window_spec = Window.orderBy("probabilidade")

df_class_1 = df_class_1.withColumn("cdf", row_number().over(window_spec) / lit(df_class_1.count()))
df_class_0 = df_class_0.withColumn("cdf", row_number().over(window_spec) / lit(df_class_0.count()))

# Juntar as CDFs em um único DataFrame
df_cdf = df_class_1.select("probabilidade", "cdf").union(df_class_0.select("probabilidade", "cdf"))

# Calcular a diferença entre as CDFs
df_cdf = df_cdf.withColumn("diff", col("cdf").cast("double") - col("cdf").cast("double"))

# Encontrar a diferença máxima (estatística KS)
ks_statistic = df_cdf.agg({"diff": "max"}).collect()[0][0]
print(f"KS Statistic: {ks_statistic}")

# Fechar sessão Spark
spark.stop()