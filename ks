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