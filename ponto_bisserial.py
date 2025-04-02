from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, count, sqrt, abs, lit
from pyspark.sql.types import FloatType

# Iniciar sessão Spark
spark = SparkSession.builder.appName("feature_analysis").getOrCreate()

# DataFrame de exemplo (substitua pelo seu dataset)
data = [
    (1, 10.5, 200, None, 0),
    (2, 8.0, 150, 30.0, 1),
    (3, None, 180, 25.0, 1),
    (4, 12.0, 220, 40.0, 0)
]
df = spark.createDataFrame(data, ["ID", "feature1", "feature2", "feature3", "target"])

# Lista de features numéricas (excluindo colunas não-numéricas)
features = [f for f in df.columns if f not in ["ID", "target"]]

# Função para calcular todas as estatísticas + ponto-bisserial
def calculate_full_stats(df, feature_name):
    # Agrupar por target e calcular estatísticas
    stats = df.groupBy("target").agg(
        mean(feature_name).alias("mean"),
        stddev(feature_name).alias("stddev"),
        count(feature_name).alias("n")
    ).collect()
    
    # Extrair estatísticas para target=0 e target=1
    if len(stats) == 2:
        stats_0 = stats[0]  # target=0
        stats_1 = stats[1]  # target=1
    else:
        # Caso um dos grupos não tenha dados (ex.: todos NaN)
        stats_0 = {"mean": None, "stddev": None, "n": 0}
        stats_1 = {"mean": None, "stddev": None, "n": 0}
    
    n_0 = stats_0["n"]
    n_1 = stats_1["n"]
    mean_0 = stats_0["mean"]
    mean_1 = stats_1["mean"]
    stddev_0 = stats_0["stddev"] if stats_0["stddev"] is not None else 0.0
    stddev_1 = stats_1["stddev"] if stats_1["stddev"] is not None else 0.0
    
    # Calcular a correlação ponto-bisserial (r_pb)
    if n_0 == 0 or n_1 == 0 or (stddev_0 == 0 and stddev_1 == 0):
        r_pb = None
    else:
        pooled_std = sqrt(((n_0 - 1) * stddev_0 ** 2 + (n_1 - 1) * stddev_1 ** 2) / (n_0 + n_1 - 2))
        r_pb = (mean_1 - mean_0) * sqrt(n_0 * n_1 / (n_0 + n_1) ** 2) / pooled_std
    
    # Retornar todas as estatísticas em um dicionário
    return {
        "feature": feature_name,
        "mean_0": mean_0,
        "mean_1": mean_1,
        "stddev_0": stddev_0,
        "stddev_1": stddev_1,
        "n_0": n_0,
        "n_1": n_1,
        "r_pb": float(r_pb) if r_pb is not None else None
    }

# Calcular para todas as features
results = []
for feature in features:
    results.append(calculate_full_stats(df, feature))

# Criar DataFrame com todas as estatísticas
results_df = spark.createDataFrame(results)

# Mostrar o resultado completo
results_df.show(truncate=False)
