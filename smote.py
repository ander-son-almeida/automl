from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from smotePySpark import SmotePySpark

# 1. Iniciar sessão Spark
spark = SparkSession.builder \
    .appName("SMOTE in PySpark") \
    .getOrCreate()

# 2. Carregar dados (exemplo com dataset desbalanceado)
# Substitua pelo seu DataFrame
data = [
    (1.0, 2.0, 0),
    (1.5, 1.8, 0),
    (2.0, 3.0, 0),
    (3.0, 2.5, 1),  # Classe 1 (minoria)
    (3.5, 2.0, 1),  # Classe 1 (minoria)
]
columns = ["feature1", "feature2", "target"]
df = spark.createDataFrame(data, columns)

# 3. Criar Vetor de Features (requerido pelo SMOTE)
assembler = VectorAssembler(
    inputCols=["feature1", "feature2"],
    outputCol="features"
)
df_assembled = assembler.transform(df)

# 4. Aplicar SMOTE
smote = SmotePySpark(
    k=5,  # Número de vizinhos (padrão=5)
    sampling_strategy="auto",  # Balanceia para a classe minoritária
    featuresCol="features",
    labelCol="target",
    random_seed=42
)

# 5. Balancear os dados
df_balanced = smote.sample(df_assembled)

# 6. Mostrar resultados
print("Antes do SMOTE (contagem de classes):")
df.groupBy("target").count().show()

print("Depois do SMOTE (contagem de classes):")
df_balanced.groupBy("target").count().show()



