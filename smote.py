from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, rand, array, udf
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, FloatType
import numpy as np
from itertools import combinations

# 1. Iniciar sessão Spark
spark = SparkSession.builder \
    .appName("GeneralizedSMOTE") \
    .getOrCreate()

# 2. Função para calcular distâncias entre pontos
@udf(returnType=ArrayType(FloatType()))
def calculate_distances(features, all_features):
    point = np.array(features.toArray())
    distances = []
    for other in all_features:
        other_point = np.array(other.toArray())
        distance = float(np.linalg.norm(point - other_point))
        distances.append(distance)
    return distances

# 3. Função para gerar amostras sintéticas
def generate_synthetic_samples(spark_df, numeric_cols, categorical_cols, k=5, sampling_ratio=1.0):
    """
    Gera amostras sintéticas usando SMOTE para dados no PySpark
    
    Args:
        spark_df: DataFrame Spark com os dados originais
        numeric_cols: Lista de colunas numéricas
        categorical_cols: Lista de colunas categóricas
        k: Número de vizinhos mais próximos a considerar
        sampling_ratio: Proporção de oversampling (1.0 = balanceia completamente)
    
    Returns:
        DataFrame Spark com amostras originais + sintéticas
    """
    # Filtrar classe minoritária
    class_counts = spark_df.groupBy("label").count().collect()
    minority_label = sorted([(x.count, x.label) for x in class_counts])[0][1]
    majority_label = 1 - minority_label if spark_df.schema["label"].dataType == IntegerType() else "0"
    
    minority_df = spark_df.filter(col("label") == minority_label)
    majority_df = spark_df.filter(col("label") == majority_label)
    
    count_minority = minority_df.count()
    count_majority = majority_df.count()
    
    # Calcular número de amostras sintéticas necessárias
    num_synthetic = int((count_majority - count_minority) * sampling_ratio)
    if num_synthetic <= 0:
        return spark_df
    
    # Normalizar features numéricas
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
    minority_numeric = assembler.transform(minority_df)
    
    scaler = MinMaxScaler(inputCol="numeric_features", outputCol="scaled_features")
    scaler_model = scaler.fit(minority_numeric)
    minority_scaled = scaler_model.transform(minority_numeric)
    
    # Coletar features escaladas para cálculo de distância
    minority_points = minority_scaled.select("scaled_features").collect()
    all_points = [row.scaled_features for row in minority_points]
    
    # Adicionar distâncias a cada ponto
    minority_with_distances = minority_scaled.withColumn(
        "distances",
        calculate_distances(col("scaled_features"), array([lit(x) for x in all_points]))
    )
    
    # Encontrar k vizinhos mais próximos para cada ponto
    @udf(returnType=ArrayType(ArrayType(FloatType())))
    def get_k_neighbors(distances, k):
        indices = np.argpartition(distances, k)[:k+1]  # +1 para incluir o próprio ponto
        return [[float(i), float(distances[i])] for i in indices if i != 0]  # Excluir o próprio ponto
    
    minority_with_neighbors = minority_with_distances.withColumn(
        "neighbors",
        get_k_neighbors(col("distances"), lit(k))
    )
    
    # Gerar amostras sintéticas
    synthetic_samples = []
    minority_rows = minority_with_neighbors.collect()
    
    for i, row in enumerate(minority_rows):
        if not row.neighbors:
            continue
            
        # Escolher um vizinho aleatório
        neighbor_idx, distance = row.neighbors[np.random.randint(0, len(row.neighbors))]
        neighbor_row = minority_rows[int(neighbor_idx)]
        
        # Interpolar features numéricas
        synthetic_numeric = {}
        for col_name in numeric_cols:
            original_val = row[col_name]
            neighbor_val = neighbor_row[col_name]
            gap = np.random.random()
            synthetic_val = original_val + gap * (neighbor_val - original_val)
            synthetic_numeric[col_name] = float(synthetic_val)
        
        # Manter features categóricas do ponto original
        synthetic_categ = {col_name: row[col_name] for col_name in categorical_cols}
        
        # Criar linha sintética
        synthetic_row = {**synthetic_numeric, **synthetic_categ, "label": minority_label}
        synthetic_samples.append(synthetic_row)
        
        if len(synthetic_samples) >= num_synthetic:
            break
    
    # Criar DataFrame com amostras sintéticas
    synthetic_df = spark.createDataFrame(synthetic_samples)
    
    # Combinar com dados originais
    return spark_df.unionByName(synthetic_df)

# 4. Exemplo de uso
if __name__ == "__main__":
    # Criar dados de exemplo
    data = [
        (1.0, 2.0, "A", 0),
        (1.5, 1.8, "B", 0),
        (2.0, 3.0, "A", 0),
        (3.0, 2.5, "B", 1),
        (3.5, 2.0, "A", 1),
    ]
    columns = ["feature1", "feature2", "category", "label"]
    df = spark.createDataFrame(data, columns)
    
    # Aplicar SMOTE
    numeric_cols = ["feature1", "feature2"]
    categorical_cols = ["category"]
    balanced_df = generate_synthetic_samples(df, numeric_cols, categorical_cols, k=2)
    
    # Mostrar resultados
    print("Contagem original por classe:")
    df.groupBy("label").count().show()
    
    print("Contagem após SMOTE:")
    balanced_df.groupBy("label").count().show()
    
    print("Amostras sintéticas geradas:")
    balanced_df.filter(col("label") == 1).show()