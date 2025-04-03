from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, rand, array, udf
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql.types import ArrayType, FloatType
import numpy as np

# Iniciar sessão Spark
spark = SparkSession.builder.appName("SMOTENumeric").getOrCreate()

# Função UDF para calcular distâncias
@udf(returnType=ArrayType(FloatType()))
def calculate_distances(features, all_features):
    point = np.array(features.toArray())
    distances = []
    for other in all_features:
        other_point = np.array(other.toArray())
        distance = float(np.linalg.norm(point - other_point))
        distances.append(distance)
    return distances

# Função UDF para obter vizinhos mais próximos
@udf(returnType=ArrayType(ArrayType(FloatType())))
def get_k_neighbors(distances, k):
    indices = np.argpartition(distances, k)[:k+1]
    return [[float(i), float(distances[i])] for i in indices if distances[i] > 0]  # Exclui o próprio ponto

# Função principal SMOTE para colunas numéricas
def smote_numeric(df, target_col, numeric_cols, k=5, sampling_ratio=1.0):
    """
    Gera amostras sintéticas para colunas numéricas usando SMOTE
    
    Args:
        df: DataFrame Spark
        target_col: Nome da coluna target (string)
        numeric_cols: Lista de colunas numéricas (list)
        k: Número de vizinhos (int)
        sampling_ratio: Razão de oversampling (float)
    
    Returns:
        DataFrame balanceado
    """
    # Identificar classes
    class_counts = df.groupBy(target_col).count().collect()
    minority_label = sorted([(x.count, x[target_col]) for x in class_counts])[0][1]
    majority_label = sorted([(x.count, x[target_col]) for x in class_counts])[1][1]
    
    minority_df = df.filter(col(target_col) == minority_label)
    majority_df = df.filter(col(target_col) == majority_label)
    
    count_minority = minority_df.count()
    count_majority = majority_df.count()
    
    # Calcular número de amostras sintéticas
    num_synthetic = int((count_majority - count_minority) * sampling_ratio)
    if num_synthetic <= 0:
        return df
    
    # Pré-processamento: Normalizar features
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_vec")
    minority_vec = assembler.transform(minority_df)
    
    scaler = MinMaxScaler(inputCol="features_vec", outputCol="scaled_features")
    scaler_model = scaler.fit(minority_vec)
    minority_scaled = scaler_model.transform(minority_vec)
    
    # Coletar pontos para cálculo de distância
    minority_points = minority_scaled.select("scaled_features").collect()
    all_points = [row.scaled_features for row in minority_points]
    
    # Calcular distâncias
    minority_with_distances = minority_scaled.withColumn(
        "distances", 
        calculate_distances(col("scaled_features"), array([lit(x) for x in all_points]))
    )
    
    # Identificar k vizinhos mais próximos
    minority_with_neighbors = minority_with_distances.withColumn(
        "neighbors",
        get_k_neighbors(col("distances"), lit(k))
    )
    
    # Gerar amostras sintéticas
    synthetic_samples = []
    minority_rows = minority_with_neighbors.collect()
    
    for i in range(num_synthetic):
        # Selecionar ponto aleatório da minoria
        idx = np.random.randint(0, len(minority_rows))
        row = minority_rows[idx]
        
        if not row.neighbors:
            continue
            
        # Selecionar vizinho aleatório
        neighbor_idx, _ = row.neighbors[np.random.randint(0, len(row.neighbors))]
        neighbor_row = minority_rows[int(neighbor_idx)]
        
        # Gerar ponto sintético
        synthetic_row = {}
        for col_name in numeric_cols:
            original_val = row[col_name]
            neighbor_val = neighbor_row[col_name]
            gap = np.random.random()
            synthetic_val = original_val + gap * (neighbor_val - original_val)
            synthetic_row[col_name] = float(synthetic_val)
        
        synthetic_row[target_col] = minority_label
        synthetic_samples.append(synthetic_row)
    
    # Combinar com dados originais
    if synthetic_samples:
        synthetic_df = spark.createDataFrame(synthetic_samples)
        return df.unionByName(synthetic_df)
    return df

# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    data = [
        (1.0, 2.0, 0),
        (1.5, 1.8, 0),
        (2.0, 3.0, 0),
        (3.0, 2.5, 1),
        (3.5, 2.0, 1),
    ]
    columns = ["feature1", "feature2", "label"]
    df = spark.createDataFrame(data, columns)
    
    # Parâmetros
    target_column = "label"
    numeric_columns = ["feature1", "feature2"]
    
    # Aplicar SMOTE
    balanced_df = smote_numeric(
        df=df,
        target_col=target_column,
        numeric_cols=numeric_columns,
        k=2,
        sampling_ratio=1.0
    )
    
    # Resultados
    print("Distribuição original:")
    df.groupBy("label").count().show()
    
    print("Distribuição após SMOTE:")
    balanced_df.groupBy("label").count().show()