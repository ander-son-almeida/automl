from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler, StringIndexer, BucketedRandomProjectionLSH
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.linalg import VectorUDT
import random
from functools import reduce
from typing import List

class SmoteConfig:
    def __init__(self, k=5, multiplier=1, bucketLength=2.0, seed=42):
        self.k = k  # número de vizinhos mais próximos a considerar
        self.multiplier = multiplier  # quantas vezes replicar os dados sintéticos
        self.bucketLength = bucketLength  # parâmetro para LSH
        self.seed = seed  # semente para reprodutibilidade

def smote_spark(df: DataFrame, target_col: str, num_cols: List[str], cat_cols: List[str] = None, config: SmoteConfig = None) -> DataFrame:
    """
    Aplica SMOTE para oversampling da classe minoritária em um DataFrame Spark.
    
    Args:
        df: DataFrame Spark contendo os dados
        target_col: Nome da coluna alvo binária (1 = classe minoritária)
        num_cols: Lista de colunas numéricas
        cat_cols: Lista de colunas categóricas (opcional)
        config: Configurações do SMOTE (opcional)
        
    Returns:
        DataFrame Spark com oversampling aplicado
    """
    if cat_cols is None:
        cat_cols = []
    
    if config is None:
        config = SmoteConfig()
    
    # Verifica se a coluna alvo é binária
    distinct_values = df.select(target_col).distinct().collect()
    if len(distinct_values) != 2:
        raise ValueError("A coluna alvo deve ter exatamente 2 valores distintos")
    
    # Pré-processamento: preparar os dados para SMOTE
    processed_df = _preprocess_for_smote(df, num_cols, cat_cols, target_col)
    
    # Aplicar SMOTE
    oversampled_df = _apply_smote(processed_df, config)
    
    return oversampled_df

def _preprocess_for_smote(df: DataFrame, num_cols: List[str], cat_cols: List[str], target_col: str) -> DataFrame:
    """Prepara o DataFrame para SMOTE."""
    # Remove a target_col das num_cols se estiver presente
    if target_col in num_cols:
        num_cols.remove(target_col)
    
    # Indexador para colunas categóricas (exceto target)
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index").fit(df) 
        for col in cat_cols 
        if col != target_col
    ]
    
    # Assembler para colunas numéricas
    assembler = VectorAssembler(inputCols=num_cols, outputCol="features")
    
    # Pipeline de transformação
    pipeline = Pipeline(stages=indexers + [assembler])
    transformed_df = pipeline.fit(df).transform(df)
    
    # Seleciona apenas colunas necessárias e renomeia target_col para "label"
    keep_cols = [col for col in transformed_df.columns 
                if col not in num_cols + cat_cols] + ["features"]
    
    return transformed_df.select(*keep_cols).withColumnRenamed(target_col, "label")

def _apply_smote(df: DataFrame, config: SmoteConfig) -> DataFrame:
    """Aplica o algoritmo SMOTE ao DataFrame preparado."""
    # Separa classes minoritária (1) e majoritária (0)
    df_min = df.filter(F.col("label") == 1)
    df_maj = df.filter(F.col("label") == 0)
    
    # Se não há instâncias minoritárias, retorna o original
    if df_min.count() == 0:
        return df
    
    # Aplica LSH para encontrar vizinhos mais próximos
    brp = BucketedRandomProjectionLSH(
        inputCol="features",
        outputCol="hashes",
        seed=config.seed,
        bucketLength=config.bucketLength
    )
    model = brp.fit(df_min)
    
    # Encontra k vizinhos mais próximos para cada ponto
    self_join = model.approxSimilarityJoin(
        df_min, df_min, float("inf"), distCol="EuclideanDistance"
    ).filter("EuclideanDistance > 0")  # remove auto-comparações
    
    # Ordena por distância e seleciona os k mais próximos
    window = Window.partitionBy("datasetA").orderBy("EuclideanDistance")
    neighbors_df = self_join.withColumn("rn", F.row_number().over(window)) \
                          .filter(F.col("rn") <= config.k) \
                          .drop("rn")
    
    # Gera dados sintéticos
    synthetic_data = []
    
    # UDFs para operações vetoriais
    subtract_udf = F.udf(lambda arr: random.uniform(0, 1) * (arr[0] - arr[1]), VectorUDT())
    add_udf = F.udf(lambda arr: arr[0] + arr[1], VectorUDT())
    
    original_cols = [col for col in df_min.columns if col != "features"]
    
    for _ in range(config.multiplier):
        # Seleciona vizinho aleatório para cada ponto
        rand_df = neighbors_df.withColumn("rand", F.rand(seed=config.seed))
        max_rand_df = rand_df.withColumn("max_rand", F.max("rand").over(Window.partitionBy("datasetA")))
        selected_df = max_rand_df.filter(F.col("rand") == F.col("max_rand")).drop("rand", "max_rand")
        
        # Cria features sintéticas
        diff_df = selected_df.select("*", subtract_udf(F.array("datasetA.features", "datasetB.features")).alias("diff"))
        synth_df = diff_df.select("*", add_udf(F.array("datasetA.features", "diff")).alias("features"))
        
        # Copia outras colunas (aleatoriamente do original ou vizinho)
        for col_name in original_cols:
            choice = random.choice(["datasetA", "datasetB"])
            synth_df = synth_df.withColumn(col_name, F.col(f"{choice}.{col_name}"))
        
        synth_df = synth_df.drop("datasetA", "datasetB", "diff", "EuclideanDistance")
        synthetic_data.append(synth_df)
    
    # Combina todos os dados sintéticos
    if synthetic_data:
        all_synthetic = reduce(DataFrame.unionAll, synthetic_data)
        # Retorna dados originais + sintéticos
        return df_maj.unionByName(df_min).unionByName(all_synthetic)
    else:
        return df


# Exemplo de uso:
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SMOTE_Example").getOrCreate()

# 1. Crie seu DataFrame Spark (df)
# 2. Defina quais colunas são numéricas e categóricas
num_cols = ["idade", "renda", "valor_emprestimo"]  # substitua com suas colunas numéricas
cat_cols = ["estado_civil", "escolaridade"]       # substitua com suas colunas categóricas
target_col = "inadimplente"                       # substitua com sua coluna alvo (1 = classe minoritária)

# 3. (Opcional) Configure os parâmetros do SMOTE
smote_config = SmoteConfig(
    k=5,           # número de vizinhos mais próximos
    multiplier=2,   # quantas vezes replicar os dados sintéticos
    bucketLength=2, # parâmetro para LSH
    seed=42         # semente para reprodutibilidade
)

# 4. Aplique SMOTE
df_balanced = smote_spark(
    df=df,
    target_col=target_col,
    num_cols=num_cols,
    cat_cols=cat_cols,
    config=smote_config
)

# 5. Mostre o resultado
print("Antes do SMOTE:")
df.groupBy(target_col).count().show()

print("Depois do SMOTE:")
df_balanced.groupBy("label").count().show()