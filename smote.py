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

from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler, StringIndexer, BucketedRandomProjectionLSH
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.linalg import VectorUDT
import random
from functools import reduce
from typing import List, Optional

class SmoteConfig:
    def __init__(self, k=5, multiplier=1, bucketLength=2.0, seed=42):
        self.k = k
        self.multiplier = multiplier
        self.bucketLength = bucketLength
        self.seed = seed

def smote_spark_dropna(df: DataFrame, target_col: str, num_cols: List[str], 
                      cat_cols: Optional[List[str]] = None, 
                      config: Optional[SmoteConfig] = None) -> DataFrame:
    """
    Aplica SMOTE removendo linhas com NaN e evitando ambiguidade de colunas.
    
    Args:
        df: DataFrame Spark
        target_col: Nome da coluna alvo binária (1 = classe minoritária)
        num_cols: Lista de colunas numéricas
        cat_cols: Lista de colunas categóricas
        config: Configurações do SMOTE
        
    Returns:
        DataFrame balanceado sem NaN
    """
    if cat_cols is None:
        cat_cols = []
    
    if config is None:
        config = SmoteConfig()
    
    # 1. Remover linhas com NaN em qualquer coluna usada no SMOTE
    non_null_df = df.dropna(subset=num_cols + cat_cols + [target_col])
    
    # 2. Pré-processamento sem NaN
    processed_df = _preprocess_dropna(non_null_df, num_cols, cat_cols, target_col)
    
    # 3. Aplicar SMOTE seguro
    oversampled_df = _apply_smote_dropna(processed_df, config)
    
    return oversampled_df

def _preprocess_dropna(df: DataFrame, num_cols: List[str], 
                      cat_cols: List[str], target_col: str) -> DataFrame:
    """Pré-processamento removendo NaN e preparando colunas."""
    # Indexar colunas categóricas (exceto target)
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index").fit(df) 
        for col in cat_cols 
        if col != target_col
    ]
    
    # Assembler para colunas numéricas (com nome único)
    assembler = VectorAssembler(inputCols=num_cols, outputCol="smote_features")
    
    # Pipeline
    pipeline = Pipeline(stages=indexers + [assembler])
    transformed_df = pipeline.fit(df).transform(df)
    
    # Selecionar colunas mantendo apenas as necessárias
    keep_cols = [
        col for col in transformed_df.columns 
        if col not in num_cols + cat_cols
    ] + ["smote_features"]
    
    return transformed_df.select(*keep_cols).withColumnRenamed(target_col, "label")

def _apply_smote_dropna(df: DataFrame, config: SmoteConfig) -> DataFrame:
    """Aplica SMOTE garantindo uso apenas de dados completos."""
    # Garantir nome único para features
    working_df = df.withColumnRenamed("smote_features", "features_for_smote")
    
    # Separar classes
    df_min = working_df.filter(F.col("label") == 1)
    df_maj = working_df.filter(F.col("label") == 0)
    
    if df_min.count() == 0:
        return df.drop("smote_features")
    
    # Configurar LSH
    brp = BucketedRandomProjectionLSH(
        inputCol="features_for_smote",
        outputCol="hashes",
        seed=config.seed,
        bucketLength=config.bucketLength
    )
    
    # Ajustar modelo apenas com dados completos
    model = brp.fit(df_min.select("features_for_smote", *[c for c in df_min.columns if c != "features_for_smote"]))
    
    # Similaridade com aliases explícitos
    self_join = model.approxSimilarityJoin(
        df_min.alias("min_clean_A"), 
        df_min.alias("min_clean_B"), 
        float("inf"), 
        distCol="EuclideanDistance"
    ).filter("EuclideanDistance > 0")
    
    # Selecionar k vizinhos mais próximos
    window = Window.partitionBy("min_clean_A").orderBy("EuclideanDistance")
    neighbors_df = self_join.withColumn("rn", F.row_number().over(window)) \
                          .filter(F.col("rn") <= config.k) \
                          .drop("rn")
    
    # Gerar dados sintéticos
    synthetic_data = []
    subtract_udf = F.udf(lambda arr: random.uniform(0, 1) * (arr[0] - arr[1]), VectorUDT())
    add_udf = F.udf(lambda arr: arr[0] + arr[1], VectorUDT())
    
    original_cols = [col for col in df_min.columns if col != "features_for_smote"]
    
    for _ in range(config.multiplier):
        # Selecionar vizinho aleatório
        rand_df = neighbors_df.withColumn("rand", F.rand(seed=config.seed))
        max_rand_df = rand_df.withColumn("max_rand", F.max("rand").over(Window.partitionBy("min_clean_A")))
        selected_df = max_rand_df.filter(F.col("rand") == F.col("max_rand")).drop("rand", "max_rand")
        
        # Criar features sintéticas
        diff_df = selected_df.select("*", subtract_udf(F.array("min_clean_A.features_for_smote", "min_clean_B.features_for_smote")).alias("diff"))
        synth_df = diff_df.select("*", add_udf(F.array("min_clean_A.features_for_smote", "diff")).alias("features_for_smote"))
        
        # Copiar outras colunas (50% de chance para original/vizinho)
        for col_name in original_cols:
            choice = random.choice(["min_clean_A", "min_clean_B"])
            synth_df = synth_df.withColumn(col_name, F.col(f"{choice}.{col_name}"))
        
        synth_df = synth_df.drop("min_clean_A", "min_clean_B", "diff", "EuclideanDistance")
        synthetic_data.append(synth_df)
    
    # Combinar resultados
    if synthetic_data:
        all_synthetic = reduce(DataFrame.unionAll, synthetic_data)
        result = df_maj.unionByName(df_min).unionByName(all_synthetic)
    else:
        result = working_df
    
    return result.drop("features_for_smote").withColumnRenamed("smote_features", "features")


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
