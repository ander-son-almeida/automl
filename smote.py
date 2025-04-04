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

class SmoteConfig:
    def __init__(self, k=5, multiplier=1, bucketLength=2.0, seed=42):
        self.k = k
        self.multiplier = multiplier
        self.bucketLength = bucketLength
        self.seed = seed

def smote_spark(df: DataFrame, target_col: str, num_cols: List[str], 
                cat_cols: Optional[List[str]] = None, config: Optional[SmoteConfig] = None) -> DataFrame:
    """
    Aplica SMOTE tratando valores NaN e evitando ambiguidade de colunas.
    
    Args:
        df: DataFrame Spark
        target_col: Nome da coluna alvo binária (1 = classe minoritária)
        num_cols: Lista de colunas numéricas
        cat_cols: Lista de colunas categóricas
        config: Configurações do SMOTE
        
    Returns:
        DataFrame balanceado
    """
    if cat_cols is None:
        cat_cols = []
    
    if config is None:
        config = SmoteConfig()
    
    # Verificação da coluna alvo
    distinct_values = [row[target_col] for row in df.select(target_col).distinct().collect()]
    if len(distinct_values) != 2:
        raise ValueError("A coluna alvo deve ter exatamente 2 valores distintos")
    
    # Pré-processamento com tratamento de NaN
    processed_df = _preprocess_with_nan_handling(df, num_cols, cat_cols, target_col)
    
    # Aplicar SMOTE com verificação de ambiguidade
    oversampled_df = _apply_smote_safe(processed_df, config)
    
    return oversampled_df

def _preprocess_with_nan_handling(df: DataFrame, num_cols: List[str], 
                                 cat_cols: List[str], target_col: str) -> DataFrame:
    """Pré-processamento com tratamento de valores faltantes."""
    # 1. Tratar valores NaN nas colunas numéricas
    imputer = Imputer(inputCols=num_cols, outputCols=num_cols, strategy="mean")
    
    # 2. Indexar colunas categóricas (exceto target)
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index").fit(df) 
        for col in cat_cols 
        if col != target_col
    ]
    
    # 3. Assembler para colunas numéricas (com nome único para features)
    assembler = VectorAssembler(inputCols=num_cols, outputCol="smote_features")
    
    # Pipeline com imputação, indexação e assembler
    pipeline = Pipeline(stages=[imputer] + indexers + [assembler])
    transformed_df = pipeline.fit(df).transform(df)
    
    # Selecionar e renomear colunas para evitar ambiguidade
    keep_cols = [
        col for col in transformed_df.columns 
        if col not in num_cols + cat_cols
    ] + ["smote_features"]
    
    return transformed_df.select(*keep_cols).withColumnRenamed(target_col, "label")

def _apply_smote_safe(df: DataFrame, config: SmoteConfig) -> DataFrame:
    """Aplica SMOTE com verificação de ambiguidade."""
    # Renomear a coluna de features para evitar conflitos
    working_df = df.withColumnRenamed("smote_features", "features")
    
    # Separar classes
    df_min = working_df.filter(F.col("label") == 1)
    df_maj = working_df.filter(F.col("label") == 0)
    
    if df_min.count() == 0:
        return df.drop("smote_features")
    
    # Configurar LSH com nome explícito da coluna
    brp = BucketedRandomProjectionLSH(
        inputCol="features",
        outputCol="hashes",
        seed=config.seed,
        bucketLength=config.bucketLength
    )
    
    # Ajustar modelo garantindo que só há uma coluna "features"
    model = brp.fit(df_min.select("features", *[c for c in df_min.columns if c != "features"]))
    
    # Similaridade com nomes explícitos
    self_join = model.approxSimilarityJoin(
        df_min.alias("minA"), 
        df_min.alias("minB"), 
        float("inf"), 
        distCol="EuclideanDistance"
    ).filter("EuclideanDistance > 0")
    
    # Selecionar k vizinhos mais próximos
    window = Window.partitionBy("minA").orderBy("EuclideanDistance")
    neighbors_df = self_join.withColumn("rn", F.row_number().over(window)) \
                          .filter(F.col("rn") <= config.k) \
                          .drop("rn")
    
    # Gerar dados sintéticos
    synthetic_data = []
    subtract_udf = F.udf(lambda arr: random.uniform(0, 1) * (arr[0] - arr[1]), VectorUDT())
    add_udf = F.udf(lambda arr: arr[0] + arr[1], VectorUDT())
    
    original_cols = [col for col in df_min.columns if col != "features"]
    
    for _ in range(config.multiplier):
        # Selecionar vizinho aleatório
        rand_df = neighbors_df.withColumn("rand", F.rand(seed=config.seed))
        max_rand_df = rand_df.withColumn("max_rand", F.max("rand").over(Window.partitionBy("minA")))
        selected_df = max_rand_df.filter(F.col("rand") == F.col("max_rand")).drop("rand", "max_rand")
        
        # Criar features sintéticas
        diff_df = selected_df.select("*", subtract_udf(F.array("minA.features", "minB.features")).alias("diff"))
        synth_df = diff_df.select("*", add_udf(F.array("minA.features", "diff")).alias("features"))
        
        # Copiar outras colunas
        for col_name in original_cols:
            choice = random.choice(["minA", "minB"])
            synth_df = synth_df.withColumn(col_name, F.col(f"{choice}.{col_name}"))
        
        synth_df = synth_df.drop("minA", "minB", "diff", "EuclideanDistance")
        synthetic_data.append(synth_df)
    
    # Combinar resultados
    if synthetic_data:
        all_synthetic = reduce(DataFrame.unionAll, synthetic_data)
        result = df_maj.unionByName(df_min).unionByName(all_synthetic)
    else:
        result = working_df
    
    return result.drop("features").withColumnRenamed("smote_features", "features")


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
