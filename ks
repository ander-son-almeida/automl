from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import numpy as np

# Inicializa a sessão do Spark
spark = SparkSession.builder \
    .appName("Cálculo KS") \
    .getOrCreate()

# Configuração para gerar dados de exemplo
np.random.seed(42)
n_amostras = 1000

# Simula probabilidades (0 a 1) - bons têm probabilidades menores em média
prob_bons = np.random.beta(2, 5, size=int(n_amostras*0.7))  # 70% bons
prob_maus = np.random.beta(5, 2, size=int(n_amostras*0.3))  # 30% maus

# Combina os dados
probabilidades = np.concatenate([prob_bons, prob_maus])
target = np.concatenate([np.zeros(len(prob_bons)), np.ones(len(prob_maus))])

# Embaralha os dados
indices = np.arange(n_amostras)
np.random.shuffle(indices)
probabilidades = probabilidades[indices]
target = target[indices]

# Cria DataFrame do PySpark
dados = list(zip(target.astype(int), probabilidades.astype(float)))
df = spark.createDataFrame(dados, ["target", "probabilidade"])

# Mostra as primeiras linhas
print("Primeiras linhas dos dados:")
df.show(5)

# Estatísticas básicas
print("\nEstatísticas descritivas:")
df.groupBy("target").agg(
    F.count("probabilidade").alias("contagem"),
    F.mean("probabilidade").alias("média"),
    F.stddev("probabilidade").alias("desvio_padrao"),
    F.min("probabilidade").alias("minimo"),
    F.max("probabilidade").alias("maximo")
).show()

# Função para calcular o KS no PySpark
def calcular_ks_spark(df, coluna_target="target", coluna_prob="probabilidade"):
    """
    Calcula a estatística KS em um DataFrame do PySpark
    
    Parâmetros:
    df - DataFrame do PySpark
    coluna_target - nome da coluna com os valores reais (0 e 1)
    coluna_prob - nome da coluna com as probabilidades previstas
    
    Retorna:
    valor_ks - valor da estatística KS
    df_ks_pandas - DataFrame pandas com as curvas acumuladas (apenas para plotagem)
    """
    # Calcula totais de bons e maus
    totais = df.agg(
        F.sum(F.when(F.col(coluna_target) == 0, 1).otherwise(0)).alias("total_bons"),
        F.sum(F.when(F.col(coluna_target) == 1, 1).otherwise(0)).alias("total_maus")
    ).collect()[0]
    
    total_bons = totais["total_bons"]
    total_maus = totais["total_maus"]
    
    # Ordena por probabilidade decrescente
    janela = Window.orderBy(F.desc(coluna_prob))
    
    # Calcula as distribuições acumuladas
    df_ks = df.withColumn("num_linha", F.row_number().over(janela)) \
        .withColumn("acum_bons", F.sum(F.when(F.col(coluna_target) == 0, 1).otherwise(0)).over(janela) / total_bons) \
        .withColumn("acum_maus", F.sum(F.when(F.col(coluna_target) == 1, 1).otherwise(0)).over(janela) / total_maus) \
        .withColumn("ks", F.col("acum_maus") - F.col("acum_bons"))
    
    # Encontra o máximo KS
    linha_ks = df_ks.orderBy(F.desc("ks")).first()
    valor_ks = linha_ks["ks"]
    
    # Converte apenas as colunas necessárias para pandas (curvas acumuladas)
    df_ks_pandas = df_ks.select(coluna_prob, "acum_bons", "acum_maus", "ks").orderBy(coluna_prob).toPandas()
    
    return valor_ks, df_ks_pandas

# Função para plotar o gráfico KS
def plotar_ks_spark(df_ks_pandas, valor_ks, coluna_prob="probabilidade"):
    """
    Plota as curvas acumuladas e destaca a estatística KS
    
    Parâmetros:
    df_ks_pandas - DataFrame pandas com as curvas acumuladas
    valor_ks - valor da estatística KS
    coluna_prob - nome da coluna com as probabilidades
    """
    # Encontra o ponto do KS
    ponto_ks = df_ks_pandas['ks'].idxmax()
    
    # Configurações do gráfico
    plt.figure(figsize=(10, 6))
    
    # Plota as curvas acumuladas
    plt.plot(df_ks_pandas[coluna_prob], df_ks_pandas['acum_bons'], label='Bons (0)', color='blue')
    plt.plot(df_ks_pandas[coluna_prob], df_ks_pandas['acum_maus'], label='Maus (1)', color='red')
    
    # Destaca o ponto do KS
    plt.axvline(x=df_ks_pandas.loc[ponto_ks, coluna_prob], color='gray', linestyle='--', 
                label=f'Ponto KS (Prob={df_ks_pandas.loc[ponto_ks, coluna_prob]:.2f})')
    
    # Adiciona linha da diferença KS
    plt.plot([df_ks_pandas.loc[ponto_ks, coluna_prob], df_ks_pandas.loc[ponto_ks, coluna_prob]],
             [df_ks_pandas.loc[ponto_ks, 'acum_bons'], df_ks_pandas.loc[ponto_ks, 'acum_maus']],
             color='green', linestyle='-', linewidth=2,
             label=f'Diferença KS = {valor_ks:.3f}')
    
    # Configurações adicionais
    plt.title(f'Curva KS - Estatística KS = {valor_ks:.3f}', fontsize=14)
    plt.xlabel('Probabilidade Cortada', fontsize=12)
    plt.ylabel('Proporção Acumulada', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    plt.show()

# ------------------------------------------
# CALCULA E PLOTA O KS
# ------------------------------------------
valor_ks, df_ks_pandas = calcular_ks_spark(df)
print(f"\nEstatística KS calculada: {valor_ks:.4f}")

# Plotagem do gráfico (aqui converte para pandas apenas as curvas acumuladas)
plotar_ks_spark(df_ks_pandas, valor_ks)

# Interpretação do KS
if valor_ks < 0.2:
    print("Interpretação: Poder discriminatório fraco")
elif 0.2 <= valor_ks < 0.3:
    print("Interpretação: Poder discriminatório razoável")
elif 0.3 <= valor_ks < 0.5:
    print("Interpretação: Poder discriminatório bom")
else:
    print("Interpretação: Poder discriminatório muito forte")

# Encerra a sessão do Spark
spark.stop()
