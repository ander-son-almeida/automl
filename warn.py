def test_spark_dataframe(df):
    """
    Testa se o objeto fornecido é um Spark DataFrame.
    Se não for, imprime uma mensagem de erro e interrompe a execução.
    
    Parâmetros:
    df: Objeto a ser testado (pode ser Pandas DataFrame, Spark DataFrame ou outros)
    """
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
        if not isinstance(df, SparkDataFrame):
            print("Erro: O DataFrame fornecido não é um Spark DataFrame.")
            print("Por favor, forneça um pyspark.sql.DataFrame.")
            exit(1)  # Encerra a execução com código de erro
    except ImportError:
        print("Erro: PySpark não está instalado neste ambiente.")
        print("Este código requer PySpark para funcionar corretamente.")
        exit(1)

# Exemplo de uso:
if __name__ == "__main__":
    # Teste com um DataFrame que não é do Spark (simulação)
    import pandas as pd
    pandas_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    
    print("Testando com Pandas DataFrame:")
    test_spark_dataframe(pandas_df)  # Isso deve parar a execução
    
    # O código abaixo não será executado se o teste falhar
    print("Esta mensagem não será exibida se o teste falhar.")


from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark_df = spark.createDataFrame([(1, 3), (2, 4)], ['col1', 'col2'])

print("Testando com Spark DataFrame:")
test_spark_dataframe(spark_df)  # Isso deve passar e continuar a execução
print("Teste passou, continuando a execução...")
