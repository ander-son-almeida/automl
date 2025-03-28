from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, isnull, when
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Imputer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType, IntegerType, FloatType, LongType
import numpy as np

def imputar_mice_spark(df_spark, coluna_alvo=None, max_iteracoes=10, tolerancia=1e-4, limite_faltantes=0.3):
    """
    Aplica imputação MICE em um DataFrame Spark e retorna um novo DataFrame imputado
    
    Parâmetros:
    -----------
    df_spark : DataFrame Spark
        DataFrame de entrada com valores faltantes
    coluna_alvo : str (opcional)
        Nome da coluna target (se existir)
    max_iteracoes : int
        Número máximo de iterações do MICE
    tolerancia : float
        Critério de convergência (percentual de registros alterados)
    limite_faltantes : float
        Percentual máximo de valores faltantes para manter a coluna (0-1)
    
    Retorna:
    --------
    DataFrame Spark
        DataFrame com valores imputados mantendo a mesma estrutura original
    """
    
    # 1. Verificação inicial
    if not isinstance(df_spark, type(SparkSession.createDataFrame([(1,)])):
        raise ValueError("O parâmetro df_spark deve ser um DataFrame do PySpark")
    
    # 2. Identificar colunas com valores faltantes
    print("\n🔍 Analisando valores faltantes...")
    colunas_faltantes = []
    for nome_coluna in df_spark.columns:
        qtd_faltantes = df_spark.filter(isnull(col(nome_coluna)) | isnan(col(nome_coluna))).count()
        percentual = qtd_faltantes / df_spark.count()
        
        if percentual > limite_faltantes:
            print(f"⚠️ Removendo coluna '{nome_coluna}' ({percentual:.1%} faltantes)")
            df_spark = df_spark.drop(nome_coluna)
        elif qtd_faltantes > 0:
            colunas_faltantes.append(nome_coluna)
            print(f"✅ '{nome_coluna}': {qtd_faltantes} faltantes ({percentual:.1%})")
    
    if not colunas_faltantes:
        print("Nenhuma coluna com valores faltantes dentro do limite encontrada")
        return df_spark
    
    # 3. Classificar colunas por tipo
    print("\n📊 Classificando colunas...")
    colunas_numericas = []
    colunas_categoricas = []
    
    for campo in df_spark.schema.fields:
        if campo.name == coluna_alvo:
            continue
            
        if isinstance(campo.dataType, (DoubleType, IntegerType, FloatType, LongType)):
            colunas_numericas.append(campo.name)
        else:
            colunas_categoricas.append(campo.name)
    
    print(f"Númericas: {colunas_numericas}")
    print(f"Categóricas: {colunas_categoricas}")
    
    # 4. Pré-processamento
    print("\n⚙️ Preparando dados para MICE...")
    
    # Pipeline para variáveis categóricas
    etapas_preprocessamento = []
    
    if colunas_categoricas:
        indexadores = [
            StringIndexer(
                inputCol=coluna,
                outputCol=f"{coluna}_idx",
                handleInvalid="keep"
            ) for coluna in colunas_categoricas
        ]
        
        codificadores = OneHotEncoder(
            inputCols=[f"{coluna}_idx" for coluna in colunas_categoricas],
            outputCols=[f"{coluna}_enc" for coluna in colunas_categoricas]
        )
        
        etapas_preprocessamento.extend(indexadores)
        etapas_preprocessamento.append(codificadores)
    
    # Adicionar todas as colunas ao VectorAssembler
    features = colunas_numericas + [f"{col}_enc" for col in colunas_categoricas]
    if coluna_alvo:
        features.append(coluna_alvo)
    
    # 5. Imputação inicial (mediana para numéricas)
    print("\n🔄 Realizando imputação inicial...")
    imputador_inicial = Imputer(
        strategy="median",
        inputCols=colunas_numericas,
        outputCols=colunas_numericas
    )
    etapas_preprocessamento.append(imputador_inicial)
    
    pipeline_preprocessamento = Pipeline(stages=etapas_preprocessamento)
    modelo_preprocessamento = pipeline_preprocessamento.fit(df_spark)
    df_transformado = modelo_preprocessamento.transform(df_spark)
    
    # 6. Algoritmo MICE
    print("\n🚀 Executando algoritmo MICE...")
    for iteracao in range(max_iteracoes):
        print(f"\n📌 Iteração {iteracao + 1}/{max_iteracoes}")
        df_anterior = df_transformado.select(colunas_faltantes).cache()
        
        for coluna_imputar in colunas_faltantes:
            print(f"  ➡️ Processando coluna: {coluna_imputar}")
            
            # Features temporárias (todas exceto a atual)
            features_temporarias = [f for f in features if f != coluna_imputar]
            
            # Configurar modelo apropriado
            montador = VectorAssembler(
                inputCols=features_temporarias,
                outputCol="features_temp"
            )
            
            if coluna_imputar in colunas_numericas:
                modelo = LinearRegression(
                    featuresCol="features_temp",
                    labelCol=coluna_imputar
                )
            else:
                modelo = LogisticRegression(
                    featuresCol="features_temp",
                    labelCol=coluna_imputar,
                    family="multinomial"
                )
            
            pipeline = Pipeline(stages=[montador, modelo])
            
            # Treinar com dados completos
            dados_completos = df_transformado.filter(
                ~isnull(col(coluna_imputar)) & ~isnan(col(coluna_imputar))
            
            if dados_completos.count() > 0:
                modelo_treinado = pipeline.fit(dados_completos)
                
                # Prever valores faltantes
                dados_faltantes = df_transformado.filter(
                    isnull(col(coluna_imputar)) | isnan(col(coluna_imputar)))
                
                if dados_faltantes.count() > 0:
                    previsoes = modelo_treinado.transform(dados_faltantes)
                    
                    # Atualizar DataFrame
                    df_transformado = df_transformado.join(
                        previsoes.select(coluna_imputar, "prediction"),
                        on=coluna_imputar,
                        how="left"
                    ).withColumn(
                        coluna_imputar,
                        when(isnull(col(coluna_imputar)), col("prediction")
                    ).otherwise(col(coluna_imputar))
                    ).drop("prediction")
        
        # Verificar convergência
        df_atual = df_transformado.select(colunas_faltantes).cache()
        diferenca = df_anterior.join(df_atual, on=colunas_faltantes, how="left_anti").count()
        df_anterior.unpersist()
        
        print(f"  🔄 Registros alterados: {diferenca}")
        if diferenca < tolerancia * df_transformado.count():
            print("\n✅ Convergência alcançada!")
            break
    
    # 7. Pós-processamento (reverter transformações se necessário)
    print("\n✨ Finalizando processamento...")
    
    # Selecionar apenas colunas originais
    colunas_originais = df_spark.columns
    df_final = df_transformado.select(colunas_originais)
    
    print("\n🎉 Imputação concluída com sucesso!")
    return df_final

# Exemplo de uso:
if __name__ == "__main__":
    # 1. Iniciar sessão Spark
    spark = SparkSession.builder \
        .appName("ImputacaoMICE") \
        .getOrCreate()
    
    # 2. Carregar dados de exemplo (substitua pelo seu DataFrame)
    dados_exemplo = [
        (25.0, 5000.0, "SP", "bom"),
        (None, 6000.0, "RJ", None),
        (30.0, None, None, "bom"),
        (35.0, 8000.0, "SP", "mau"),
        (None, 3500.0, "MG", "bom"),
        (28.0, None, "RJ", "mau"),
        (40.0, 9000.0, None, "bom")
    ]
    
    colunas = ["idade", "renda", "cidade", "status_pagamento"]
    df = spark.createDataFrame(dados_exemplo, colunas)
    
    # 3. Aplicar imputação
    df_imputado = imputar_mice_spark(
        df_spark=df,
        coluna_alvo="status_pagamento",
        max_iteracoes=5,
        tolerancia=0.01
    )
    
    # 4. Mostrar resultados
    print("\nResultado final:")
    df_imputado.show()
    
    # 5. Salvar ou continuar processamento
    # df_imputado.write.parquet("caminho/para/salvar")


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, isnull, when
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Imputer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType, IntegerType, FloatType, LongType
import numpy as np

def imputar_mice_spark(df_spark, coluna_alvo=None, max_iteracoes=10, tolerancia=1e-4, limite_faltantes=0.3):
    """
    Aplica imputação MICE em um DataFrame Spark e retorna um novo DataFrame imputado
    
    Parâmetros:
    -----------
    df_spark : DataFrame Spark
        DataFrame de entrada com valores faltantes
    coluna_alvo : str (opcional)
        Nome da coluna target (se existir)
    max_iteracoes : int
        Número máximo de iterações do MICE
    tolerancia : float
        Critério de convergência (percentual de registros alterados)
    limite_faltantes : float
        Percentual máximo de valores faltantes para manter a coluna (0-1)
    
    Retorna:
    --------
    DataFrame Spark
        DataFrame com valores imputados mantendo a mesma estrutura original
    """
    
    # 1. Verificação inicial
    if not isinstance(df_spark, type(SparkSession.createDataFrame([(1,)])):
        raise ValueError("O parâmetro df_spark deve ser um DataFrame do PySpark")
    
    # 2. Identificar colunas com valores faltantes
    print("\n🔍 Analisando valores faltantes...")
    colunas_faltantes = []
    for nome_coluna in df_spark.columns:
        qtd_faltantes = df_spark.filter(isnull(col(nome_coluna)) | isnan(col(nome_coluna))).count()
        percentual = qtd_faltantes / df_spark.count()
        
        if percentual > limite_faltantes:
            print(f"⚠️ Removendo coluna '{nome_coluna}' ({percentual:.1%} faltantes)")
            df_spark = df_spark.drop(nome_coluna)
        elif qtd_faltantes > 0:
            colunas_faltantes.append(nome_coluna)
            print(f"✅ '{nome_coluna}': {qtd_faltantes} faltantes ({percentual:.1%})")
    
    if not colunas_faltantes:
        print("Nenhuma coluna com valores faltantes dentro do limite encontrada")
        return df_spark
    
    # 3. Classificar colunas por tipo
    print("\n📊 Classificando colunas...")
    colunas_numericas = []
    colunas_categoricas = []
    
    for campo in df_spark.schema.fields:
        if campo.name == coluna_alvo:
            continue
            
        if isinstance(campo.dataType, (DoubleType, IntegerType, FloatType, LongType)):
            colunas_numericas.append(campo.name)
        else:
            colunas_categoricas.append(campo.name)
    
    print(f"Númericas: {colunas_numericas}")
    print(f"Categóricas: {colunas_categoricas}")
    
    # 4. Pré-processamento
    print("\n⚙️ Preparando dados para MICE...")
    
    # Pipeline para variáveis categóricas
    etapas_preprocessamento = []
    
    if colunas_categoricas:
        indexadores = [
            StringIndexer(
                inputCol=coluna,
                outputCol=f"{coluna}_idx",
                handleInvalid="keep"
            ) for coluna in colunas_categoricas
        ]
        
        codificadores = OneHotEncoder(
            inputCols=[f"{coluna}_idx" for coluna in colunas_categoricas],
            outputCols=[f"{coluna}_enc" for coluna in colunas_categoricas]
        )
        
        etapas_preprocessamento.extend(indexadores)
        etapas_preprocessamento.append(codificadores)
    
    # Adicionar todas as colunas ao VectorAssembler
    features = colunas_numericas + [f"{col}_enc" for col in colunas_categoricas]
    if coluna_alvo:
        features.append(coluna_alvo)
    
    # 5. Imputação inicial (mediana para numéricas)
    print("\n🔄 Realizando imputação inicial...")
    imputador_inicial = Imputer(
        strategy="median",
        inputCols=colunas_numericas,
        outputCols=colunas_numericas
    )
    etapas_preprocessamento.append(imputador_inicial)
    
    pipeline_preprocessamento = Pipeline(stages=etapas_preprocessamento)
    modelo_preprocessamento = pipeline_preprocessamento.fit(df_spark)
    df_transformado = modelo_preprocessamento.transform(df_spark)
    
    # 6. Algoritmo MICE
    print("\n🚀 Executando algoritmo MICE...")
    for iteracao in range(max_iteracoes):
        print(f"\n📌 Iteração {iteracao + 1}/{max_iteracoes}")
        df_anterior = df_transformado.select(colunas_faltantes).cache()
        
        for coluna_imputar in colunas_faltantes:
            print(f"  ➡️ Processando coluna: {coluna_imputar}")
            
            # Features temporárias (todas exceto a atual)
            features_temporarias = [f for f in features if f != coluna_imputar]
            
            # Configurar modelo apropriado
            montador = VectorAssembler(
                inputCols=features_temporarias,
                outputCol="features_temp"
            )
            
            if coluna_imputar in colunas_numericas:
                modelo = LinearRegression(
                    featuresCol="features_temp",
                    labelCol=coluna_imputar
                )
            else:
                modelo = LogisticRegression(
                    featuresCol="features_temp",
                    labelCol=coluna_imputar,
                    family="multinomial"
                )
            
            pipeline = Pipeline(stages=[montador, modelo])
            
            # Treinar com dados completos
            dados_completos = df_transformado.filter(
                ~isnull(col(coluna_imputar)) & ~isnan(col(coluna_imputar))
            
            if dados_completos.count() > 0:
                modelo_treinado = pipeline.fit(dados_completos)
                
                # Prever valores faltantes
                dados_faltantes = df_transformado.filter(
                    isnull(col(coluna_imputar)) | isnan(col(coluna_imputar)))
                
                if dados_faltantes.count() > 0:
                    previsoes = modelo_treinado.transform(dados_faltantes)
                    
                    # Atualizar DataFrame
                    df_transformado = df_transformado.join(
                        previsoes.select(coluna_imputar, "prediction"),
                        on=coluna_imputar,
                        how="left"
                    ).withColumn(
                        coluna_imputar,
                        when(isnull(col(coluna_imputar)), col("prediction")
                    ).otherwise(col(coluna_imputar))
                    ).drop("prediction")
        
        # Verificar convergência
        df_atual = df_transformado.select(colunas_faltantes).cache()
        diferenca = df_anterior.join(df_atual, on=colunas_faltantes, how="left_anti").count()
        df_anterior.unpersist()
        
        print(f"  🔄 Registros alterados: {diferenca}")
        if diferenca < tolerancia * df_transformado.count():
            print("\n✅ Convergência alcançada!")
            break
    
    # 7. Pós-processamento (reverter transformações se necessário)
    print("\n✨ Finalizando processamento...")
    
    # Selecionar apenas colunas originais
    colunas_originais = df_spark.columns
    df_final = df_transformado.select(colunas_originais)
    
    print("\n🎉 Imputação concluída com sucesso!")
    return df_final

# Exemplo de uso:
if __name__ == "__main__":
    # 1. Iniciar sessão Spark
    spark = SparkSession.builder \
        .appName("ImputacaoMICE") \
        .getOrCreate()
    
    # 2. Carregar dados de exemplo (substitua pelo seu DataFrame)
    dados_exemplo = [
        (25.0, 5000.0, "SP", "bom"),
        (None, 6000.0, "RJ", None),
        (30.0, None, None, "bom"),
        (35.0, 8000.0, "SP", "mau"),
        (None, 3500.0, "MG", "bom"),
        (28.0, None, "RJ", "mau"),
        (40.0, 9000.0, None, "bom")
    ]
    
    colunas = ["idade", "renda", "cidade", "status_pagamento"]
    df = spark.createDataFrame(dados_exemplo, colunas)
    
    # 3. Aplicar imputação
    df_imputado = imputar_mice_spark(
        df_spark=df,
        coluna_alvo="status_pagamento",
        max_iteracoes=5,
        tolerancia=0.01
    )
    
    # 4. Mostrar resultados
    print("\nResultado final:")
    df_imputado.show()
    
    # 5. Salvar ou continuar processamento
    # df_imputado.write.parquet("caminho/para/salvar")
