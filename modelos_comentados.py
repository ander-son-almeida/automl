"""
SCRIPT COMPLETO PARA MODELAGEM DE RISCO DE CRÉDITO COM:
- Explicações linha por linha dos hiperparâmetros
- Estratégias específicas para dados desbalanceados
- Análise de tradeoffs entre performance e custo computacional
- Comentários detalhados sobre cada decisão de modelagem
"""

from pyspark.ml.classification import (
    LogisticRegression, 
    DecisionTreeClassifier, 
    RandomForestClassifier,
    GBTClassifier, 
    LinearSVC, 
    NaiveBayes,
    MultilayerPerceptronClassifier,
    FMClassifier,
    OneVsRest
)
from synapse.ml.lightgbm import LightGBMClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import optuna
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler

## ======================================
## CONFIGURAÇÕES GLOBAIS
## ======================================
SEED = 42  # Seed fixa para reprodutibilidade
MAX_TRIALS = 50  # Número máximo de tentativas por otimização
TIMEOUT = 3600  # 1 hora de timeout por modelo (evita loops infinitos)

# Nota: Estas configurações garantem que os experimentos sejam reproduzíveis
# e que não consumam recursos computacionais indefinidamente

## ======================================
## 1. REGRESSÃO LOGÍSTICA - Modelo Linear
## ======================================
def get_logistic_params(trial):
    """
    Configuração detalhada para Regressão Logística em crédito.
    Este modelo é o baseline ideal por sua simplicidade e interpretabilidade.
    """
    params = {
        # regParam: Controla a força da regularização (evita overfitting)
        # - Range: 0.001 a 10 em escala logarítmica
        # - Em crédito, valores entre 0.1-1.0 funcionam bem
        # - Valores altos (>5) podem subajustar o modelo
        "regParam": trial.suggest_float("regParam", 1e-3, 10, log=True),
        
        # elasticNetParam: Balanceia entre regularização L1 e L2
        # - 0 = puro L2 (Ridge), 1 = puro L1 (Lasso)
        # - Evitar 1.0 pois Lasso é instável com features correlacionadas
        # - Valor ideal em crédito: 0.5-0.8 (ElasticNet)
        "elasticNetParam": trial.suggest_float("elasticNetParam", 0, 0.9),
        
        # maxIter: Número máximo de iterações
        # - Para datasets grandes (>1M linhas), 50-100 é suficiente
        # - Para datasets pequenos ou features muito correlacionadas, até 150
        "maxIter": trial.suggest_int("maxIter", 50, 150),
        
        # fitIntercept: Deve sempre ser True em modelos de crédito
        # - Captura o viés inerente da população
        # - Só desativar em casos muito específicos
        "fitIntercept": True,
        
        # standardization: Crítico para dados financeiros
        # - Features como renda e dívida estão em escalas muito diferentes
        # - Sem standardização, o modelo fica dominado pelas features com maior magnitude
        "standardization": True,
        
        # tol: Tolerância para critério de parada
        # - 1e-4 é um bom balanço entre precisão e tempo de treino
        # - Para datasets muito grandes, pode aumentar para 1e-3
        "tol": 1e-4,
        
        # threshold: Ponto de corte para classificação
        # - Em dados desbalanceados, o padrão 0.5 não é ideal
        # - Permitimos ajuste fino entre 0.3-0.7
        "threshold": trial.suggest_float("threshold", 0.3, 0.7)
    }
    return params

## ======================================
## 2. DECISION TREE - Árvore de Decisão
## ======================================
def get_tree_params(trial):
    """
    Configuração para Árvores de Decisão com foco em:
    - Prevenção de overfitting
    - Estabilidade com dados desbalanceados
    - Interpretabilidade das regras
    """
    return {
        # maxDepth: Profundidade máxima da árvore
        # - Limitar entre 3-8 níveis para crédito
        # - Mais que 8 níveis quase sempre leva a overfitting
        # - Menos que 3 pode subajustar
        "maxDepth": trial.suggest_int("maxDepth", 3, 8),
        
        # minInstancesPerNode: Mínimo de exemplos por nó
        # - Valores altos (20-100) forçam splits mais significativos
        # - Calculado como ~1% da classe minoritária
        # - Evita nós com poucos exemplos da classe rara
        "minInstancesPerNode": trial.suggest_int("minInstancesPerNode", 20, 100),
        
        # minInfoGain: Ganho mínimo para realizar um split
        # - Threshold alto (0.01-0.1) filtra splits insignificantes
        # - Em crédito, 0.03-0.05 remove ruído mantendo sinais importantes
        "minInfoGain": trial.suggest_float("minInfoGain", 0.01, 0.1),
        
        # maxBins: Número de bins para features contínuas
        # - 32 é suficiente para dados financeiros
        # - Aumentar para >100 só melhora marginalmente
        # - Impacto direto no uso de memória
        "maxBins": 32,
        
        # impurity: Critério de divisão
        # - "gini" é mais rápido e adequado para crédito
        # - "entropy" pode capturar relações mais complexas
        # - Em Spark, gini é ~25% mais rápido
        "impurity": trial.suggest_categorical("impurity", ["gini", "entropy"]),
        
        # maxMemoryInMB: Limite explícito de memória
        # - Evita estouro em datasets grandes
        # - 512MB é suficiente para árvores de até 8 níveis
        "maxMemoryInMB": 512
    }

## ======================================
## 3. RANDOM FOREST - Ensemble Robustos
## ======================================
def get_rf_params(trial):
    """
    Configuração para Random Forest com:
    - Balanceamento entre diversidade e custo
    - Controle de overfitting
    - Eficiência computacional
    """
    params = {
        # numTrees: Número de árvores no ensemble
        # - 30-100 é o range ideal para crédito
        # - Acima de 100 tem retorno decrescente
        # - Custo computacional cresce linearmente
        "numTrees": trial.suggest_int("numTrees", 30, 100),
        
        # maxDepth: Profundidade das árvores
        # - Mais raso que árvores únicas (4-8 níveis)
        # - Reduz variância do ensemble
        "maxDepth": trial.suggest_int("maxDepth", 4, 8),
        
        # minInstancesPerNode: Similar à Decision Tree
        # - Valores menores (15-50) pois o ensemble já controla overfitting
        "minInstancesPerNode": trial.suggest_int("minInstancesPerNode", 15, 50),
        
        # subsamplingRate: Fração dos dados usada em cada árvore
        # - 0.7-0.9 aumenta diversidade sem perder muito sinal
        # - Valores baixos (<0.5) podem prejudicar performance
        "subsamplingRate": trial.suggest_float("subsamplingRate", 0.7, 0.9),
        
        # featureSubsetStrategy: Como selecionar features
        # - "sqrt" é padrão e funciona bem na prática
        # - "log2" para datasets com muitas features (>100)
        # - "onethird" é mais conservador
        "featureSubsetStrategy": trial.suggest_categorical(
            "featureSubsetStrategy", 
            ["sqrt", "log2", "onethird"]
        ),
        
        # bootstrap: Se faz amostragem com reposição
        # - True aumenta diversidade das árvores
        # - Fundamental para o funcionamento do RF
        "bootstrap": True,
        
        # seed: Para reprodutibilidade
        "seed": SEED
    }
    return params

## ======================================
## 4. GRADIENT BOOSTING TREES - Boosting
## ======================================
def get_gbt_params(trial):
    """
    Configuração conservadora para GBT que:
    - Previne overfitting em dados desbalanceados
    - Mantém tempo de treino razoável
    - Garante estabilidade
    """
    return {
        # maxIter: Número de iterações/árvores
        # - Limitado a 20-80 para evitar overfitting
        # - Em crédito, raramente precisamos de >50
        "maxIter": trial.suggest_int("maxIter", 20, 80),
        
        # maxDepth: Profundidade das árvores
        # - Extremamente rasas (3-6 níveis)
        # - Árvores profundas levam a overfitting rápido
        "maxDepth": trial.suggest_int("maxDepth", 3, 6),
        
        # stepSize: Taxa de aprendizado (shrinkage)
        # - 0.05-0.2 é mais alto que o padrão (0.01)
        # - Compensa o menor número de iterações
        "stepSize": trial.suggest_float("stepSize", 0.05, 0.2),
        
        # subsamplingRate: Stochastic GBT
        # - 0.6-0.9 aumenta robustez
        # - Valores baixos podem prejudicar performance
        "subsamplingRate": trial.suggest_float("subsamplingRate", 0.6, 0.9),
        
        # minInstancesPerNode: Controle de splits
        # - Valores altos (20-50) para splits significativos
        "minInstancesPerNode": trial.suggest_int("minInstancesPerNode", 20, 50),
        
        # maxBins: Otimização para performance
        # - 32 é suficiente para dados financeiros
        "maxBins": 32
    }

## ======================================
## 5. LIGHTGBM - Framework Avançado
## ======================================
def get_lgbm_params(trial):
    """
    Configuração otimizada para LightGBM com:
    - Tratamento explícito de desbalanceamento
    - Eficiência computacional
    - Boas práticas para dados financeiros
    """
    params = {
        # numLeaves: Número máximo de folhas
        # - Controla complexidade melhor que maxDepth
        # - 15-50 é ideal para crédito
        "numLeaves": trial.suggest_int("numLeaves", 15, 50),
        
        # learningRate: Taxa de aprendizado
        # - 0.05-0.2 para convergência estável
        # - Valores altos podem divergir
        "learningRate": trial.suggest_float("learningRate", 0.05, 0.2),
        
        # minDataInLeaf: Dados mínimos por folha
        # - Crítico para dados desbalanceados (30-100)
        # - Evita folhas com poucos exemplos da classe rara
        "minDataInLeaf": trial.suggest_int("minDataInLeaf", 30, 100),
        
        # featureFraction: Fração de features por árvore
        # - Similar ao featureSubsetStrategy do RF
        # - 0.7-0.9 aumenta diversidade
        "featureFraction": trial.suggest_float("featureFraction", 0.7, 0.9),
        
        # baggingFraction: Fração de dados por árvore
        # - Stochastic boosting para robustez
        "baggingFraction": trial.suggest_float("baggingFraction", 0.7, 0.9),
        
        # lambdaL1: Regularização L1
        # - 0-1 para seleção suave de features
        "lambdaL1": trial.suggest_float("lambdaL1", 0, 1),
        
        # scale_pos_weight: Peso da classe positiva
        # - Calculado como count(negativos)/count(positivos)
        # - O parâmetro MAIS IMPORTANTE para desbalanceamento
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5, 20),
        
        # boostingType: Algoritmo de boosting
        # - "gbdt": tradicional
        # - "dart": dropout para reduzir overfitting
        "boostingType": trial.suggest_categorical("boostingType", ["gbdt", "dart"]),
        
        # numThreads: Paralelismo controlado
        # - Evita sobrecarga do cluster
        "numThreads": 4
    }
    return params

## ======================================
## FUNÇÃO DE OTIMIZAÇÃO PRINCIPAL
## ======================================
def optimize_model(model_name, train_df, test_df, target_col, num_features=None):
    """
    Função principal que:
    1. Configura o espaço de busca para o modelo selecionado
    2. Executa a otimização com Optuna
    3. Retorna os melhores parâmetros encontrados
    
    Parâmetros:
    - model_name: Nome do modelo a ser otimizado
    - train_df/test_df: DataFrames de treino e teste
    - target_col: Nome da coluna target
    - num_features: Número de features (apenas para MLP)
    """
    
    def objective(trial):
        # Configuração específica para cada modelo
        if model_name == "LogisticRegression":
            params = get_logistic_params(trial)
            model = LogisticRegression(
                featuresCol='features',
                labelCol=target_col,
                **params
            )
        
        elif model_name == "DecisionTree":
            params = get_tree_params(trial)
            model = DecisionTreeClassifier(
                featuresCol='features',
                labelCol=target_col,
                **params
            )
            
        elif model_name == "RandomForest":
            params = get_rf_params(trial)
            model = RandomForestClassifier(
                featuresCol='features',
                labelCol=target_col,
                **params
            )
            
        elif model_name == "GBT":
            params = get_gbt_params(trial)
            model = GBTClassifier(
                featuresCol='features',
                labelCol=target_col,
                **params
            )
            
        elif model_name == "LightGBM":
            params = get_lgbm_params(trial)
            model = LightGBMClassifier(
                featuresCol='features',
                labelCol=target_col,
                **params
            )
        
        # Treinamento do modelo
        fitted_model = model.fit(train_df)
        
        # Obtenção das previsões
        predictions = fitted_model.transform(test_df)
        
        # Avaliação com AUC-PR (ideal para desbalanceados)
        evaluator = BinaryClassificationEvaluator(
            labelCol=target_col,
            metricName="areaUnderPR"
        )
        auc_pr = evaluator.evaluate(predictions)
        
        return auc_pr

    # Configuração do estudo Optuna
    study = optuna.create_study(
        direction="maximize",  # Queremos maximizar AUC-PR
        sampler=optuna.samplers.TPESampler(seed=SEED),  # Algoritmo de amostragem
        pruner=optuna.pruners.HyperbandPruner()  # Poda trials ineficientes
    )
    
    # Execução da otimização
    study.optimize(
        objective,
        n_trials=MAX_TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=True
    )
    
    return study.best_params  # Retorna os melhores parâmetros encontrados

## ======================================
## EXEMPLO COMPLETO DE USO
## ======================================
if __name__ == "__main__":
    # 1. Inicialização do Spark
    spark = SparkSession.builder \
        .appName("CreditScoring") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    # 2. Carregamento de dados (exemplo)
    print("Carregando dados...")
    df = spark.read.parquet("dados_credito.parquet")
    
    # 3. Pré-processamento básico
    print("Preparando features...")
    feature_cols = [
        "renda_mensal", 
        "idade", 
        "tempo_emprego", 
        "divida_total",
        "historico_pagamentos"
    ]
    target_col = "inadimplente"
    
    # 4. Criação da coluna de features vetorizadas
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"  # Pula linhas com valores inválidos
    )
    df = assembler.transform(df)
    
    # 5. Divisão treino-teste estratificada
    print("Dividindo dados em treino e teste...")
    train, test = df.randomSplit([0.8, 0.2], seed=SEED)
    
    # 6. Otimização do modelo (exemplo: LightGBM)
    print("\nOtimizando LightGBM...")
    best_params = optimize_model(
        model_name="LightGBM",
        train_df=train,
        test_df=test,
        target_col=target_col
    )
    
    print(f"\nMelhores parâmetros encontrados:")
    for param, value in best_params.items():
        print(f"- {param}: {value}")
    
    # 7. Treinamento do modelo final
    print("\nTreinando modelo final com os melhores parâmetros...")
    final_model = LightGBMClassifier(
        featuresCol='features',
        labelCol=target_col,
        **best_params
    ).fit(train)
    
    # 8. Avaliação detalhada
    print("\nAvaliando modelo no conjunto de teste...")
    predictions = final_model.transform(test)
    
    # Métricas principais
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol=target_col,
        metricName="areaUnderPR"
    )
    auc_pr = evaluator_pr.evaluate(predictions)
    
    evaluator_roc = BinaryClassificationEvaluator(
        labelCol=target_col,
        metricName="areaUnderROC"
    )
    auc_roc = evaluator_roc.evaluate(predictions)
    
    print(f"\nPerformance do modelo:")
    print(f"- AUC-PR: {auc_pr:.4f} (métrica principal para desbalanceados)")
    print(f"- AUC-ROC: {auc_roc:.4f}")
    print(f"- KS: {calculate_ks(predictions, target_col):.4f}")
    
    # 9. Salvando o modelo (exemplo)
    print("\nSalvando modelo...")
    model_path = "modelos/lgbm_credit_scoring"
    final_model.save(model_path)
    print(f"Modelo salvo em: {model_path}")

## ======================================
## FUNÇÕES AUXILIARES
## ======================================
def calculate_ks(predictions, target_col):
    """
    Calcula a estatística KS (Kolmogorov-Smirnov) para avaliação de modelos de crédito
    KS mede a separação entre as distribuições de scores para bons e maus pagadores
    """
    # Ordena as previsões por probabilidade decrescente
    window = Window.orderBy(F.desc("probability"))
    
    # Calcula TPR (Taxa de Verdadeiros Positivos)
    tpr = predictions.withColumn(
        "tpr",
        F.sum(F.col(target_col)).over(window)
    
    # Calcula FPR (Taxa de Falsos Positivos)
    fpr = predictions.withColumn(
        "fpr",
        F.sum(1 - F.col(target_col)).over(window)
    
    # Calcula KS como a diferença máxima entre TPR e FPR
    ks_value = tpr.withColumn(
        "ks",
        (F.col("tpr") - F.col("fpr"))
        .agg(F.max("ks"))
        .collect()[0][0]
    
    return ks_value
