1. Logistic Regression - O Básico Robustos
python
Copy
{
    "regParam": trial.suggest_float("regParam", 1e-3, 10, log=True),
    # Regularização: Começa suave (1e-3) até forte (10) em escala logarítmica
    # Em crédito, valores médios (0.1-1) costumam funcionar bem
    # Custo: Quase nenhum impacto no tempo de treino
    
    "elasticNetParam": trial.suggest_float("elasticNetParam", 0, 0.9),
    # Mix L1/L2: 0 = Ridge (L2), 1 = Lasso (L1)
    # Evitamos 1.0 puro pois Lasso pode ser instável com features correlacionadas
    # Em crédito, 0.5-0.8 (elastic net) ajuda na seleção de features
    
    "maxIter": trial.suggest_int("maxIter", 50, 150),
    # Número suficiente para convergência, mesmo com features altamente correlacionadas
    # Valores altos (>200) raramente trazem benefícios em crédito
    
    "fitIntercept": True,
    # Sempre True em modelos de crédito para capturar viés base
    
    "standardization": True,
    # Crucial quando temos features em escalas diferentes (ex: renda vs idade)
    
    "tol": 1e-4,
    # Tolerância de convergência: balanceia precisão vs tempo
    # 1e-4 é um bom equilíbrio para dados financeiros
}
Por que funciona para desbalanceamento?

A regularização (regParam) previne overfitting à classe majoritária

ElasticNet ajuda a selecionar as features mais discriminativas para a classe minoritária

Standardization garante que todas features contribuam igualmente

Custo Computacional:

Treino extremamente rápido (<1s mesmo para milhões de linhas)

Pode rodar diretamente no Spark sem overhead significativo

2. Decision Tree - A Árvore Conservadora
python
Copy
{
    "maxDepth": trial.suggest_int("maxDepth", 3, 8),
    # Profundidade limitada: árvores muito profundas overfittam a classe majoritária
    # Para crédito, 5-6 níveis costumam ser suficientes
    
    "minInstancesPerNode": trial.suggest_int("minInstancesPerNode", 20, 100),
    # Número mínimo de exemplos por nó: valores altos forçam a considerar grupos da classe minoritária
    # Típico: 1% do tamanho do dataset para a classe minoritária
    
    "minInfoGain": trial.suggest_float("minInfoGain", 0.01, 0.1),
    # Ganho mínimo para fazer split: threshold alto previne splits insignificantes
    # Em crédito, 0.03-0.05 costuma filtrar ruído bem
    
    "maxBins": 32,
    # Número de bins para features contínuas: valor baixo reduz custo com ganho mínimo de precisão
    # Para dados financeiros, 32 bins capturam bem a distribuição
    
    "impurity": "gini",
    # Gini é 25% mais rápido que entropy com resultados similares em crédito
    
    "maxMemoryInMB": 512,
    # Limite explícito para evitar estouro de memória com datasets grandes
}
Estratégia para Desbalanceamento:

minInstancesPerNode alto força a árvore a agrupar exemplos da classe minoritária

maxDepth limitada cria regras mais genéricas que generalizam melhor

minInfoGain alto filtra splits que só beneficiam a classe majoritária

Custo Computacional:

Treino rápido (segundos para datasets médios)

Custo de memória linear com o número de exemplos

Pode ser paralelizado no Spark eficientemente

3. Random Forest - O Ensemble Seguro
python
Copy
{
    "numTrees": trial.suggest_int("numTrees", 30, 100),
    # Número de árvores: tradeoff clássico entre custo e performance
    # Em crédito, 50-80 árvores costumam saturar os ganhos
    
    "maxDepth": trial.suggest_int("maxDepth", 4, 8),
    # Mais raso que árvore única para reduzir variância
    
    "minInstancesPerNode": trial.suggest_int("minInstancesPerNode", 15, 50),
    # Valores menores que Decision Tree pois o ensemble já controla overfitting
    
    "subsamplingRate": trial.suggest_float("subsamplingRate", 0.7, 0.9),
    # Amostragem mais conservadora que o padrão (0.632) para estabilidade
    
    "featureSubsetStrategy": "sqrt",
    # sqrt(n_features) é o padrão ouro para RF
    # Reduz dimensionalidade eficientemente
    
    "maxBins": 32,
    # Mesma lógica da Decision Tree
    
    "seed": seed,
    # Fixar seed para reprodutibilidade
}
Por que funciona para desbalanceamento?

O bagging inerente ao RF naturalmente balanceia as classes

Subamostragem (subsamplingRate) aumenta a chance de incluir exemplos minoritários

Feature subsampling (sqrt) reduz correlação entre árvores

Custo Computacional:

Paralelizável quase linearmente (10x árvores ≈ 10x tempo com 10 cores)

Memória proporcional a numTrees × maxDepth

No Spark, usar numTrees menor com mais workers

4. GBT (Gradient Boosted Trees) - O Poderoso mas Exigente
python
Copy
{
    "maxIter": trial.suggest_int("maxIter", 20, 80),
    # Número de iterações: valores altos levam a overfitting em dados desbalanceados
    # Early stopping implícito via validação cruzada seria ideal
    
    "maxDepth": trial.suggest_int("maxDepth", 3, 6),
    # Árvores extremamente rasas (stumps de 3-4 níveis)
    # Fundamental para evitar overfitting
    
    "stepSize": trial.suggest_float("stepSize", 0.05, 0.1),
    # Shrinkage/learning rate: valores mais altos que o usual (0.01) para convergência mais rápida
    
    "subsamplingRate": 0.8,
    # Fixo para estabilidade - stochastic GBT ajuda contra overfitting
    
    "lossType": "logistic",
    # Única opção sensata para classificação
    
    "minInstancesPerNode": 30,
    # Valor alto para forçar splits significativos
}
Desafios com Desbalanceamento:

GBT é naturalmente mais suscetível a overfitting na classe majoritária

Requer parâmetros mais conservadores que RF

Sensível a outliers (problema comum em dados financeiros)

Custo Computacional:

Treino sequencial (não paraleliza bem entre iterações)

Custo cresce linearmente com maxIter

Memória proporcional a maxDepth × maxIter

Recomendado apenas para datasets médios (<1M linhas)

5. LightGBM - O Estado da Arte Eficiente
python
Copy
{
    "numLeaves": trial.suggest_int("numLeaves", 15, 50),
    # Controle direto de complexidade: melhor que maxDepth para LightGBM
    # Valores típicos: 25-35 para crédito
    
    "maxDepth": -1,
    # Desabilitado quando usamos numLeaves
    
    "learningRate": trial.suggest_float("learningRate", 0.05, 0.2),
    # Range maior que GBT pois LightGBM é mais estável
    
    "minDataInLeaf": trial.suggest_int("minDataInLeaf", 30, 100),
    # Crítico para desbalanceamento: valores altos previnem leaves muito específicas
    
    "featureFraction": 0.8,
    # Similar ao featureSubsetStrategy do RF
    
    "baggingFraction": 0.8,
    # Subamostragem de dados para cada iteração
    
    "lambdaL1": trial.suggest_float("lambdaL1", 0, 1),
    # Regularização L1 opcional
    
    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5, 20),
    # Peso da classe positiva: calculado como (n_negativos/n_positivos)
    
    "numThreads": 4,
    # Limitar paralelismo para não sobrecarregar o cluster
}
Vantagens para Crédito:

scale_pos_weight compensa automaticamente o desbalanceamento

Crescimento assimétrico de árvores beneficia a classe minoritária

Suporte nativo a missing values (comum em dados financeiros)

Custo Computacional:

Extremamente eficiente (10-100x mais rápido que GBT)

Uso de memória otimizado via histogramas

Paralelização eficiente (mas limitar threads para evitar contention)

6. XGBoost - O Clássico Poderoso
python
Copy
{
    "n_estimators": trial.suggest_int("n_estimators", 50, 100),
    # Número conservador de árvores (com early stopping seria ideal)
    
    "max_depth": trial.suggest_int("max_depth", 3, 6),
    # Profundidade limitada como em outros ensembles
    
    "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
    # Taxa de aprendizado mais alta que o padrão (0.3) para crédito
    
    "subsample": trial.suggest_float("subsample", 0.7, 0.9),
    # Amostragem aleatória para diversidade
    
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
    # Amostragem de features como no RF
    
    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5, 20),
    # O parâmetro MAIS IMPORTANTE para desbalanceamento
    # Calcular como count(negativos)/count(positivos)
    
    "tree_method": "hist",
    # Método baseado em histogramas para eficiência
    
    "grow_policy": "depthwise",
    # Mais conservador que lossguide
    
    "eval_metric": "aucpr",
    # AUC-PR é melhor que AUC-ROC para dados desbalanceados
    
    "n_jobs": 4,
    # Paralelismo controlado
}
Otimizações Específicas:

tree_method="hist": até 10x mais rápido que o método exato

enable_categorical: para features como "estado civil" sem one-hot

monotone_constraints: para impor relações monotônicas (ex: renda ↑ score ↑)
