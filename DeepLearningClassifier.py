from synapse.ml.deeplearning import DeepLearningClassifier

models = [
    # ... outros modelos existentes ...
    ("DeepLearningClassifier", DeepLearningClassifier(
        featuresCol='features',
        labelCol=variavel_resposta,
        layers=[len(features_cols), 10, 5, 2],  # Arquitetura da rede
        maxEpochs=10,
        seed=seed
    )),
    # ... outros modelos ...
]


elif model_name == "DeepLearningClassifier":
    model = DeepLearningClassifier(
        featuresCol='features',
        labelCol=variavel_resposta,
        layers=[
            len(features_cols),
            trial.suggest_int("hidden1_size", 5, 20),
            trial.suggest_int("hidden2_size", 3, 10),
            2
        ],
        maxEpochs=trial.suggest_int("maxEpochs", 5, 20),
        learningRate=trial.suggest_float("learningRate", 1e-4, 1e-1, log=True),
        batchSize=trial.suggest_categorical("batchSize", [32, 64, 128]),
        seed=seed
    )


elif winning_model_name == "DeepLearningClassifier":
    model = DeepLearningClassifier(
        featuresCol='features',
        labelCol=variavel_resposta,
        **best_params
    )

# Tamanho de Amostra Recomendado para Deep Learning em Dados Tabulares

Para o `DeepLearningClassifier` (ou qualquer modelo de redes neurais em dados estruturados), as recomendações de tamanho amostral variam conforme a complexidade do problema, mas aqui estão diretrizes gerais baseadas na literatura:

## **Recomendações Gerais**

1. **Mínimo Absoluto**  
   - **10.000+ registros** para problemas simples (poucas features, padrões claros)
   - **50.000+ registros** para problemas moderadamente complexos

2. **Casos Ideais**  
   - **>100.000 registros** para aproveitar plenamente o poder de deep learning
   - **5.000-10.000 amostras por classe** em problemas de classificação balanceados

3. **Relação Features-Amostras**  
   - Mínimo de **50-100 amostras por feature relevante**  
   - Exemplo: Se seu dataset tem 100 features, idealmente ter **5.000-10.000 registros**

## **Contexto Específico para Modelos de Crédito**

Para problemas tradicionais de scoring de crédito (onde o `DeepLearningClassifier` competiria com XGBoost/LightGBM):

| Complexidade do Problema | Tamanho Mínimo Recomendado | Observações |
|-------------------------|---------------------------|-------------|
| Modelos simples (10-20 features) | 50.000-100.000 registros | Redes rasas (1-2 camadas ocultas) podem funcionar |
| Modelos complexos (100+ features) | 250.000+ registros | Necessário para modelos profundos (>3 camadas) |
| Dados desbalanceados (ex: fraude) | 5.000+ da classe minoritária | Técnicas como oversampling são essenciais |

## **Quando Considerar Deep Learning?**
Só vale a pena testar o `DeepLearningClassifier` se:
1. Seu dataset tem **>100.000 amostras**
2. Você suspeita de **padrões não-lineares complexos** não capturáveis por GBDT (XGBoost/LightGBM)
3. Tem **recursos computacionais** adequados (GPUs, tempo de treino)

## **Alternativas para Datasets Pequenos**
Se seu dataset é menor que 50.000 registros:
- **Prefira XGBoost/LightGBM**: Melhor custo-benefício em dados tabulares pequenos/médios
- **Redes Neurais Rasas**: Use no máximo 1-2 camadas ocultas com regularização pesada
- **Transfer Learning**: Se possível, inicialize pesos com modelos pré-treinados em dados similares

## **Referências da Literatura**
- Estudos empíricos (como [este paper da Google](https://arxiv.org/abs/2106.03253)) mostram que redes neurais só superam GBDTs em dados tabulares quando:
  - Dataset > 10.000 amostras
  - Há features complexas (embedding de texto, imagens)
- Livros como *"Deep Learning with Python"* (Chollet) recomendam 10x mais dados que parâmetros do modelo

Quer que eu ajuste sua função `optimize_model` para incluir recomendações de arquitetura baseadas no tamanho do dataset?
