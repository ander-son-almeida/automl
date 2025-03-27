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
