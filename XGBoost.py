from xgboost.spark import SparkXGBClassifier

models = [
    # ... outros modelos existentes ...
    ("XGBoostClassifier", SparkXGBClassifier(
        features_col="features",
        label_col=variavel_resposta,
        prediction_col="prediction",
        probability_col="probability",
        seed=seed
    )),
    # ... outros modelos ...
]

elif model_name == "XGBoostClassifier":
    model = SparkXGBClassifier(
        features_col="features",
        label_col=variavel_resposta,
        num_workers=4,  # Número de tarefas Spark para paralelismo
        n_estimators=trial.suggest_int("n_estimators", 50, 200),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0, 1),
        reg_lambda=trial.suggest_float("reg_lambda", 0, 1),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        gamma=trial.suggest_float("gamma", 0, 1),
        seed=seed
    )


elif winning_model_name == "XGBoostClassifier":
    model = SparkXGBClassifier(
        features_col="features",
        label_col=variavel_resposta,
        **best_params
    )
