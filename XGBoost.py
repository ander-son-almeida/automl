
from synapse.ml.xgboost import XGBoostClassifier

# Modifique a função optimize_model
def optimize_model(model_name, train_df, test_df, variavel_resposta, seed):
    def objective(trial):
        if model_name == "LogisticRegression":
            model = LogisticRegression(
                featuresCol='features',
                labelCol=variavel_resposta,
                regParam=trial.suggest_float("regParam", 0.01, 10.0, log=True),
                elasticNetParam=trial.suggest_float("elasticNetParam", 0.0, 1.0)
            )
        # ... outros modelos existentes ...
        elif model_name == "XGBoostClassifier":
            model = XGBoostClassifier(
                featuresCol='features',
                labelCol=variavel_resposta,
                predictionCol="prediction",
                seed=seed,
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                gamma=trial.suggest_float("gamma", 0, 1),
                reg_alpha=trial.suggest_float("reg_alpha", 0, 1),
                reg_lambda=trial.suggest_float("reg_lambda", 0, 1)
            )
        # ... resto da função objective ...
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=10)
    return study.best_params

# Atualize a lista de modelos na função main
def main(df, features_cols, variavel_resposta, metricas_disponiveis, metrica_vencedora, seed, save_dir):
    # ... código anterior ...
    
    models = [
        # ... outros modelos ...
        ("XGBoostClassifier", XGBoostClassifier(featuresCol='features', labelCol=variavel_resposta, predictionCol="prediction", seed=seed)),
        ("OneVsRest", OneVsRest(classifier=LogisticRegression(featuresCol='features', labelCol=variavel_resposta)))
    ]
    
    # ... código para avaliar modelos ...
    
    # Adicione o caso para XGBoost no treinamento do modelo otimizado
    elif winning_model_name == "XGBoostClassifier":
        model = XGBoostClassifier(
            featuresCol='features',
            labelCol=variavel_resposta,
            predictionCol="prediction",
            seed=seed,
            **best_params
        )
    
    # ... resto da função main ...
