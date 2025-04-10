import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, auc,
                            precision_recall_curve, average_precision_score,
                            confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve

# Modelos equivalentes
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, 
                             GradientBoostingClassifier)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

class SparkToSklearnConverter:
    def __init__(self, spark_model_name, spark_params, random_state=42):
        self.spark_model_name = spark_model_name
        self.spark_params = spark_params
        self.random_state = random_state
        self.sklearn_model = None
        self.sklearn_params = None
        
    def convert(self):
        """Converte o modelo Spark para scikit-learn"""
        converter_method = getattr(self, f"convert_{self.spark_model_name}", None)
        if converter_method:
            return converter_method()
        else:
            raise ValueError(f"Modelo {self.spark_model_name} não suportado para conversão")
    
    def convert_LogisticRegression(self):
        """Converte LogisticRegression do Spark para scikit-learn"""
        params_map = {
            'regParam': 'C',
            'elasticNetParam': 'l1_ratio',
            'maxIter': 'max_iter',
            'tol': 'tol'
        }
        
        sklearn_params = {
            'penalty': 'elasticnet' if self.spark_params.get('elasticNetParam', 0) > 0 else 'l2',
            'solver': 'saga',
            'random_state': self.random_state
        }
        
        for spark_param, sklearn_param in params_map.items():
            if spark_param in self.spark_params:
                sklearn_params[sklearn_param] = self.spark_params[spark_param]
        
        # Inversão do parâmetro C (Spark: regParam, sklearn: 1/C)
        if 'C' in sklearn_params:
            sklearn_params['C'] = 1.0 / max(sklearn_params['C'], 1e-10)
        
        self.sklearn_model = LogisticRegression(**sklearn_params)
        self.sklearn_params = sklearn_params
        return self
    
    def convert_DecisionTreeClassifier(self):
        """Converte DecisionTreeClassifier do Spark para scikit-learn"""
        params_map = {
            'maxDepth': 'max_depth',
            'minInstancesPerNode': 'min_samples_split',
            'minInfoGain': 'min_impurity_decrease',
            'maxBins': None,  # Não tem equivalente direto
            'seed': 'random_state'
        }
        
        sklearn_params = {
            'random_state': self.random_state
        }
        
        for spark_param, sklearn_param in params_map.items():
            if spark_param in self.spark_params and sklearn_param is not None:
                sklearn_params[sklearn_param] = self.spark_params[spark_param]
        
        self.sklearn_model = DecisionTreeClassifier(**sklearn_params)
        self.sklearn_params = sklearn_params
        return self
    
    def convert_RandomForestClassifier(self):
        """Converte RandomForestClassifier do Spark para scikit-learn"""
        params_map = {
            'numTrees': 'n_estimators',
            'maxDepth': 'max_depth',
            'minInstancesPerNode': 'min_samples_split',
            'featureSubsetStrategy': 'max_features',
            'seed': 'random_state'
        }
        
        sklearn_params = {
            'random_state': self.random_state,
            'n_jobs': -1  # Usar todos os cores
        }
        
        for spark_param, sklearn_param in params_map.items():
            if spark_param in self.spark_params:
                if spark_param == 'featureSubsetStrategy':
                    # Mapear estratégias de subconjunto de features
                    strategy = self.spark_params[spark_param]
                    if strategy == 'auto':
                        sklearn_params[sklearn_param] = 'sqrt'
                    elif strategy == 'all':
                        sklearn_params[sklearn_param] = 1.0
                    elif strategy == 'onethird':
                        sklearn_params[sklearn_param] = 0.33
                    else:
                        sklearn_params[sklearn_param] = 'sqrt'
                else:
                    sklearn_params[sklearn_param] = self.spark_params[spark_param]
        
        self.sklearn_model = RandomForestClassifier(**sklearn_params)
        self.sklearn_params = sklearn_params
        return self
    
    def convert_GBTClassifier(self):
        """Converte GBTClassifier do Spark para scikit-learn"""
        params_map = {
            'maxIter': 'n_estimators',
            'maxDepth': 'max_depth',
            'stepSize': 'learning_rate',
            'minInstancesPerNode': 'min_samples_split',
            'seed': 'random_state'
        }
        
        sklearn_params = {
            'random_state': self.random_state
        }
        
        for spark_param, sklearn_param in params_map.items():
            if spark_param in self.spark_params:
                sklearn_params[sklearn_param] = self.spark_params[spark_param]
        
        self.sklearn_model = GradientBoostingClassifier(**sklearn_params)
        self.sklearn_params = sklearn_params
        return self
    
    def convert_LightGBMClassifier(self):
        """Converte LightGBMClassifier do Spark para scikit-learn"""
        params_map = {
            'numLeaves': 'num_leaves',
            'maxDepth': 'max_depth',
            'learningRate': 'learning_rate',
            'featureFraction': 'feature_fraction',
            'baggingFraction': 'subsample',
            'baggingFreq': 'subsample_freq',
            'lambdaL1': 'reg_alpha',
            'lambdaL2': 'reg_lambda',
            'minGainToSplit': 'min_split_gain',
            'minDataInLeaf': 'min_child_samples',
            'maxBin': 'max_bin',
            'seed': 'random_state'
        }
        
        sklearn_params = {
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        
        for spark_param, sklearn_param in params_map.items():
            if spark_param in self.spark_params:
                sklearn_params[sklearn_param] = self.spark_params[spark_param]
        
        self.sklearn_model = LGBMClassifier(**sklearn_params)
        self.sklearn_params = sklearn_params
        return self
    
    def convert_XGBoostClassifier(self):
        """Converte XGBoostClassifier do Spark para scikit-learn"""
        params_map = {
            'maxDepth': 'max_depth',
            'eta': 'learning_rate',
            'minChildWeight': 'min_child_weight',
            'subsample': 'subsample',
            'colsampleByTree': 'colsample_bytree',
            'colsampleByLevel': 'colsample_bylevel',
            'lambda': 'reg_lambda',
            'alpha': 'reg_alpha',
            'gamma': 'gamma',
            'numRound': 'n_estimators',
            'seed': 'random_state'
        }
        
        sklearn_params = {
            'random_state': self.random_state,
            'n_jobs': -1,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        for spark_param, sklearn_param in params_map.items():
            if spark_param in self.spark_params:
                sklearn_params[sklearn_param] = self.spark_params[spark_param]
        
        self.sklearn_model = XGBClassifier(**sklearn_params)
        self.sklearn_params = sklearn_params
        return self
    
    def get_model(self):
        """Retorna o modelo scikit-learn convertido"""
        if self.sklearn_model is None:
            raise ValueError("Modelo não convertido. Chame o método convert() primeiro.")
        return self.sklearn_model
    
    def get_params(self):
        """Retorna os parâmetros scikit-learn mapeados"""
        if self.sklearn_params is None:
            raise ValueError("Parâmetros não convertidos. Chame o método convert() primeiro.")
        return self.sklearn_params


class ModelEvaluator:
    def __init__(self, model, model_name, X_train, X_test, y_train, y_test):
        self.model = model
        self.model_name = model_name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.is_fitted = False
        self.probabilities = None
        self.predictions = None
        self.metrics = {}
        
    def fit(self):
        """Treina o modelo"""
        self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True
        return self
    
    def predict(self):
        """Faz previsões e calcula probabilidades"""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Chame o método fit() primeiro.")
        
        self.predictions = self.model.predict(self.X_test)
        
        # Para modelos que suportam probabilidades
        if hasattr(self.model, "predict_proba"):
            self.probabilities = self.model.predict_proba(self.X_test)[:, 1]
        elif hasattr(self.model, "decision_function"):
            # Para modelos como SVM que não têm predict_proba
            decision_scores = self.model.decision_function(self.X_test)
            self.probabilities = 1 / (1 + np.exp(-decision_scores))
        else:
            self.probabilities = None
            
        return self
    
    def calculate_metrics(self):
        """Calcula todas as métricas de avaliação"""
        if self.predictions is None:
            raise ValueError("Previsões não calculadas. Chame o método predict() primeiro.")
        
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, self.predictions),
            'precision': precision_score(self.y_test, self.predictions),
            'recall': recall_score(self.y_test, self.predictions),
            'f1': f1_score(self.y_test, self.predictions),
            'roc_auc': roc_auc_score(self.y_test, self.probabilities) if self.probabilities is not None else None,
            'average_precision': average_precision_score(self.y_test, self.probabilities) if self.probabilities is not None else None,
            'classification_report': classification_report(self.y_test, self.predictions, output_dict=True)
        }
        
        return self.metrics
    
    def plot_confusion_matrix(self):
        """Plota a matriz de confusão"""
        cm = confusion_matrix(self.y_test, self.predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negativo', 'Positivo'], 
                    yticklabels=['Negativo', 'Positivo'])
        plt.title(f'Matriz de Confusão - {self.model_name}')
        plt.xlabel('Previsto')
        plt.ylabel('Real')
        plt.show()
    
    def plot_roc_curve(self):
        """Plota a curva ROC"""
        if self.probabilities is None:
            print("Modelo não suporta probabilidades. Não é possível plotar ROC.")
            return
        
        fpr, tpr, _ = roc_curve(self.y_test, self.probabilities)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - {self.model_name}')
        plt.legend(loc="lower right")
        plt.show()
    
    def plot_pr_curve(self):
        """Plota a curva Precision-Recall"""
        if self.probabilities is None:
            print("Modelo não suporta probabilidades. Não é possível plotar PR curve.")
            return
        
        precision, recall, _ = precision_recall_curve(self.y_test, self.probabilities)
        average_precision = average_precision_score(self.y_test, self.probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Curva Precision-Recall - {self.model_name}')
        plt.legend(loc="lower left")
        plt.show()
    
    def plot_calibration_curve(self):
        """Plota a curva de calibração"""
        if self.probabilities is None:
            print("Modelo não suporta probabilidades. Não é possível plotar calibration curve.")
            return
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_test, self.probabilities, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=f'{self.model_name}')
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted value")
        plt.title(f'Curva de Calibração - {self.model_name}')
        plt.legend()
        plt.show()
    
    def plot_probability_distribution(self):
        """Plota a distribuição de probabilidades com KS"""
        if self.probabilities is None:
            print("Modelo não suporta probabilidades. Não é possível plotar distribuição.")
            return
        
        # Calcula estatística KS
        pos_probs = self.probabilities[self.y_test == 1]
        neg_probs = self.probabilities[self.y_test == 0]
        ks_stat = np.max(np.abs(np.linspace(0, 1, len(pos_probs)) - np.sort(pos_probs)))
        
        plt.figure(figsize=(12, 8))
        plt.hist(pos_probs, bins=30, alpha=0.5, 
                label=f'Positivo (n={len(pos_probs)})', color='green')
        plt.hist(neg_probs, bins=30, alpha=0.5, 
                label=f'Negativo (n={len(neg_probs)})', color='red')
        
        # Adiciona linha do KS
        ks_x = np.argmax(np.abs(np.cumsum(pos_probs) - np.cumsum(neg_probs)))
        plt.axvline(x=ks_x/len(self.probabilities), color='blue', 
                   linestyle='--', label=f'KS = {ks_stat:.3f}')
        
        plt.xlabel('Probabilidade Prevista')
        plt.ylabel('Frequência')
        plt.title(f'Distribuição de Probabilidades - {self.model_name}\nKS = {ks_stat:.3f}')
        plt.legend()
        plt.show()
    
    def plot_feature_importance(self):
        """Plota a importância das features"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.X_train.columns
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Importância das Features - {self.model_name}')
            plt.bar(range(len(importances)), importances[indices],
                   color='b', align='center')
            plt.xticks(range(len(importances)), np.array(feature_names)[indices], 
                   rotation=90)
            plt.xlim([-1, len(importances)])
            plt.tight_layout()
            plt.show()
        elif hasattr(self.model, 'coef_'):
            # Para modelos lineares como LogisticRegression
            coef = self.model.coef_[0]
            feature_names = self.X_train.columns
            indices = np.argsort(np.abs(coef))[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Coeficientes das Features - {self.model_name}')
            plt.bar(range(len(coef)), coef[indices],
                   color='b', align='center')
            plt.xticks(range(len(coef)), np.array(feature_names)[indices], 
                   rotation=90)
            plt.xlim([-1, len(coef)])
            plt.tight_layout()
            plt.show()
        else:
            print(f"Modelo {self.model_name} não suporta importância de features.")
    
    def calculate_shap_values(self, sample_size=100):
        """Calcula SHAP values para o modelo"""
        if not hasattr(self.model, 'predict_proba'):
            print(f"Modelo {self.model_name} não suporta SHAP values.")
            return None
        
        # Amostra os dados para tornar o cálculo mais rápido
        if len(self.X_train) > sample_size:
            X_sample = shap.utils.sample(self.X_train, sample_size)
        else:
            X_sample = self.X_train
        
        # Cria o explainer SHAP baseado no tipo de modelo
        if isinstance(self.model, (XGBClassifier, LGBMClassifier, 
                                 GradientBoostingClassifier, 
                                 RandomForestClassifier, DecisionTreeClassifier)):
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
        elif isinstance(self.model, (LogisticRegression, LinearSVC)):
            explainer = shap.LinearExplainer(self.model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        else:
            # Usa Kernel SHAP como fallback
            explainer = shap.KernelExplainer(self.model.predict_proba, X_sample)
            shap_values = explainer.shap_values(X_sample)
        
        return explainer, shap_values, X_sample
    
    def plot_shap_summary(self, shap_values, X_sample, feature_names=None):
        """Plota o summary plot dos SHAP values"""
        if feature_names is None:
            feature_names = self.X_train.columns
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary - {self.model_name}')
        plt.tight_layout()
        plt.show()
    
    def plot_shap_dependence(self, shap_values, X_sample, feature_name, interaction_index=None):
        """Plota o dependence plot para uma feature específica"""
        if feature_name not in self.X_train.columns:
            print(f"Feature {feature_name} não encontrada.")
            return
        
        feature_index = list(self.X_train.columns).index(feature_name)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_index, shap_values, X_sample, 
                           interaction_index=interaction_index,
                           show=False)
        plt.title(f'SHAP Dependence - {feature_name} - {self.model_name}')
        plt.tight_layout()
        plt.show()
    
    def evaluate_all(self, plot_shap=True, shap_sample_size=100):
        """Executa todas as avaliações e plots"""
        self.fit()
        self.predict()
        self.calculate_metrics()
        
        print("\n" + "="*50)
        print(f"AVALIAÇÃO DO MODELO: {self.model_name}")
        print("="*50)
        
        print("\nMétricas de Avaliação:")
        for metric, value in self.metrics.items():
            if metric != 'classification_report':
                print(f"{metric.capitalize()}: {value:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.predictions))
        
        # Plots de avaliação
        print("\nGráficos de Avaliação:")
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_pr_curve()
        self.plot_calibration_curve()
        self.plot_probability_distribution()
        self.plot_feature_importance()
        
        # SHAP values (se suportado e solicitado)
        if plot_shap and hasattr(self.model, 'predict_proba'):
            print("\nCalculando SHAP values...")
            try:
                explainer, shap_values, X_sample = self.calculate_shap_values(shap_sample_size)
                
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    # Para classificadores binários, pegamos os valores para a classe positiva
                    shap_values = shap_values[1]
                
                self.plot_shap_summary(shap_values, X_sample)
                
                # Plota dependence plot para as top 3 features
                if hasattr(self.model, 'feature_importances_'):
                    top_features = np.argsort(self.model.feature_importances_)[-3:][::-1]
                    for i in top_features:
                        feature_name = self.X_train.columns[i]
                        self.plot_shap_dependence(shap_values, X_sample, feature_name)
            except Exception as e:
                print(f"Erro ao calcular SHAP values: {str(e)}")
        
        return self.metrics


# Exemplo de uso:
if __name__ == "__main__":
    # 1. Carregar dados de exemplo
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Suponha que temos parâmetros otimizados do Optuna para um modelo Spark
    spark_params = {
        'maxDepth': 5,
        'numTrees': 100,
        'featureSubsetStrategy': 'auto',
        'seed': 42
    }
    
    # 3. Converter o modelo Spark para scikit-learn
    converter = SparkToSklearnConverter('RandomForestClassifier', spark_params)
    converter.convert()
    sklearn_model = converter.get_model()
    print("Parâmetros convertidos:", converter.get_params())
    
    # 4. Avaliar o modelo
    evaluator = ModelEvaluator(sklearn_model, 'RandomForest (convertido)', 
                             X_train, X_test, y_train, y_test)
    metrics = evaluator.evaluate_all(plot_shap=True, shap_sample_size=100)
