import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from pyswarm import pso

# 1. Inicializar a sessão Spark
spark = SparkSession.builder.appName("PSO_Feature_Selection").getOrCreate()

# 2. Carregar dados (exemplo: substitua pelo seu DataFrame)
df = spark.read.parquet("caminho/para/seu/dataset.parquet")
feature_cols = [col for col in df.columns if col != "target"]
target_col = "target"

# Converter para Pandas para o PSO (se os dados couberem na memória)
pdf = df.select(feature_cols + [target_col]).toPandas()
X = pdf[feature_cols].values
y = pdf[target_col].values

# 3. Configuração do PSO
n_features = len(feature_cols)
lb = np.zeros(n_features)  # Limite inferior (0 = feature não selecionada)
ub = np.ones(n_features)   # Limite superior (1 = feature selecionada)

# Histórico para armazenar resultados
history = {
    'fitness': [],
    'num_features': [],
    'ks_scores': []
}

# 4. Função para calcular o KS
def calculate_ks(features_selected):
    selected_indices = np.where(features_selected > 0.5)[0]
    if len(selected_indices) == 0:
        return 0.0
    
    X_subset = X[:, selected_indices]
    model = LGBMClassifier(random_state=42)
    probas = model.fit(X_subset, y).predict_proba(X_subset)[:, 1]
    ks_stat, _ = ks_2samp(probas[y == 1], probas[y == 0])
    return ks_stat

# 5. Função de Fitness (com penalização por muitas features)
def fitness_function(features):
    ks = calculate_ks(features)
    num_features = sum(features > 0.5)
    penalty = 0.01 * num_features  # Peso ajustável
    
    # Armazenar histórico
    history['fitness'].append(-ks + penalty)
    history['ks_scores'].append(ks)
    history['num_features'].append(num_features)
    
    return -ks + penalty  # Minimizar = Maximizar KS

# 6. Executar PSO
best_features, best_fitness = pso(
    fitness_function,
    lb, ub,
    swarmsize=30,      # Número de partículas
    maxiter=50,        # Número de iterações
    debug=True         # Mostra progresso
)

# 7. Resultados
selected_features = best_features > 0.5
selected_feature_names = [feature_cols[i] for i in np.where(selected_features)[0]]

print("\n=== Features Selecionadas ===")
print(f"Número de Features: {sum(selected_features)}")
print(f"Lista: {selected_feature_names}")
print(f"Melhor KS: {max(history['ks_scores']):.4f}")

# 8. Gráficos
plt.figure(figsize=(15, 5))

# Gráfico 1: Evolução do KS
plt.subplot(1, 3, 1)
plt.plot(history['ks_scores'], 'b-')
plt.xlabel("Iteração")
plt.ylabel("KS Score")
plt.title("Evolução do KS")
plt.grid()

# Gráfico 2: Número de Features Selecionadas
plt.subplot(1, 3, 2)
plt.plot(history['num_features'], 'r--')
plt.xlabel("Iteração")
plt.ylabel("Número de Features")
plt.title("Features Selecionadas por Iteração")
plt.grid()

# Gráfico 3: Matriz de Seleção (últimas 20 iterações)
plt.subplot(1, 3, 3)
last_iterations = np.array([pos > 0.5 for pos in history['position'][-20:]])
plt.imshow(last_iterations, cmap='Blues', aspect='auto')
plt.xlabel("Feature Index")
plt.ylabel("Iteração (últimas 20)")
plt.title("Matriz de Seleção")
plt.colorbar(label="Selecionada (1)")

plt.tight_layout()
plt.show()

# 9. Salvar resultados (opcional)
# spark.createDataFrame([(name,) for name in selected_feature_names], ["feature"]).write.parquet("features_selecionadas.parquet")
