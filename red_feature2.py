import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from pyswarm import pso

# 1. Inicializar Spark
spark = SparkSession.builder.appName("PSO_Feature_Selection_Full").getOrCreate()

# 2. Carregar dados (substitua pelo seu DataFrame)
df = spark.read.parquet("caminho/para/seu/dataset.parquet")
feature_cols = [col for col in df.columns if col != "target"]
target_col = "target"

# Converter para Pandas (para o PSO)
pdf = df.select(feature_cols + [target_col]).toPandas()
X = pdf[feature_cols].values
y = pdf[target_col].values

# 3. Configuração do PSO e histórico
n_features = len(feature_cols)
lb = np.zeros(n_features)  # Limite inferior (0 = feature não selecionada)
ub = np.ones(n_features)   # Limite superior (1 = feature selecionada)

history = {
    'positions': [],      # Armazena todas as posições do enxame
    'fitness': [],        # Armazena os valores de fitness
    'ks_scores': [],      # Armazena os KS scores
    'num_features': []    # Armazena o número de features selecionadas
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

# 5. Função de Fitness (com histórico)
def fitness_function(features):
    ks = calculate_ks(features)
    num_features = sum(features > 0.5)
    penalty = 0.01 * num_features  # Peso ajustável
    
    # Armazenar histórico (será preenchido durante a execução do PSO)
    return -ks + penalty  # Minimizar = Maximizar KS

# 6. Função de callback para capturar o histórico
def callback(positions, fitness):
    history['positions'].append(positions.copy())  # Armazena todas as posições do enxame
    history['fitness'].append(np.min(fitness))    # Armazena o melhor fitness da iteração
    
    # Calcular métricas adicionais para o melhor indivíduo
    best_idx = np.argmin(fitness)
    best_position = positions[best_idx]
    ks = calculate_ks(best_position)
    history['ks_scores'].append(ks)
    history['num_features'].append(sum(best_position > 0.5))

# 7. Executar PSO
best_features, best_fitness = pso(
    fitness_function,
    lb, ub,
    swarmsize=30,
    maxiter=50,
    debug=True,
    callback=callback  # Usamos o callback para capturar o histórico
)

# 8. Processar resultados
selected_features = best_features > 0.5
selected_feature_names = [feature_cols[i] for i in np.where(selected_features)[0]]

print("\n=== Resultados ===")
print(f"Melhor KS: {max(history['ks_scores']):.4f}")
print(f"Número de Features Selecionadas: {sum(selected_features)}")
print(f"Features: {selected_feature_names}")

# 9. Gráficos
plt.figure(figsize=(15, 5))

# Gráfico 1: Evolução do KS
plt.subplot(1, 3, 1)
plt.plot(history['ks_scores'], 'b-o')
plt.xlabel("Iteração")
plt.ylabel("KS Score")
plt.title("Melhor KS por Iteração")
plt.grid()

# Gráfico 2: Número de Features Selecionadas
plt.subplot(1, 3, 2)
plt.plot(history['num_features'], 'r--o')
plt.xlabel("Iteração")
plt.ylabel("Número de Features")
plt.title("Features Selecionadas (Melhor Partícula)")
plt.grid()

# Gráfico 3: Matriz de Seleção (todas as partículas na última iteração)
plt.subplot(1, 3, 3)
last_positions = history['positions'][-1]  # Posições na última iteração
selection_matrix = np.array(last_positions) > 0.5
plt.imshow(selection_matrix, cmap='Blues', aspect='auto')
plt.xlabel("Índice da Feature")
plt.ylabel("Partícula")
plt.title(f"Seleção na Última Iteração\n(Enxame = {len(last_positions)} partículas)")
plt.colorbar(label="Selecionada (1)")

plt.tight_layout()
plt.show()

# 10. Salvar features selecionadas (opcional)
# spark.createDataFrame([(name,) for name in selected_feature_names], ["feature"]).write.parquet("features_selecionadas.parquet")
