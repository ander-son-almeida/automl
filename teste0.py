import matplotlib.pyplot as plt
import numpy as np

# Configurações
np.random.seed(42)
TOTAL_FEATURES = 25  # Altere para seu número real de features
features = [f'Feature {i+1}' for i in range(TOTAL_FEATURES)]
status = np.random.choice(['Indicado', 'Não Indicado', 'Reprovado'], size=TOTAL_FEATURES)

# Mapeamento de emojis e cores
emoji_map = {'Indicado': '🟢', 'Não Indicado': '🔴', 'Reprovado': '⚪'}

# Criar figura
fig = plt.figure(figsize=(12, 8 if TOTAL_FEATURES > 10 else 6))

# --- Função para adicionar features a um subplot ---
def add_features(ax, features_subset, status_subset):
    for i, (feature, stat) in enumerate(zip(features_subset, status_subset)):
        ax.text(
            0.1, len(features_subset) - i - 0.5,
            f"{emoji_map[stat]} {feature}",
            fontsize=12,
            va='center'
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(features_subset))
    ax.axis('off')

# --- Lógica de plot único ou subplots ---
if TOTAL_FEATURES <= 10:
    add_features(plt.gca(), features, status)
else:
    FEATURES_PER_SUBPLOT = 10
    n_subplots = int(np.ceil(TOTAL_FEATURES / FEATURES_PER_SUBPLOT))
    n_rows = int(np.ceil(n_subplots / 2))
    
    for i in range(n_subplots):
        ax = fig.add_subplot(n_rows, 2, i+1)
        start = i * FEATURES_PER_SUBPLOT
        subset = features[start:start + FEATURES_PER_SUBPLOT]
        subset_status = status[start:start + FEATURES_PER_SUBPLOT]
        add_features(ax, subset, subset_status)

# --- Legenda única (fora dos subplots) ---
legend_text = "\n".join([f"{emoji_map[s]} = {s}" for s in emoji_map.keys()])
plt.figtext(
    0.5, 0.95, 
    legend_text, 
    ha='center', 
    va='top', 
    fontsize=11,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    
plt.tight_layout(rect=[0, 0, 1, 0.92])  # Ajusta espaço para a legenda
plt.show()
