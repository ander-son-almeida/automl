import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Configurações
plt.style.use('seaborn-white')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Dados de exemplo
features_modelo1 = [
    'idade', 'renda', 'escolaridade', 'tempo_emprego', 'valor_emprestimo',
    'num_contas', 'regiao', 'tipo_residencia', 'valor_patrimonio', 
    'limite_cartao', 'idade_cartao', 'num_transacoes',  # Exclusivas Modelo 1
    'historico_pagamentos', 'divida_total'  # Compartilhadas
]

features_modelo2 = [
    'score_credito', 'renda', 'valor_imovel', 'tempo_residencia', 
    'estado_civil', 'num_dependentes', 'tipo_emprego', 'uso_cartao',
    'score_comportamento', 'faturamento_anual',  # Exclusivas Modelo 2
    'historico_pagamentos', 'divida_total'  # Compartilhadas
]

# Preparar os dados
set1 = set(features_modelo1)
set2 = set(features_modelo2)
features_all = sorted(list(set1.union(set2)))

# Criar DataFrame com presença/ausência
df = pd.DataFrame({
    'Feature': features_all,
    'Modelo 1': [f in set1 for f in features_all],
    'Modelo 2': [f in set2 for f in features_all]
}).set_index('Feature')

# Cores personalizadas
cor_presente = '#3498db'  # Azul para presente
cor_ausente = '#ecf0f1'   # Cinza claro para ausente

# Criar figura
fig, ax = plt.subplots(figsize=(12, 8))

# Heatmap com cores individuais para cada célula
sns.heatmap(
    df,
    cmap=[cor_ausente, cor_presente],  # 0=cinza, 1=azul
    linewidths=0.5,
    linecolor='white',
    annot=False,
    cbar=False,
    ax=ax
)

# Ajustar eixos
plt.yticks(rotation=0, fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title('Presença de Features em Cada Modelo', fontsize=14, pad=20)

# Legenda
legend_elements = [
    Patch(facecolor=cor_presente, edgecolor='white', label='Presente no Modelo'),
    Patch(facecolor=cor_ausente, edgecolor='white', label='Ausente no Modelo')
]

ax.legend(
    handles=legend_elements,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    frameon=True,
    title='Legenda',
    title_fontsize='12'
)

# Adicionar informação sobre features
total_features = len(features_all)
compartilhadas = len(set1 & set2)
info_text = f"Total de features: {total_features} | Compartilhadas: {compartilhadas}"
plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.7})

plt.tight_layout()
plt.show()
