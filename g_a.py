import pandas as pd

# Dados de exemplo (já com KS calculado)
data = {
    "Faixa": ["300-400", "401-500", "501-600", "601-700"],
    "Bons": [50, 100, 200, 250],
    "Maus": [30, 20, 10, 5],
    "Acum. Bons": [50, 150, 350, 600],
    "Acum. Maus": [30, 50, 60, 65],
    "KS": [20, 45, 65, 80]
}
df = pd.DataFrame(data)

# Totais de Bons e Maus (para normalização)
total_bons = df["Acum. Bons"].max()
total_maus = df["Acum. Maus"].max()

# Inicializa listas para AUC e Gini parciais
auc_parcial = []
gini_parcial = []

# Loop para calcular AUC e Gini por faixa
for i in range(len(df)):
    if i == 0:
        # Primeira faixa: compara com (0, 0)
        auc_i = (df.loc[i, "Acum. Maus"] / total_maus) * (df.loc[i, "Acum. Bons"] / total_bons) / 2
    else:
        # Demais faixas: compara com a faixa anterior
        auc_i = (
            (df.loc[i, "Acum. Maus"] / total_maus + df.loc[i-1, "Acum. Maus"] / total_maus) *
            (df.loc[i, "Acum. Bons"] / total_bons - df.loc[i-1, "Acum. Bons"] / total_bons)
        ) / 2
    
    # Gini parcial = 2 * AUC parcial - (FPR_i - FPR_{i-1})
    if i == 0:
        gini_i = 2 * auc_i - (df.loc[i, "Acum. Maus"] / total_maus - 0)
    else:
        gini_i = 2 * auc_i - (df.loc[i, "Acum. Maus"] / total_maus - df.loc[i-1, "Acum. Maus"] / total_maus)
    
    auc_parcial.append(auc_i)
    gini_parcial.append(gini_i)

# Adiciona ao DataFrame
df["AUC Parcial"] = auc_parcial
df["Gini Parcial"] = gini_parcial

# Exibe o DataFrame final
print(df)
