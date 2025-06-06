import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import ks_2samp, norm
from typing import Optional, Tuple

def calcular_tamanho_amostra(populacao: int, confianca: float = 0.95, margem_erro: float = 0.05) -> int:
    """Calcula tamanho da amostra usando distribuição normal para proporções"""
    z = norm.ppf(1 - (1 - confianca)/2)
    n = (z**2 * 0.25) / (margem_erro**2)
    return int(np.ceil(n / (1 + (n - 1)/populacao)))

def pipeline_completo(
    df_spark,
    col_data: str,
    col_target: Optional[str] = None,
    usar_modelo: bool = True,
    meses_baseline: int = 3,
    threshold_moda: float = 0.9,
    confianca: float = 0.95,
    margem_erro: float = 0.05
) -> Tuple[list, pd.DataFrame]:
    """
    Pipeline completo de seleção de features com relatório detalhado
    
    Parâmetros:
    - df_spark: DataFrame Spark
    - col_data: Nome da coluna de data
    - col_target: Nome da coluna target (opcional)
    - usar_modelo: Se True e target fornecido, usa XGBoost para seleção final
    - meses_baseline: Meses para baseline temporal
    - threshold_moda: Threshold para filtro por moda
    - confianca: Nível de confiança para cálculo da amostra
    - margem_erro: Margem de erro para cálculo da amostra
    
    Retorna:
    - Lista de features selecionadas
    - DataFrame com relatório detalhado
    """
    
    # DataFrame para armazenar o relatório
    relatorio = pd.DataFrame(columns=['feature', 'testes_aplicados', 'status'])
    
    print("\n=== INÍCIO DO PROCESSAMENTO ===")
    print(f"Total de linhas original: {df_spark.count():,}")
    print(f"Target {'fornecido' if col_target else 'não fornecido'}")
    
    # 1. Ordena por data para consistência temporal
    df_spark = df_spark.orderBy(col_data)
    
    # 2. Cálculo do tamanho da amostra
    tamanho_amostra = calcular_tamanho_amostra(df_spark.count(), confianca, margem_erro)
    print(f"\n1. Tamanho da amostra calculado: {tamanho_amostra:,} linhas")
    
    # 3. Amostragem estratificada por mês
    df_spark = df_spark.withColumn("mes_ano", F.date_format(F.col(col_data), "yyyy-MM"))
    meses = [row['mes_ano'] for row in df_spark.select("mes_ano").distinct().collect()]
    
    # Verifica desbalanceamento se target fornecido
    is_desbalanceado = False
    scale_pos_weight = None
    
    if col_target:
        counts = df_spark.groupBy(col_target).count().toPandas()
        prop_classe_maioritaria = counts['count'].max() / counts['count'].sum()
        is_desbalanceado = prop_classe_maioritaria > 0.8
        scale_pos_weight = len(counts[counts[col_target] == 0]) / len(counts[counts[col_target] == 1]) if is_desbalanceado else None
        
        print(f"\n2. Amostragem estratificada por {len(meses)} meses")
        print(f"   Proporção target: {counts[counts[col_target] == 0]['count'].values[0]:,} (0) vs {counts[counts[col_target] == 1]['count'].values[0]:,} (1)")
        print(f"   {'⚠️ Dados DESBALANCEADOS' if is_desbalanceado else '✅ Dados balanceados'}")
        
        # Coleta amostras mantendo proporção por classe
        amostras = []
        for mes in meses:
            df_mes = df_spark.filter(F.col("mes_ano") == mes)
            if is_desbalanceado:
                n0 = int(df_mes.filter(F.col(col_target) == 0).count() * tamanho_amostra / df_spark.count())
                n1 = max(1, int(df_mes.filter(F.col(col_target) == 1).count() * tamanho_amostra / df_spark.count()))
                amostras.append(df_mes.filter(F.col(col_target) == 0).orderBy(F.rand()).limit(n0).union(
                       df_mes.filter(F.col(col_target) == 1).orderBy(F.rand()).limit(n1)))
            else:
                amostras.append(df_mes.orderBy(F.rand()).limit(int(tamanho_amostra/len(meses))))
    else:
        # Amostragem simples sem estratificação por target
        amostras = []
        for mes in meses:
            df_mes = df_spark.filter(F.col("mes_ano") == mes)
            amostras.append(df_mes.orderBy(F.rand()).limit(int(tamanho_amostra/len(meses))))
    
    df_amostra = spark.createDataFrame(amostras[0].rdd)
    for a in amostras[1:]:
        df_amostra = df_amostra.union(a)
    df = df_amostra.toPandas()
    
    # 4. Limpeza inicial
    cols_remover = [col_data, "mes_ano"]
    if col_target:
        cols_remover.append(col_target)
    
    X = df.drop(columns=cols_remover)
    y = df[col_target] if col_target else None
    
    # Remove colunas com >95% missings
    cols_missing = X.columns[X.isnull().mean() > 0.95].tolist()
    X_clean = X.drop(columns=cols_missing)
    
    # Atualiza relatório
    for col in X.columns:
        testes = []
        status = "Mantida"
        
        if col in cols_missing:
            testes.append("missing>95%")
            status = "Removida"
        
        relatorio.loc[len(relatorio)] = [col, ", ".join(testes), status]
    
    # Filtro por moda
    cols_moda = []
    for col in X_clean.columns:
        if X_clean[col].nunique() > 1:
            moda = X_clean[col].mode()[0]
            proporcao_moda = (X_clean[col] == moda).mean()
            if proporcao_moda > threshold_moda:
                cols_moda.append(col)
                relatorio.loc[relatorio['feature'] == col, 'testes_aplicados'] += ", moda>" + str(threshold_moda)
                relatorio.loc[relatorio['feature'] == col, 'status'] = "Removida"
    
    X_clean = X_clean.drop(columns=cols_moda)
    
    print(f"\n3. Limpeza inicial:")
    print(f"   - Removidas {len(cols_missing)} features com >95% missings")
    print(f"   - Removidas {len(cols_moda)} features com moda > {threshold_moda*100}%")
    print(f"   - Features restantes: {len(X_clean.columns)}")
    
    # 5. Seleção univariada (apenas se target fornecido)
    if col_target:
        metric_scores = {}
        for col in X_clean.columns:
            if is_desbalanceado:
                precision, recall, _ = precision_recall_curve(y, X_clean[col].fillna(X_clean[col].median()))
                metric_scores[col] = auc(recall, precision)
            else:
                metric_scores[col] = roc_auc_score(y, X_clean[col].fillna(X_clean[col].median()))
        
        mi = mutual_info_classif(X_clean.fillna(X_clean.median()), y, random_state=42)
        
        threshold = 0.2 if is_desbalanceado else 0.55
        selected = [col for i, col in enumerate(X_clean.columns) 
                   if (metric_scores[col] > threshold) and (mi[i] > 0.01)]
        X_filtered = X_clean[selected]
        
        # Atualiza relatório
        for col in X_clean.columns:
            if col in selected:
                relatorio.loc[relatorio['feature'] == col, 'testes_aplicados'] += ", univariada"
            else:
                if col in X_clean.columns:
                    relatorio.loc[relatorio['feature'] == col, 'status'] = "Removida"
                    relatorio.loc[relatorio['feature'] == col, 'testes_aplicados'] += ", univariada"
        
        print(f"\n4. Seleção univariada - {len(selected)} features selecionadas")
        print(f"   Métrica usada: {'AUC-PR' if is_desbalanceado else 'AUC-ROC'}")
    else:
        X_filtered = X_clean
        print("\n4. Target não fornecido - Pulando seleção univariada")
    
    # 6. Redução de redundância
    corr_matrix = X_filtered.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
    X_low_corr = X_filtered.drop(columns=to_drop)
    
    # Atualiza relatório
    for col in X_filtered.columns:
        if col in to_drop:
            relatorio.loc[relatorio['feature'] == col, 'testes_aplicados'] += ", correlação>0.9"
            relatorio.loc[relatorio['feature'] == col, 'status'] = "Removida"
    
    print(f"\n5. Redução de redundância - {len(to_drop)} features removidas por alta correlação")
    print(f"   Features restantes: {len(X_low_corr.columns)}")
    
    # 7. Validação temporal
    meses_ordenados = sorted(df["mes_ano"].unique())
    baseline_data = df[df["mes_ano"].isin(meses_ordenados[:meses_baseline])]
    
    stable_features = []
    for col in X_low_corr.columns:
        psi_values = []
        baseline = baseline_data[col].fillna(baseline_data[col].median())
        
        for mes in meses_ordenados[meses_baseline:]:
            current = df[df["mes_ano"] == mes][col].fillna(df[col].median())
            
            bins = np.histogram_bin_edges(np.concatenate([baseline, current]), bins=10)
            baseline_hist = np.histogram(baseline, bins=bins)[0] + 1e-6
            current_hist = np.histogram(current, bins=bins)[0] + 1e-6
            psi = np.sum((current_hist - baseline_hist) * np.log(current_hist / baseline_hist))
            psi_values.append(psi)
        
        if np.mean(psi_values) < 0.1:
            stable_features.append(col)
        else:
            relatorio.loc[relatorio['feature'] == col, 'testes_aplicados'] += ", PSI>0.1"
            relatorio.loc[relatorio['feature'] == col, 'status'] = "Removida"
    
    X_stable = X_low_corr[stable_features]
    
    print(f"\n6. Validação temporal (baseline={meses_baseline} meses)")
    print(f"   Features estáveis: {len(stable_features)}")
    print(f"   Features removidas: {len(X_low_corr.columns)-len(stable_features)}")
    
    # 8. Seleção por modelo (apenas se target fornecido e usar_modelo=True)
    if col_target and usar_modelo:
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr' if is_desbalanceado else 'auc',
            n_estimators=100,
            subsample=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_stable.fillna(X_stable.median()), y)
        
        importance = model.feature_importances_
        selected_final = [col for col, imp in zip(X_stable.columns, importance) if imp > np.mean(importance)]
        
        # Atualiza relatório
        for col in X_stable.columns:
            if col not in selected_final:
                relatorio.loc[relatorio['feature'] == col, 'testes_aplicados'] += ", importância_modelo"
                relatorio.loc[relatorio['feature'] == col, 'status'] = "Removida"
        
        print(f"\n7. Seleção por modelo - {len(selected_final)} features selecionadas")
    else:
        selected_final = stable_features
        print("\n7. Seleção por modelo não aplicada")
    
    print("\n=== PROCESSAMENTO CONCLUÍDO ===")
    
    # Filtra relatório apenas para features que passaram por algum teste
    relatorio = relatorio[relatorio['testes_aplicados'] != ""]
    
    return selected_final, relatorio

# Exemplo de uso
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    df_spark = spark.read.parquet("caminho/do/seu/dataframe")
    
    # Caso com target
    features_com_target, relatorio_com_target = pipeline_completo(
        df_spark=df_spark,
        col_data="data_mensal",
        col_target="target",
        usar_modelo=True
    )
    
    print("\nRelatório com target:")
    print(relatorio_com_target)
    
    # Caso sem target
    features_sem_target, relatorio_sem_target = pipeline_completo(
        df_spark=df_spark,
        col_data="data_mensal",
        col_target=None
    )
    
    print("\nRelatório sem target:")
    print(relatorio_sem_target)
