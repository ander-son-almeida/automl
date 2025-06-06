import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import norm
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

def calcular_tamanho_amostra(populacao: int, confianca: float = 0.95, margem_erro: float = 0.05) -> int:
    """Calcula tamanho da amostra usando distribuição normal para proporções"""
    z = norm.ppf(1 - (1 - confianca)/2)
    n = (z**2 * 0.25) / (margem_erro**2)
    return int(np.ceil(n / (1 + (n - 1)/populacao)))

def calcular_psi(baseline: pd.Series, corrente: pd.Series, bins: int = 10) -> float:
    """Calcula o Population Stability Index entre duas distribuições"""
    bins_edges = np.histogram_bin_edges(np.concatenate([baseline, corrente]), bins=bins)
    hist_base = np.histogram(baseline.fillna(baseline.median()), bins=bins_edges)[0] + 1e-6
    hist_current = np.histogram(corrente.fillna(corrente.median()), bins=bins_edges)[0] + 1e-6
    prop_base = hist_base / hist_base.sum()
    prop_current = hist_current / hist_current.sum()
    return np.sum((prop_current - prop_base) * np.log(prop_current / prop_base))

def pipeline_completo_flexivel(
    df_spark,
    col_data: str,
    col_target: Optional[str] = None,
    threshold_missing: float = 0.95,
    threshold_moda: float = 0.9,
    threshold_correlacao: float = 0.9,
    threshold_psi: float = 0.1,
    confianca: float = 0.95,
    margem_erro: float = 0.05,
    meses_baseline: int = 3
) -> Tuple[List[str], pd.DataFrame]:
    """
    Pipeline completo com todos os filtros e flexibilidade para com/sem target
    
    Parâmetros:
    - df_spark: DataFrame Spark
    - col_data: Coluna com data mensal (para ordenação e PSI)
    - col_target: Coluna target (opcional)
    - threshold_missing: Threshold para remoção de missings
    - threshold_moda: Threshold para filtro por moda
    - threshold_correlacao: Threshold para correlação
    - threshold_psi: Threshold para PSI
    - confianca: Nível de confiança para cálculo da amostra
    - margem_erro: Margem de erro para cálculo da amostra
    - meses_baseline: Meses para baseline do PSI
    
    Retorna:
    - Lista de features selecionadas
    - DataFrame com relatório detalhado
    """
    
    # Inicialização
    relatorio = pd.DataFrame(columns=['feature', 'testes', 'status', 'valor_teste'])
    spark = SparkSession.builder.getOrCreate()
    
    print("\n" + "="*60)
    print(" INÍCIO DO PIPELINE DE SELEÇÃO DE FEATURES ".center(60, "="))
    print("="*60)
    print(f"\nConfigurações:")
    print(f"- Target: {'Fornecido' if col_target else 'Não fornecido'}")
    print(f"- Coluna de data: {col_data}")
    print(f"- Thresholds: Missing={threshold_missing}, Moda={threshold_moda}")
    print(f"- PSI: Baseline={meses_baseline} meses, Threshold={threshold_psi}")
    
    # 1. Pré-processamento
    df_spark = df_spark.orderBy(col_data)
    df_spark = df_spark.withColumn("mes_ano", F.date_format(F.col(col_data), "yyyy-MM"))
    
    # 2. Amostragem aleatória
    tamanho_amostra = calcular_tamanho_amostra(df_spark.count(), confianca, margem_erro)
    df = df_spark.orderBy(F.rand()).limit(tamanho_amostra).toPandas()
    print(f"\n[1] Amostra aleatória coletada: {len(df):,} linhas")
    
    # 3. Separação temporal para PSI
    meses_unicos = sorted(df["mes_ano"].unique())
    if len(meses_unicos) < meses_baseline + 1:
        raise ValueError(f"Necessário pelo menos {meses_baseline+1} meses para validação PSI")
    
    baseline_data = df[df["mes_ano"].isin(meses_unicos[:meses_baseline])]
    corrente_data = df[df["mes_ano"].isin(meses_unicos[meses_baseline:])]
    
    # 4. Limpeza inicial
    X = df.drop(columns=[col_data, "mes_ano"] + ([col_target] if col_target else []))
    y = df[col_target] if col_target else None
    
    # 4.1 Remoção por missing
    missing_ratio = X.isnull().mean()
    cols_missing = missing_ratio[missing_ratio > threshold_missing].index.tolist()
    X_clean = X.drop(columns=cols_missing)
    
    for col in X.columns:
        if col in cols_missing:
            relatorio.loc[len(relatorio)] = [col, 'missing', 'Removida', f"{missing_ratio[col]:.1%}"]
    
    # 4.2 Remoção por moda
    cols_moda = []
    for col in X_clean.columns:
        if X_clean[col].nunique() > 1:
            moda = X_clean[col].mode()[0]
            proporcao_moda = (X_clean[col] == moda).mean()
            if proporcao_moda > threshold_moda:
                cols_moda.append(col)
                relatorio.loc[len(relatorio)] = [col, 'moda', 'Removida', f"{proporcao_moda:.1%}"]
    
    X_clean = X_clean.drop(columns=cols_moda)
    print(f"\n[2] Limpeza inicial:")
    print(f"- Features removidas por missing > {threshold_missing:.0%}: {len(cols_missing)}")
    print(f"- Features removidas por moda > {threshold_moda:.0%}: {len(cols_moda)}")
    print(f"- Features restantes: {len(X_clean.columns)}")
    
    # 5. Seleção univariada (se target fornecido)
    if col_target:
        # 5.1 Verifica desbalanceamento
        counts = y.value_counts()
        prop_maioritaria = counts.max() / counts.sum()
        is_desbalanceado = prop_maioritaria > 0.8
        
        print(f"\n[3] Análise do target:")
        print(f"- Classe 0: {counts.get(0, 0)} | Classe 1: {counts.get(1, 0)}")
        print(f"- Proporção maioritária: {prop_maioritaria:.1%} {'(DESBALANCEADO)' if is_desbalanceado else ''}")
        
        # 5.2 Seleção por métricas
        metric_scores = {}
        for col in X_clean.columns:
            if is_desbalanceado:
                precision, recall, _ = precision_recall_curve(y, X_clean[col].fillna(X_clean[col].median()))
                metric_scores[col] = auc(recall, precision)
            else:
                metric_scores[col] = roc_auc_score(y, X_clean[col].fillna(X_clean[col].median()))
        
        mi_scores = mutual_info_classif(X_clean.fillna(X_clean.median()), y, random_state=42)
        
        # 5.3 Aplica thresholds
        threshold_metric = 0.2 if is_desbalanceado else 0.55
        selected_univariate = [
            col for i, col in enumerate(X_clean.columns)
            if (metric_scores[col] > threshold_metric) and (mi_scores[i] > 0.01)
        ]
        
        # Atualiza relatório
        for col in X_clean.columns:
            status = "Selecionada" if col in selected_univariate else "Removida"
            teste = f"AUC={'PR' if is_desbalanceado else 'ROC'}:{metric_scores[col]:.3f}, MI:{mi_scores[i]:.3f}"
            relatorio.loc[len(relatorio)] = [col, 'univariada', status, teste]
        
        X_filtered = X_clean[selected_univariate]
        print(f"\n[4] Seleção univariada:")
        print(f"- Métrica usada: {'AUC-PR' if is_desbalanceado else 'AUC-ROC'} + Mutual Information")
        print(f"- Features selecionadas: {len(selected_univariate)}/{len(X_clean.columns)}")
    else:
        X_filtered = X_clean
        print("\n[3] Target não fornecido - Pulando seleção univariada")
    
    # 6. Redução de redundância
    corr_matrix = X_filtered.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
    cols_correlacionadas = [col for col in upper.columns if any(upper[col] > threshold_correlacao)]
    X_low_corr = X_filtered.drop(columns=cols_correlacionadas)
    
    # Atualiza relatório
    for col in X_filtered.columns:
        if col in cols_correlacionadas:
            correlacao_max = upper[col].max()
            relatorio.loc[len(relatorio)] = [col, 'correlação', 'Removida', f"{correlacao_max:.3f}"]
    
    print(f"\n[5] Redução de redundância:")
    print(f"- Threshold de correlação: {threshold_correlacao}")
    print(f"- Features removidas: {len(cols_correlacionadas)}")
    print(f"- Features restantes: {len(X_low_corr.columns)}")
    
    # 7. Validação temporal com PSI
    stable_features = []
    for col in X_low_corr.columns:
        psi = calcular_psi(
            baseline_data[col],
            corrente_data[col]
        )
        
        if psi < threshold_psi:
            stable_features.append(col)
            relatorio.loc[len(relatorio)] = [col, 'PSI', 'Estável', f"{psi:.4f}"]
        else:
            relatorio.loc[len(relatorio)] = [col, 'PSI', 'Instável', f"{psi:.4f}"]
    
    X_stable = X_low_corr[stable_features]
    print(f"\n[6] Validação temporal (PSI):")
    print(f"- Baseline: primeiros {meses_baseline} meses")
    print(f"- Threshold PSI: {threshold_psi}")
    print(f"- Features estáveis: {len(stable_features)}")
    print(f"- Features instáveis: {len(X_low_corr.columns)-len(stable_features)}")
    
    # 8. Seleção por modelo (se target fornecido)
    selected_final = stable_features
    if col_target:
        print("\n[7] Seleção por modelo XGBoost:")
        
        scale_pos_weight = None
        if is_desbalanceado:
            scale_pos_weight = counts.get(0, 1) / counts.get(1, 1)
            print(f"- Ajuste para desbalanceamento (scale_pos_weight={scale_pos_weight:.2f})")
        
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
        importance_mean = np.mean(importance)
        selected_final = [col for col, imp in zip(X_stable.columns, importance) if imp > importance_mean]
        
        # Atualiza relatório
        for col in X_stable.columns:
            status = "Selecionada" if col in selected_final else "Removida"
            relatorio.loc[len(relatorio)] = [col, 'XGBoost', status, f"{importance[i]:.4f} (mean={importance_mean:.4f})"]
        
        print(f"- Features selecionadas: {len(selected_final)}/{len(X_stable.columns)}")
        print(f"- Threshold de importância: > {importance_mean:.4f}")
    else:
        print("\n[7] Target não fornecido - Pulando seleção por modelo")
    
    print("\n" + "="*60)
    print(" PROCESSAMENTO CONCLUÍDO ".center(60, "="))
    print("="*60)
    
    # Filtra relatório para features que passaram por algum teste
    relatorio = relatorio[relatorio['testes'] != '']
    
    return selected_final, relatorio

# Exemplo de uso
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    
    # Simulação de dados (substitua pelo seu DataFrame real)
    data = {
        "data_mensal": pd.date_range(start='2023-01-01', periods=1000, freq='D'),
        "target": np.random.choice([0, 1], size=1000, p=[0.9, 0.1]),
        "feature1": np.random.normal(0, 1, 1000),
        "feature2": np.random.choice([1, 2, 3], size=1000, p=[0.9, 0.05, 0.05]),
        "feature3": np.random.uniform(0, 1, 1000),
        "feature4": np.random.choice([0, 1], size=1000, p=[0.5, 0.5]),
    }
    df_pd = pd.DataFrame(data)
    df_spark = spark.createDataFrame(df_pd)
    
    # Caso com target
    print("\nExecutando pipeline COM target:")
    features_com_target, relatorio_com_target = pipeline_completo_flexivel(
        df_spark=df_spark,
        col_data="data_mensal",
        col_target="target"
    )
    
    print("\nFeatures selecionadas (com target):")
    print(features_com_target)
    
    print("\nRelatório detalhado (com target):")
    print(relatorio_com_target)
    
    # Caso sem target
    print("\nExecutando pipeline SEM target:")
    features_sem_target, relatorio_sem_target = pipeline_completo_flexivel(
        df_spark=df_spark,
        col_data="data_mensal",
        col_target=None
    )
    
    print("\nFeatures selecionadas (sem target):")
    print(features_sem_target)
    
    print("\nRelatório detalhado (sem target):")
    print(relatorio_sem_target)

# Exemplo com parâmetros customizados
features, relatorio = pipeline_completo_flexivel(
    df_spark=df,
    col_data="data_mensal",
    col_target="target",
    threshold_missing=0.9,    # Mais rigoroso com missings
    threshold_moda=0.85,      # Mais rigoroso com moda
    threshold_correlacao=0.85,# Mais rigoroso com correlação
    meses_baseline=4          # Baseline maior para PSI
)
