import math

def calcular_tamanho_amostra(margem_erro, nivel_confianca, populacao_total=None):
    """
    Calcula o tamanho da amostra com base na margem de erro, nível de confiança e população total (opcional).
    
    Args:
        margem_erro (float): Margem de erro desejada (ex.: 0.05 para 5%).
        nivel_confianca (float): Nível de confiança desejado (ex.: 95 para 95%).
        populacao_total (int, optional): Tamanho total da população. Se None, assume população infinita.
    
    Returns:
        int: Tamanho mínimo da amostra necessário.
    """
    # Valor Z para os níveis de confiança mais comuns (90%, 95%, 99%)
    z_scores = {
        90: 1.645,
        95: 1.96,
        99: 2.576
    }
    
    # Verifica se o nível de confiança é válido
    if nivel_confianca not in z_scores:
        raise ValueError("Nível de confiança deve ser 90, 95 ou 99.")
    
    z = z_scores[nivel_confianca]
    p = 0.5  # Proporção conservadora (maximiza a amostra)
    e = margem_erro
    
    # Cálculo inicial (população infinita)
    n = (z ** 2) * p * (1 - p) / (e ** 2)
    
    # Ajuste para população finita (se aplicável)
    if populacao_total is not None:
        n = n / (1 + (n - 1) / populacao_total)
    
    return math.ceil(n)  # Arredonda para cima

# Exemplo de uso:
margem_erro = 0.05  # 5%
nivel_confianca = 95  # 95%
populacao_total = 1000  # Opcional (None para população infinita)

tamanho_amostra = calcular_tamanho_amostra(margem_erro, nivel_confianca, populacao_total)
print(f"Tamanho mínimo da amostra: {tamanho_amostra}")