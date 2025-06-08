from pyspark.sql import DataFrame
from pyspark.sql.functions import col, regexp_replace, regexp_extract, trim, when, lit
from pyspark.sql.types import DoubleType
import re

def convert_all_numeric_columns(df: DataFrame, sample_size: int = 1000) -> DataFrame:
    """
    Converte todas as colunas string que contêm valores numéricos em qualquer formato para DoubleType.
    Inclui tratamento para: separadores de milhar, símbolos de moeda, notação científica, etc.
    Retorna o DataFrame convertido com log detalhado.
    """
    # Identificar todas as colunas string
    string_cols = [field.name for field in df.schema.fields 
                  if str(field.dataType) == 'StringType()']
    
    print(f"\n🔍 Iniciando análise de {len(string_cols)} colunas do tipo string...")
    
    conversion_report = []
    converted_cols = []
    problematic_cols = []
    
    for col_name in string_cols:
        print(f"\n📊 Analisando coluna: '{col_name}'")
        
        # Etapa 1: Verificar padrões numéricos
        sample = df.select(col_name).filter(col(col_name).isNotNull()).limit(sample_size).collect()
        total_non_null = len(sample)
        
        if total_non_null == 0:
            print("   - Coluna vazia ou todos valores nulos - ignorando")
            continue
        
        # Contadores para estatísticas
        patterns = {
            'standard': 0,       # 1234.56 ou 1,234.56
            'brazilian': 0,      # 1.234,56
            'currency': 0,       # $1,234.56 ou R$ 1.234,56
            'percentage': 0,     # 12.34%
            'scientific': 0,     # 1.23E+10
            'other_numeric': 0,  # Outros formatos numéricos
            'non_numeric': 0     # Valores não numéricos
        }
        
        examples = set()
        
        for row in sample:
            value = str(row[0]).strip()
            if len(examples) < 5:  # Coletar até 5 exemplos
                examples.add(value)
            
            # Testar todos os padrões numéricos conhecidos
            if re.match(r'^[-+]?[\d,]+(?:\.\d+)?$', value):  # 1,234.56
                patterns['standard'] += 1
            elif re.match(r'^[-+]?[\d\.]+(?:,\d+)?$', value):  # 1.234,56
                patterns['brazilian'] += 1
            elif re.match(r'^[$\u20AC\u00A3R]\s?[-+]?[\d,\.]+$', value):  # $, €, £, R$
                patterns['currency'] += 1
            elif re.match(r'^[-+]?[\d,\.]+%$', value):  # 12.34%
                patterns['percentage'] += 1
            elif re.match(r'^[-+]?[\d,\.]+[Ee][-+]?\d+$', value):  # 1.23E+10
                patterns['scientific'] += 1
            elif re.match(r'^[-+]?[\d]+$', value):  # Apenas dígitos
                patterns['other_numeric'] += 1
            else:
                patterns['non_numeric'] += 1
        
        # Calcular porcentagens
        numeric_total = sum(patterns.values()) - patterns['non_numeric']
        numeric_percentage = (numeric_total / total_non_null) * 100
        
        # Registrar estatísticas
        stats = {
            'coluna': col_name,
            'total': total_non_null,
            'numeric_percentage': numeric_percentage,
            'patterns': patterns,
            'examples': list(examples)
        }
        conversion_report.append(stats)
        
        print(f"   - Porcentagem de valores numéricos: {numeric_percentage:.2f}%")
        print(f"   - Formatos detectados:")
        for pattern, count in patterns.items():
            if count > 0:
                print(f"     • {pattern}: {count} ocorrências")
        print(f"   - Exemplos encontrados: {examples}")
        
        # Etapa 2: Tentar conversão se maioria for numérica
        if numeric_percentage > 70:  # Limiar de 70% de valores numéricos
            try:
                # Expressão complexa para cobrir todos os casos
                df = df.withColumn(
                    col_name,
                    when(col(col_name).isNull(), lit(None)).otherwise(
                    when(col(col_name).rlike(r'^[$\u20AC\u00A3R]'),  # Símbolos de moeda
                         regexp_replace(
                             regexp_replace(
                                 regexp_replace(trim(col(col_name)), 
                                 r'^[$\u20AC\u00A3R]\s?', ''),  # Remove símbolo
                             r'\.', ''),  # Remove separador de milhar
                         ',', '.').cast('double'))
                    .when(col(col_name).rlike(r'[\d]\.\d,\d'),  # Formato 1.234,56
                         regexp_replace(
                             regexp_replace(trim(col(col_name))),
                             r'\.', ''),  # Remove separador de milhar
                         ',', '.').cast('double'))
                    .when(col(col_name).rlike(r'[\d],\d\.\d'),  # Formato 1,234.56
                         regexp_replace(trim(col(col_name))), ',', '').cast('double'))
                    .when(col(col_name).rlike(r'%$'),  # Porcentagem
                         regexp_replace(trim(col(col_name))), '%', '').cast('double') / 100)
                    .when(col(col_name).rlike(r'[Ee]'),  # Notação científica
                         col(col_name).cast('double'))
                    .otherwise(  # Tentativa genérica
                         regexp_replace(
                             regexp_replace(trim(col(col_name))),
                             r'[^\d.-]', ''),  # Mantém apenas dígitos, ponto e sinal
                         ',', '.').cast('double'))
                )
                
                # Verificar se a conversão foi bem-sucedida
                null_count = df.filter(col(col_name).isNull()).count()
                total_count = df.count()
                
                if (null_count / total_count) < 0.3:  # Menos de 30% nulos
                    converted_cols.append(col_name)
                    print(f"   ✅ SUCCESS: Coluna '{col_name}' convertida para double")
                else:
                    problematic_cols.append(col_name)
                    print(f"   ⚠️ AVISO: Conversão de '{col_name}' gerou muitos nulos - revertendo")
                    df = df.withColumn(col_name, col(col_name).cast('string'))  # Reverte
            except Exception as e:
                problematic_cols.append(col_name)
                print(f"   ❌ ERRO ao converter '{col_name}': {str(e)}")
        else:
            print(f"   ⏩ IGNORADA: Coluna '{col_name}' tem muitos valores não numéricos")
    
    # Relatório final
    print("\n📝 RELATÓRIO FINAL:")
    print(f"• Colunas analisadas: {len(string_cols)}")
    print(f"• Colunas convertidas: {len(converted_cols)}")
    print(f"• Colunas problemáticas: {len(problematic_cols)}")
    
    if converted_cols:
        print("\n🔧 Colunas convertidas com sucesso:")
        for col_name in converted_cols:
            stats = next(item for item in conversion_report if item["coluna"] == col_name)
            print(f"   - {col_name} ({stats['numeric_percentage']:.2f}% numéricos)")
    
    if problematic_cols:
        print("\n⚠️ Colunas com problemas de conversão:")
        for col_name in problematic_cols:
            stats = next(item for item in conversion_report if item["coluna"] == col_name)
            print(f"   - {col_name} (Motivo: {stats.get('conversion_error', 'Padrão não reconhecido')})")
    
    return df

# Exemplo de uso:
# df = convert_all_numeric_columns(df)