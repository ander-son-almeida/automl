import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str = 'app_logger',
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    console_log: bool = True,
    timestamp_in_filename: bool = True
) -> logging.Logger:
    """
    Configura um logger com arquivos nomeados por timestamp e mantém os parâmetros originais.
    
    Args:
        name: Nome do logger
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Caminho base para o arquivo de log (None para desativar)
        console_log: Se True, exibe logs no console
        timestamp_in_filename: Se True, adiciona timestamp ao nome do arquivo
        
    Returns:
        Instância de logger configurada
    """
    logger = logging.getLogger(name)
    
    # Remove handlers existentes para evitar duplicação
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Define o nível de log
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Formato padrão
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para console
    if console_log:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Handler para arquivo (se log_file foi fornecido)
    if log_file:
        # Adiciona timestamp ao nome do arquivo se solicitado
        if timestamp_in_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(log_file)
            log_file = f"{log_path.parent}/{log_path.stem}_{timestamp}{log_path.suffix}"
        
        # Cria diretório se não existir
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
