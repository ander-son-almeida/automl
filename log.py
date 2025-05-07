# logger_config.py
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = 'app_logger', 
                log_level: str = 'INFO', 
                log_file: str = None,
                console_log: bool = True) -> logging.Logger:
    """
    Configura um logger genérico e retorna a instância configurada.
    
    Args:
        name: Nome do logger
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Caminho para arquivo de log (opcional)
        console_log: Se deve logar no console
        
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
    
    # Handler para arquivo
    if log_file:
        # Cria diretório se não existir
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
