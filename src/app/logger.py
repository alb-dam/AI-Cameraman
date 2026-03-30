"""Configurazione centralizzata del logging per l'applicazione."""

import logging
import sys


def setup_logger(level: int = logging.INFO) -> None:
    """Inizializza il logger di root per garantire coerenza su tutti i moduli."""
    
    # Rimuoviamo eventuali handler esistenti
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler per stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    
    # Handler per File (Rotating, max 5MB, 3 backup)
    try:
        from logging.handlers import RotatingFileHandler
        import os
        
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            
        logs_dir = os.path.join(base_path, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, "ai_cameraman.log")
        
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Attenzione: impossibile inizializzare i log su file. {e}")

    root_logger.setLevel(level)
    
    # Riduciamo la verbosità di alcune librerie esterne rumorose se ci fossero
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Utility per ottenere un logger con il nome specificato."""
    return logging.getLogger(name)
