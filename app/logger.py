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
    root_logger.setLevel(level)
    
    # Riduciamo la verbosità di alcune librerie esterne rumorose se ci fossero
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Utility per ottenere un logger con il nome specificato."""
    return logging.getLogger(name)
