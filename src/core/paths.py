"""Gestione centralizzata e sicura dei percorsi del progetto."""

import os
import sys
import logging

def get_base_path() -> str:
    """Restituisce il percorso di base dell'applicazione per dati, log e file temporanei.
    
    In sviluppo: risale alla root del progetto (fuori da src/).
    In PyInstaller (.app macOS): '.app/Contents/MacOS'.
    Inoltre, su macOS, crea symlink dentro '.app/Contents/Resources' affinché 
    le cartelle 'tmp', 'logs' e 'assets' siano pulite e facilmente accessibili.
    """
    if getattr(sys, 'frozen', False):
        # Cartella con eseguibile (e dove pyinstaller estrarrà/manterrà assests in onedir)
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        
        # Se siamo in macOS bundle (.app/Contents/MacOS)
        if base.endswith(os.path.join("Contents", "MacOS")):
            resources_dir = os.path.join(os.path.dirname(base), "Resources")
            os.makedirs(resources_dir, exist_ok=True)
            
            # Crea dei symlink puliti in Contents/Resources per facile accessibilità
            for folder in ["assets", "tmp", "logs"]:
                src = os.path.join(base, folder)
                dst = os.path.join(resources_dir, folder)
                
                # Assicuriamoci che la cartella esista in MacOS prima di linkarla
                os.makedirs(src, exist_ok=True)
                
                # Crea il symlink se non esiste già
                if not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except OSError as e:
                        # Ignora eventuali problemi di permessi silenziosamente
                        pass
        return base
    else:
        # Quando non è frozen, risale a root/ (due dir su rispetto a questo file)
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def get_assets_path() -> str:
    """Restituisce il percorso della cartella 'assets'."""
    path = os.path.join(get_base_path(), "assets")
    os.makedirs(path, exist_ok=True)
    return path

def get_tmp_path() -> str:
    """Restituisce il percorso della cartella 'tmp'."""
    path = os.path.join(get_base_path(), "tmp")
    os.makedirs(path, exist_ok=True)
    return path

def get_logs_path() -> str:
    """Restituisce il percorso della cartella 'logs'."""
    path = os.path.join(get_base_path(), "logs")
    os.makedirs(path, exist_ok=True)
    return path
