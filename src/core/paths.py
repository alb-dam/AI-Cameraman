"""Gestione centralizzata e sicura dei percorsi del progetto."""

import os
import sys
import logging

def get_base_path() -> str:
    """Restituisce il percorso di base dell'applicazione per dati, log e file temporanei.
    
    In sviluppo: risale alla root del progetto (fuori da src/).
    In PyInstaller (.app macOS): '.app/Contents/MacOS'.
    Inoltre, su macOS, crea symlink dentro '.app/Contents/Data' affinché 
    le cartelle 'tmp', 'logs' e 'assets' siano pulite e facilmente accessibili.
    """
    if getattr(sys, 'frozen', False):
        # Cartella con eseguibile (e dove pyinstaller estrarrà/manterrà assests in onedir)
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        
        # Normalizza per evitare problemi con trailing slash (es. /Contents/MacOS/)
        base = os.path.normpath(base)
        
        # Se siamo in macOS bundle (.app/Contents/...)
        # PyInstaller può impostare _MEIPASS su Frameworks o MacOS.
        if "Contents/Frameworks" in base or "Contents/MacOS" in base:
            # Ricaviamo il path di Contents/
            contents_dir = os.path.dirname(base)
            if contents_dir.endswith("MacOS") or contents_dir.endswith("Frameworks"):
                contents_dir = os.path.dirname(contents_dir)
                
            data_dir = os.path.join(contents_dir, "Data")
            os.makedirs(data_dir, exist_ok=True)
            
            # La cartella dati reale di PyInstaller per i BUNDLE è Resources
            resources_dir = os.path.join(contents_dir, "Resources")
            os.makedirs(resources_dir, exist_ok=True)
            
            # Crea dei symlink puliti in Contents/Data verso Contents/Resources
            for folder in ["assets", "tmp", "logs"]:
                src = os.path.join(resources_dir, folder)
                dst = os.path.join(data_dir, folder)
                
                # Assicuriamoci che la cartella effettiva esista in Resources
                os.makedirs(src, exist_ok=True)
                
                # Crea il symlink se non esiste già
                if not os.path.exists(dst):
                    try:
                        # ATTENZIONE: Usa un path relativo perché il .app può essere spostato
                        # Da "Contents/Data/cartella" punta a "../Resources/cartella"
                        os.symlink(os.path.join("..", "Resources", folder), dst)
                    except OSError:
                        pass
                        
            # Forza la base a Resources. 
            # PyInstaller carica le librerie da Frameworks, ma i file creati dinamicamente
            # (come i log o i tmp) devono stare in Resources per rispettare i nostri symlink.
            base = resources_dir
            
        return base
    else:
        # Quando non è frozen, risale a root/Data/ (due dir su rispetto a questo file)
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        data_dir = os.path.join(root_dir, 'Data')
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

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
