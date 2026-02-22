"""Gestione centralizzata della configurazione.

Utilizza una dataclass tipizzata per i settings e una classe 
manager per il loading/saving atomico del file JSON.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, asdict, fields

logger = logging.getLogger(__name__)


@dataclass
class AppSettings:
    """I settings dell'applicazione fortemente tipizzati."""
    
    source_type: str = "webcam"
    source_path: str = ""
    debug_mode: bool = False
    
    # Impostazioni Output indipendenti dall'input
    output_width: int = 1280
    output_height: int = 720
    output_fps: int = 30
    
    fixed_zoom_percent: float = 50.0
    dynamic_zoom_percent: float = 50.0
    kalman_preset_percent: float = 50.0
    
    kalman_q_smooth: float = 0.01
    kalman_r_smooth: float = 100.0
    kalman_q_reactive: float = 100.0
    kalman_r_reactive: float = 0.01
    
    last_roi_path: str = "roi.json"
    yolo_model: str = "assets/yolo26n.pt"
    yolo_imgsz: int = 640
    yolo_inference_interval: int = 3


class SettingsManager:
    """Gestione del ciclo di vita dei settings (load, save, get, set)."""

    def __init__(self, config_file: str = "config.json") -> None:
        self.config_file: str = config_file
        self.settings: AppSettings = AppSettings()
        self.load()

    # ── Lettura / Scrittura ─────────────────────────────────────────────

    def get(self, key: str) -> any:
        """Restituisce il valore del setting richiesto."""
        if not hasattr(self.settings, key):
            logger.warning("get() chiave sconosciuta: %s", key)
            return None
        return getattr(self.settings, key)

    def set(self, key: str, value: any, save_to_disk: bool = True) -> None:
        """Imposta un setting ed esegue opzionalmente il salvataggio su disco."""
        if not hasattr(self.settings, key):
            raise KeyError(f"Chiave non valida: {key}")
            
        # Trova il campo per verificare il tipo
        for field in fields(self.settings):
            if field.name == key:
                expected_type = field.type
                # Float accetta anche int, quindi un po' di dinamicità
                if expected_type == float and isinstance(value, int):
                    value = float(value)
                elif not isinstance(value, expected_type):
                    raise TypeError(f"{key}: atteso {expected_type}, ricevuto {type(value).__name__}")
                
                setattr(self.settings, key, value)
                if save_to_disk:
                    self.save()
                return

    # ── Persistenza ─────────────────────────────────────────────────────

    def load(self) -> None:
        """Carica la configurazione, gestendo fallback e chiavi obsolete."""
        if not os.path.exists(self.config_file):
            logger.info("File %s non trovato, creo con default.", self.config_file)
            self.settings = AppSettings()
            self.save()
            return

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Errore lettura %s: %s — ripristino default.", self.config_file, e)
            self.settings = AppSettings()
            self.save()
            return

        # Popoliamo la dataclass solo con i field conosciuti
        valid_keys = {f.name for f in fields(AppSettings)}
        stale_keys = [k for k in loaded if k not in valid_keys]
        
        for k in stale_keys:
            logger.info("Rimossa chiave obsoleta: %s", k)
            
        filtered_data = {k: v for k, v in loaded.items() if k in valid_keys}
        
        # Uniamo data caricata ai default della nuova istanza
        new_settings = AppSettings()
        for k, v in filtered_data.items():
            setattr(new_settings, k, v)
            
        self.settings = new_settings
        
        if stale_keys or len(filtered_data) < len(valid_keys):
            self.save()

    def save(self) -> None:
        """Salvataggio atomico per non corrompere il JSON in caso di crash."""
        dir_name = os.path.dirname(os.path.abspath(self.config_file))
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(asdict(self.settings), f, indent=4)
            os.replace(tmp_path, self.config_file)
        except OSError as e:
            logger.error("Errore salvataggio config: %s", e)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def reset(self, key: str = None) -> None:
        """Resetta un singolo valore al default, o tutto se key è None."""
        default_settings = AppSettings()
        if key is not None:
            if not hasattr(self.settings, key):
                raise KeyError(f"Chiave non valida: {key}")
            setattr(self.settings, key, getattr(default_settings, key))
        else:
            self.settings = default_settings
        self.save()


def settings_run(config_file: str = "config.json") -> SettingsManager:
    """Entry point per l'inizializzazione dei settings."""
    return SettingsManager(config_file)
