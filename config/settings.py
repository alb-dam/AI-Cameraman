"""Gestione centralizzata della configurazione.

Utilizza una dataclass tipizzata per i settings e una classe 
manager per il loading/saving atomico del file JSON.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, asdict, fields
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AppSettings:
    """I settings dell'applicazione fortemente tipizzati."""
    
    source_type: str = "webcam"
    source_path: str = ""
    debug_mode: bool = False
    enable_performance_monitor: bool = False
    
    # Impostazioni Output indipendenti dall'input
    output_width: int = 1280
    output_height: int = 720
    output_fps: int = 30
    
    fixed_zoom_percent: float = 25.0
    dynamic_zoom_percent: float = 50.0
    kalman_preset_percent: float = 100.0
    
    kalman_q_smooth: float = 0.01
    kalman_r_smooth: float = 100.0
    kalman_q_reactive: float = 100.0
    kalman_r_reactive: float = 0.01
    
    last_roi_path: str = "roi.json"
    
    # Nomi delle sorgenti NDI
    ndi_ai_name: str = "AI-Cameraman AI"
    ndi_native_name: str = "AI-Cameraman Native"
    yolo_model: str = "assets/yolo26s.pt"
    yolo_imgsz: int = 640
    yolo_inference_interval: int = 6
    
    # Spread massimo dei giocatori prima che il bonus dinamico venga annullato (espresso in percentuale 0-1 basata sulla diagonale del frame).
    # Valori più alti (es. 0.8) = la camera attende che i giocatori siano molto più lontani prima di fare zoom out.
    director_max_spread: float = 1.0
    
    # Moltiplicatore massimo per il bonus di zoom dinamico.
    # Valori più bassi (es. 0.3) = zoom dinamico più dolce. Valori ad. 1.0 = zoom aggressivo. (Due corrisponde ad un 3x)
    director_dynamic_scale: float = 1.5
    
    # Fattore di addolcimento (smoothing) per i cambi di zoom.
    # Valori più bassi (es. 0.01) = transizioni di zoom lentissime. Valori verso 1.0 = zoom istantaneo.
    director_zoom_smoothing: float = 0.1
    
    # Tolleranza (deadzone) sui cambi di zoom prima di applicarli. 
    # 0.1 significa ignorare variazioni inferiori al 10%. Aiuta ad evitare l'effetto "zoom hunting".
    director_zoom_deadzone: float = 0.2
    
    # Deadzone spaziale (in percentuale) per il Pan/Tilt. Questo è il valore *base* applicato allo zoom 1x.
    # Man mano che la camera zooma, questa percentuale viene dinamicamente ridotta per reagire ai
    # micro-movimenti in modo più preciso senza generare salti bruschi.
    director_pan_tilt_deadzone: float = 0.2
    
    # Fattore di addolcimento (smoothing) direzionale (Pan e Tilt). Questo è il valore *base* allo zoom 1x.
    # Quando la camera zooma, lo smoothing viene dinamicamente diminuito (movimenti più lenti e smorzati) 
    # per evitare l'effetto "mal di mare". Valori base più bassi (es. 0.01) = movimenti ampi e cinematografici.
    director_pan_tilt_smoothing: float = 0.1


class SettingsManager:
    """Gestione del ciclo di vita dei settings (load, save, get, set)."""

    def __init__(self, config_file: str = "config.json") -> None:
        self.config_file: str = config_file
        self.settings: AppSettings = AppSettings()
        self._config_version: int = 0  # Incrementato ad ogni set() per dirty-flag
        # Lookup precompilato {nome: tipo} per O(1) nel metodo set()
        self._field_types: dict = {f.name: f.type for f in fields(AppSettings)}
        self.load()

    # ── Lettura / Scrittura ─────────────────────────────────────────────

    def get_version(self) -> int:
        """Ritorna la versione corrente della configurazione (dirty-flag pubblico)."""
        return self._config_version

    def get(self, key: str) -> any:
        """Restituisce il valore del setting richiesto."""
        if not hasattr(self.settings, key):
            raise KeyError(f"Chiave non valida: {key}")
        return getattr(self.settings, key)

    def set(self, key: str, value: any, save_to_disk: bool = True) -> None:
        """Imposta un setting ed esegue opzionalmente il salvataggio su disco."""
        if key not in self._field_types:
            raise KeyError(f"Chiave non valida: {key}")

        expected_type = self._field_types[key]
        # Float accetta anche int
        if expected_type == float and isinstance(value, int):
            value = float(value)
        elif not isinstance(value, expected_type):
            raise TypeError(f"{key}: atteso {expected_type}, ricevuto {type(value).__name__}")

        setattr(self.settings, key, value)
        self._config_version += 1
        if save_to_disk:
            self.save()

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
        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(asdict(self.settings), f, indent=4)
            os.replace(tmp_path, self.config_file)
        except OSError as e:
            logger.error("Errore salvataggio config: %s", e)
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def reset(self, key: Optional[str] = None) -> None:
        """Resetta un singolo valore al default, o tutto se key è None."""
        default_settings = AppSettings()
        if key is not None:
            if key not in self._field_types:
                raise KeyError(f"Chiave non valida: {key}")
            setattr(self.settings, key, getattr(default_settings, key))
        else:
            self.settings = default_settings
        self.save()


def settings_run(config_file: str = "config.json") -> SettingsManager:
    """Entry point per l'inizializzazione dei settings."""
    return SettingsManager(config_file)
