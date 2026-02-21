"""Gestione centralizzata della configurazione tramite file JSON.

Validazione chiavi e tipo, logging strutturato, salvataggio atomico.
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)

# Tipo o tupla di tipi accettati per isinstance()
_TypeSpec = Union[Type, Tuple[Type, ...]]


class ConfigManager:
    """Gestione centralizzata della configurazione tramite file JSON."""

    VALID_KEYS: Dict[str, Dict[str, Any]] = {
        "source_type":           {"type": str,          "default": "webcam"},
        "source_path":           {"type": str,          "default": ""},
        "debug_mode":            {"type": bool,         "default": False},
        "fixed_zoom_percent":    {"type": (int, float), "default": 50},
        "dynamic_zoom_percent":  {"type": (int, float), "default": 50},
        "kalman_preset_percent": {"type": (int, float), "default": 50},
        "kalman_q_smooth":       {"type": (int, float), "default": 0.01},
        "kalman_r_smooth":       {"type": (int, float), "default": 100.0},
        "kalman_q_reactive":     {"type": (int, float), "default": 100.0},
        "kalman_r_reactive":     {"type": (int, float), "default": 0.01},
        "last_roi_path":         {"type": str,          "default": "roi.json"},
        "yolo_model":            {"type": str,          "default": "yolo26n.pt"},
    }

    def __init__(self, config_file: str = "config.json") -> None:
        """Inizializza il ConfigManager e carica la configurazione dal file."""
        self.config_file: str = config_file
        self.config: Dict[str, Any] = {}
        self.load()

    # ── Lettura / Scrittura ─────────────────────────────────────────────

    def get(self, key: str) -> Any:
        """Restituisce il valore per la chiave specificata.

        Ritorna il default se la chiave è valida ma non presente.
        Ritorna None con warning se la chiave è sconosciuta.
        """
        if key not in self.VALID_KEYS:
            logger.warning("get() chiave sconosciuta: %s", key)
            return None
        return self.config.get(key, self.VALID_KEYS[key]["default"])

    def set(self, key: str, value: Any) -> None:
        """Imposta un nuovo valore e salva su disco.

        Raises:
            KeyError: se la chiave non è in VALID_KEYS.
            TypeError: se il tipo del valore non corrisponde all'atteso.
        """
        if key not in self.VALID_KEYS:
            raise KeyError(f"Chiave non valida: {key}")

        expected: _TypeSpec = self.VALID_KEYS[key]["type"]
        if not isinstance(value, expected):
            raise TypeError(
                f"{key}: atteso {expected}, ricevuto {type(value).__name__}"
            )

        self.config[key] = value
        self.save()

    # ── Persistenza ─────────────────────────────────────────────────────

    def load(self) -> None:
        """Carica la configurazione dal file JSON.

        - Se il file non esiste, crea con i default.
        - Se il file è corrotto, ripristina i default.
        - Rimuove automaticamente chiavi non in VALID_KEYS.
        - Aggiunge chiavi mancanti con i rispettivi default.
        """
        if not os.path.exists(self.config_file):
            logger.info("File %s non trovato, creo con default.", self.config_file)
            self.config = self._defaults()
            self.save()
            return

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except json.JSONDecodeError as e:
            logger.error("JSON corrotto in %s: %s — ripristino default.", self.config_file, e)
            self.config = self._defaults()
            self.save()
            return
        except OSError as e:
            logger.error("Impossibile leggere %s: %s — ripristino default.", self.config_file, e)
            self.config = self._defaults()
            self.save()
            return

        # Rimuovi chiavi obsolete
        stale = [k for k in loaded if k not in self.VALID_KEYS]
        for k in stale:
            logger.info("Rimossa chiave obsoleta: %s", k)
            del loaded[k]

        # Aggiungi chiavi mancanti con default
        for key, spec in self.VALID_KEYS.items():
            if key not in loaded:
                loaded[key] = spec["default"]

        self.config = loaded

        if stale:
            self.save()

    def save(self) -> None:
        """Salva la configurazione su disco in modo atomico (write + rename)."""
        dir_name = os.path.dirname(os.path.abspath(self.config_file))
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
            os.replace(tmp_path, self.config_file)
        except OSError as e:
            logger.error("Errore salvataggio config: %s", e)
            # Cleanup file temporaneo se il replace è fallito
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ── Utilità ─────────────────────────────────────────────────────────

    def reset(self, key: Optional[str] = None) -> None:
        """Ripristina il default per una chiave, o per tutte se key è None."""
        if key is not None:
            if key not in self.VALID_KEYS:
                raise KeyError(f"Chiave non valida: {key}")
            self.config[key] = self.VALID_KEYS[key]["default"]
        else:
            self.config = self._defaults()
        self.save()

    def _defaults(self) -> Dict[str, Any]:
        """Ritorna una copia del dizionario default."""
        return {k: v["default"] for k, v in self.VALID_KEYS.items()}


def config_run(config_file: str = "config.json") -> ConfigManager:
    """Crea e ritorna un ConfigManager pronto all'uso."""
    return ConfigManager(config_file)
