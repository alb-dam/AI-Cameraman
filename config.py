import json
import os
from typing import Any

class App_Config:
    """Gestisce la persistenza e lo stato di runtime dell'applicazione."""

    def __init__(self, file_path: str = "config.json"):
        self.file_path = file_path
        self.config_runtime = {}
        self.default_config = {
            "sorgente": None,
            "larghezza_target": 1280,
            "altezza_target": 720,
            "fps_target": 30,
            "in_elaborazione": False
        }

    def _crea_file_default(self) -> None:
        """Crea il file JSON con i valori di default se non esiste."""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.default_config, f, indent=4)

    def _carica_da_disco(self) -> None:
        """Legge il file JSON e lo carica nella variabile di runtime."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.config_runtime = json.load(f)

    def _salva_su_disco(self) -> None:
        """Scrive lo stato di runtime sul file JSON."""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.config_runtime, f, indent=4)

    def aggiorna_valore(self, chiave: str, valore: Any) -> None:
        """Funzione atomica per aggiornare un valore in runtime e su disco."""
        if not self.config_runtime:
            self.config_run()
        self.config_runtime[chiave] = valore
        self._salva_su_disco()

    def ottieni_valore(self, chiave: str, default: Any = None) -> Any:
        """Restituisce un valore dallo stato di runtime."""
        return self.config_runtime.get(chiave, default)

    def config_run(self) -> dict:
        """Metodo coordinatore per l'inizializzazione della configurazione."""
        if not os.path.exists(self.file_path):
            self._crea_file_default()
        self._carica_da_disco()
        return self.config_runtime