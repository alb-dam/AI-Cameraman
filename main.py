"""Entry point dell'applicazione AI-Cameraman (Clean Architecture)."""

import logging

from app.logger import setup_logger
from config.settings import settings_run
from app.factory import AppFactory
from gui.main_window import main_window_run

# Inizializziamo subito il logging per tutta l'applicazione
setup_logger(level=logging.INFO)
logger = logging.getLogger(__name__)


class AICameramanApp:
    """Classe principale che istanzia e avvia l'applicazione."""

    def __init__(self) -> None:
        """Inizializza la configurazione e il controller."""
        logger.info("Inizializzazione AI-Cameraman in corso...")
        self.settings = settings_run()
        self.controller = AppFactory.create_controller(self.settings)

    def run(self) -> None:
        """Lancia l'interfaccia grafica avviando il main loop."""
        main_window_run(self.controller, self.settings)


def main_run() -> None:
    """Entry point: istanzia e avvia l'applicazione principale."""
    app = AICameramanApp()
    app.run()


if __name__ == "__main__":
    main_run()
