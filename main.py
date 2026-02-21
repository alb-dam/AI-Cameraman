"""Entry point dell'applicazione AI-Cameraman."""

import logging

from config import config_run
from backend import Backend
from frontend import frontend_run

logger = logging.getLogger(__name__)


class AICameramanApp:
    """Classe principale che orchestra l'avvio dell'applicazione."""

    def __init__(self) -> None:
        """Inizializza la configurazione e il backend."""
        logger.info("Inizializzazione AI-Cameraman in corso...")
        self.config = config_run()
        self.backend = Backend(self.config)

    def run(self) -> None:
        """Lancia l'interfaccia grafica e avvia l'applicazione."""
        frontend_run(self.backend, self.config)


def main_run() -> None:
    """Entry point: istanzia e avvia l'applicazione principale."""
    app = AICameramanApp()
    app.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_run()
