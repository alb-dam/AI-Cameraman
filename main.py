"""Entry point dell'applicazione AI-Cameraman (Clean Architecture)."""

import logging
import multiprocessing

from app.logger import setup_logger
from config.settings import settings_run
from app.factory import AppFactory
from gui.main_window import main_window_run

# Inizializziamo subito il logging per tutta l'applicazione
setup_logger(level=logging.INFO)


def main_run() -> None:
    """Entry point: istanzia e avvia l'applicazione principale."""
    settings = settings_run()
    controller = AppFactory.create_controller(settings)
    main_window_run(controller, settings)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main_run()
