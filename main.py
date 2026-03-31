"""Entry point dell'applicazione AI-Cameraman (Clean Architecture)."""

import logging
import multiprocessing
import os
import sys

# Aggiunge src/ al path per risolvere i pacchetti interni
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

sys.path.insert(0, os.path.join(base_path, 'src'))

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
