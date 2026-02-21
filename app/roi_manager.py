"""Coordinatore ROI: gestisce file, stato di editing e mask engine.

Nessuna conoscenza dell'interfaccia grafica o del formato dei frame.
"""

import json
import os
from typing import List, Tuple, Optional

import numpy as np

from app.logger import get_logger
from core.roi_engine import ROIMaskEngine

logger = get_logger(__name__)


class ROIManager:
    """Coordinatore ROI: persistenza, stato di editing, proxy per ROIMaskEngine."""

    def __init__(self) -> None:
        self.mask_engine: ROIMaskEngine = ROIMaskEngine()
        self.roi_points: List[Tuple[float, float]] = []
        self.editing_mode: bool = False

    # ── Persistenza ─────────────────────────────────────────────────────

    def load_roi(self, filepath: str) -> None:
        """Carica punti normalizzati da file JSON e aggiorna il mask engine."""
        if not os.path.exists(filepath):
            logger.warning("File ROI non trovato: %s", filepath)
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Errore lettura ROI da %s: %s", filepath, e)
            return

        if "points" not in data:
            logger.warning("Formato file ROI non valido: manca 'points' in %s", filepath)
            return

        self.roi_points = [tuple(p) for p in data["points"]]
        self.mask_engine.set_polygon(self.roi_points)
        logger.info("ROI caricata: %d punti da %s", len(self.roi_points), filepath)

    def save_roi(self, filepath: str) -> None:
        """Salva i punti ROI normalizzati in un file JSON."""
        try:
            data = {"normalized": True, "points": self.roi_points}
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info("ROI salvata in %s", filepath)
        except OSError as e:
            logger.error("Errore salvataggio ROI in %s: %s", filepath, e)

    # ── Creazione da punti (API diretta) ────────────────────────────────

    def create_from_points(self, points: List[Tuple[float, float]]) -> None:
        """Imposta la ROI da una lista e aggiorna il mask engine."""
        self.roi_points = list(points)
        self.mask_engine.set_polygon(self.roi_points)

    # ── Editing State ───────────────────────────────────────────────────

    def start_roi_selection(self) -> None:
        """Reset punti e attiva modalità editing."""
        self.roi_points = []
        self.mask_engine.clear()
        self.editing_mode = True
        logger.info("Modalità editing ROI avviata.")

    def add_point(self, norm_x: float, norm_y: float) -> None:
        """Aggiunge un punto normalizzato alla ROI (solo se in editing)."""
        if not self.editing_mode:
            return
        self.roi_points.append((norm_x, norm_y))
        logger.debug("ROI punto aggiunto: (%.3f, %.3f)", norm_x, norm_y)

    def finalize_roi(self) -> None:
        """Disattiva editing e aggiorna il mask engine con i punti raccolti."""
        if not self.editing_mode:
            return
        self.editing_mode = False
        self.mask_engine.set_polygon(self.roi_points)
        logger.info("Modalità editing ROI terminata (%d punti).", len(self.roi_points))

    # ── Proxy verso ROIMaskEngine ───────────────────────────────────────

    def apply_roi(self, frame: np.ndarray) -> np.ndarray:
        """Applica maschera ROI al frame.
        Se in modalità editing, restituisce il frame invariato.
        """
        if self.editing_mode:
            return frame
        return self.mask_engine.apply_mask(frame)

    def draw_roi(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> None:
        """Disegna il poligono ROI sul frame (solo se non in editing)."""
        if self.editing_mode:
            return
        self.mask_engine.draw(frame, color=color, thickness=thickness)


def roi_manager_run(filepath: Optional[str] = None) -> ROIManager:
    """Entry point per la creazione del ROIManager."""
    rm = ROIManager()
    if filepath and os.path.exists(filepath):
        rm.load_roi(filepath)
    return rm
