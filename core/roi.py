"""Gestione unificata della Region of Interest (ROI).

Combina caricamento/salvataggio file, stato di editing (per la GUI),
e logica matematica vettoriale per le maschere di ritaglio.
"""

import json
import os
from typing import List, Tuple, Optional

import numpy as np

from app.logger import get_logger
from core.geometry import GeometryService
from core.models import ROI

logger = get_logger(__name__)


class ROIManager:
    """Coordinatore ROI: persistenza, editing, maschere vettoriali."""

    def __init__(self) -> None:
        self.roi: ROI = ROI()
        self.roi_points: List[Tuple[float, float]] = []
        self.editing_mode: bool = False
        self._cached_mask: Optional[np.ndarray] = None
        self._cached_mask_size: Optional[Tuple[int, int]] = None  # (w, h)

    @property
    def is_valid(self) -> bool:
        """True se il poligono ha almeno 3 vertici ed è stato caricato."""
        return self.roi.is_valid and self.roi.polygon is not None

    # ── Persistenza ─────────────────────────────────────────────────────

    def load_roi(self, filepath: str) -> bool:
        """Carica punti normalizzati da file JSON e aggiorna il mask engine."""
        if not os.path.exists(filepath):
            logger.warning("File ROI non trovato: %s", filepath)
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Errore lettura ROI da %s: %s", filepath, e)
            return False

        if "points" not in data:
            logger.warning("Formato file ROI non valido: manca 'points' in %s", filepath)
            return False

        self.roi_points = [tuple(p) for p in data["points"]]
        self._update_polygon(self.roi_points)
        logger.info("ROI caricata: %d punti da %s", len(self.roi_points), filepath)
        return True

    def save_roi(self, filepath: str) -> bool:
        """Salva i punti ROI normalizzati in un file JSON."""
        try:
            data = {"normalized": True, "points": self.roi_points}
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info("ROI salvata in %s", filepath)
            return True
        except OSError as e:
            logger.error("Errore salvataggio ROI in %s: %s", filepath, e)
            return False

    # ── Creazione da punti (API diretta) ────────────────────────────────

    def create_from_points(self, points: List[Tuple[float, float]]) -> None:
        """Imposta la ROI da una lista e aggiorna il mask engine."""
        self.roi_points = list(points)
        self._update_polygon(self.roi_points)

    def _update_polygon(self, points: List[Tuple[float, float]]) -> None:
        """Aggiorna la shape matematica del poligono e invalida la cache della maschera."""
        self._cached_mask = None
        self._cached_mask_size = None
        if len(points) >= 3:
            self.roi.polygon = np.array(points, dtype=np.float32)
        else:
            self.roi.polygon = None
            logger.info("Punti insufficienti per creare un poligono (min 3). ROI disattivata.")

    # ── Editing State ───────────────────────────────────────────────────

    def start_roi_selection(self) -> None:
        """Reset punti e attiva modalità editing."""
        self.roi_points = []
        self.roi.polygon = None
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
        self._update_polygon(self.roi_points)
        logger.info("Modalità editing ROI terminata (%d punti).", len(self.roi_points))

    # ── OpenCV Applicazione ed Esposizione ──────────────────────────────

    def apply_roi(self, frame: np.ndarray) -> np.ndarray:
        """Applica maschera ROI al frame con cache (zero-alloc dopo il primo frame).
        Se in modalità editing o invalida, restituisce il frame invariato.
        """
        if self.editing_mode or not self.is_valid:
            return frame
        
        h, w = frame.shape[:2]
        current_size = (w, h)
        
        # Ricostruisci la cache solo se la dimensione del frame è cambiata o la cache è vuota
        if self._cached_mask is None or self._cached_mask_size != current_size:
            self._cached_mask = GeometryService.build_polygon_mask(self.roi.polygon, w, h)
            self._cached_mask_size = current_size
            logger.debug("ROI mask cache rigenerata per dimensione %dx%d", w, h)
        
        return GeometryService.apply_precomputed_mask(frame, self._cached_mask)

    def draw_roi(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> None:
        """Disegna il poligono ROI sul frame (solo se non in editing)."""
        if self.editing_mode or not self.is_valid:
            return
        GeometryService.draw_polygon(
            frame, self.roi.polygon, color=color, thickness=thickness
        )


def roi_manager_run(filepath: Optional[str] = None) -> ROIManager:
    """Entry point per la creazione del ROIManager."""
    rm = ROIManager()
    if filepath and os.path.exists(filepath):
        rm.load_roi(filepath)
    return rm
