"""Gestione dell'Area (Region of Interest) vettoriale e maschere di ritaglio.

Separazione netta tra logica maschera (ROIMaskEngine) e coordinamento (ROIManager).
"""

import cv2
import json
import logging
import numpy as np
import os
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# ROIMaskEngine — Logica pura poligono / maschera (zero stato GUI)
# ═══════════════════════════════════════════════════════════════════════

class ROIMaskEngine:
    """Logica pura di maschera poligonale — nessuno stato GUI."""

    def __init__(self) -> None:
        self.polygon: Optional[np.ndarray] = None

    @property
    def is_valid(self) -> bool:
        """True se il poligono ha almeno 3 vertici."""
        return self.polygon is not None and len(self.polygon) >= 3

    def set_polygon(self, points: List[Tuple[float, float]]) -> None:
        """Imposta il poligono da una lista di punti normalizzati [0..1]."""
        if len(points) >= 3:
            self.polygon = np.array(points, dtype=np.float32)
        else:
            self.polygon = None
            logger.info("Punti insufficienti per creare un poligono (min 3). ROI disattivata.")

    def clear(self) -> None:
        """Rimuove il poligono corrente."""
        self.polygon = None

    def apply_mask(self, frame: np.ndarray) -> np.ndarray:
        """Applica maschera nera fuori area ROI. Ritorna il frame invariato se non valido."""
        if not self.is_valid:
            return frame

        h, w = frame.shape[:2]
        abs_points = (self.polygon * [w, h]).astype(np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [abs_points], 255)

        return cv2.bitwise_and(frame, frame, mask=mask)

    def draw(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> None:
        """Disegna il poligono ROI sul frame (in-place)."""
        if not self.is_valid:
            return

        h, w = frame.shape[:2]
        abs_points = (self.polygon * [w, h]).astype(np.int32)
        cv2.polylines(frame, [abs_points], isClosed=True, color=color, thickness=thickness)


# ═══════════════════════════════════════════════════════════════════════
# ROIManager — Coordinatore: load/save/editing, delega maschera
# ═══════════════════════════════════════════════════════════════════════

class ROIManager:
    """Coordinatore ROI: persistenza, editing GUI, e proxy verso ROIMaskEngine."""

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
        """Imposta la ROI da una lista di punti normalizzati e aggiorna il mask engine."""
        self.roi_points = list(points)
        self.mask_engine.set_polygon(self.roi_points)

    # ── Editing GUI ─────────────────────────────────────────────────────

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
        """Applica maschera ROI al frame (solo se non in editing e poligono valido)."""
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


def roi_run(filepath: Optional[str] = None) -> ROIManager:
    """Crea un ROIManager e carica opzionalmente un file ROI."""
    rm = ROIManager()
    if filepath and os.path.exists(filepath):
        rm.load_roi(filepath)
    return rm
