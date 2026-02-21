"""Logica pura per Area (Region of Interest) vettoriale e maschere di ritaglio.

Nessuno stato GUI, solo matematica e openCV.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from app.logger import get_logger

logger = get_logger(__name__)


class ROIMaskEngine:
    """Logica pura di maschera poligonale — nessuno stato."""

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
