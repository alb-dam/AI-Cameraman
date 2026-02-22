"""Servizi matematici e geometrici puri, senza alcuno stato interno."""

import cv2
import numpy as np
from typing import Tuple, Optional


class GeometryService:
    """Funzioni pure per calcoli geometrici e maschere."""

    @staticmethod
    def calculate_crop_region(
        center: Tuple[int, int], zoom: float, frame_w: int, frame_h: int
    ) -> Tuple[int, int, int, int]:
        """Genera le coordinate di ritaglio (x1, y1, x2, y2)."""
        crop_w = int(frame_w / zoom)
        crop_h = int(frame_h / zoom)
        cx, cy = center

        x1 = cx - crop_w // 2
        y1 = cy - crop_h // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        return GeometryService.clamp_to_frame(x1, y1, x2, y2, crop_w, crop_h, frame_w, frame_h)

    @staticmethod
    def clamp_to_frame(
        x1: int, y1: int, x2: int, y2: int,
        crop_w: int, crop_h: int, frame_w: int, frame_h: int
    ) -> Tuple[int, int, int, int]:
        """Clampa le coordinate ai bordi del frame, preservando la dimensione del crop."""
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x1 + crop_w)
        y2 = min(frame_h, y1 + crop_h)

        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)

        return (x1, y1, x2, y2)

    @staticmethod
    def apply_polygon_mask(frame: np.ndarray, polygon: Optional[np.ndarray]) -> np.ndarray:
        """Applica maschera nera fuori dal poligono. Ritorna il frame invariato se il poligono è nullo."""
        if polygon is None:
            return frame

        h, w = frame.shape[:2]
        abs_points = (polygon * [w, h]).astype(np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [abs_points], 255)

        return cv2.bitwise_and(frame, frame, mask=mask)

    @staticmethod
    def draw_polygon(
        frame: np.ndarray,
        polygon: Optional[np.ndarray],
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> None:
        """Disegna il poligono sul frame (in-place)."""
        if polygon is None:
            return

        h, w = frame.shape[:2]
        abs_points = (polygon * [w, h]).astype(np.int32)
        cv2.polylines(frame, [abs_points], isClosed=True, color=color, thickness=thickness)
