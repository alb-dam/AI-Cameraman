"""Overlay di debug: disegno bounding box, tracking, crop e FPS.

Nessuna logica AI. Riceve dati strutturati e li visualizza sul frame.
"""

import cv2
import numpy as np
from typing import Dict, Any

from app.logger import get_logger

logger = get_logger(__name__)

from core.models import DetectionResult, CameraInstruction, FrameMetadata


class DebugOverlay:
    """Disegna tutte le informazioni di debug su un frame."""

    def draw(
        self,
        frame: np.ndarray,
        det_out: DetectionResult,
        dir_out: CameraInstruction,
        roi_manager: Any,
        metadata: FrameMetadata,
    ) -> None:
        """Applica tutti i layer di debug sul frame (in-place)."""
        roi_manager.draw_roi(frame, color=(0, 0, 255))
        self._draw_crop_box(frame, dir_out)
        self._draw_action_center(frame, det_out)
        self._draw_raw_detections(frame, det_out)
        self._draw_tracked_players(frame, det_out)
        self._draw_ball(frame, det_out)
        self._draw_fps(frame, metadata.fps)

    # ── Singoli layer ───────────────────────────────────────────────────

    @staticmethod
    def _draw_crop_box(frame: np.ndarray, dir_out: CameraInstruction) -> None:
        """Disegna il rettangolo di crop e il livello di zoom."""
        cx1, cy1, cx2, cy2 = dir_out.crop_box
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 3)
        cv2.putText(frame, f"ZOOM: {dir_out.zoom_level:.2f}x",
                    (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    @staticmethod
    def _draw_action_center(frame: np.ndarray, det_out: DetectionResult) -> None:
        """Disegna il centro d'azione."""
        cv2.circle(frame, det_out.action_center, 8, (0, 0, 255), -1)

    @staticmethod
    def _draw_raw_detections(frame: np.ndarray, det_out: DetectionResult) -> None:
        """Disegna le bounding box grezze dei giocatori."""
        for p in det_out.raw_players:
            px1, py1, px2, py2 = p.box
            cv2.rectangle(frame, (px1, py1), (px2, py2), (100, 100, 100), 1)

    @staticmethod
    def _draw_tracked_players(frame: np.ndarray, det_out: DetectionResult) -> None:
        """Disegna i giocatori tracciati con Kalman."""
        for p in det_out.players:
            if not p.raw_box:
                continue
            px1, py1, px2, py2 = p.raw_box
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.circle(frame, p.center, 4, (0, 255, 0), -1)

    @staticmethod
    def _draw_ball(frame: np.ndarray, det_out: DetectionResult) -> None:
        """Disegna il pallone tracciato."""
        if not det_out.ball:
            return
        b = det_out.ball
        if b.raw_box:
            bx1, by1, bx2, by2 = b.raw_box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
        cv2.circle(frame, b.center, 6, (0, 165, 255), -1)

    @staticmethod
    def _draw_fps(frame: np.ndarray, fps: float) -> None:
        """Disegna gli FPS nell'angolo in alto a sinistra."""
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def overlay_run() -> DebugOverlay:
    """Crea e ritorna un'istanza DebugOverlay pronta all'uso."""
    return DebugOverlay()
