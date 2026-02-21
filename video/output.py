"""Gestione dell'output video inclusa virtual camera e preview locale."""

import cv2
import pyvirtualcam
import numpy as np
from typing import Optional

from app.logger import get_logger

logger = get_logger(__name__)


class VideoOutput:
    """Gestione dell'output video inclusa virtual camera e preview locale."""

    def __init__(self) -> None:
        """Inizializza il gestore output."""
        self.cam: Optional[pyvirtualcam.Camera] = None

    def initialize_virtual_camera(self, width: int, height: int, fps: int) -> None:
        """Inizializza la virtual camera per invio frame a OBS."""
        self.close()
        try:
            self.cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
            logger.info(f"Virtual camera attivata: {self.cam.device} ({width}x{height} @ {fps}fps)")
        except Exception as e:
            logger.error(f"Errore inizializzazione virtual camera: {e}")
            self.cam = None

    def send_frame(self, frame: np.ndarray) -> None:
        """Ridimensiona con letterbox e invia il frame a OBS (Virtual Camera)."""
        if self.cam is None or frame is None:
            return
            
        if frame.shape[:2] == (self.cam.height, self.cam.width):
            padded = frame
        else:
            padded = self._resize_and_pad(frame, (self.cam.width, self.cam.height))
            
        self.cam.send(padded)

    @staticmethod
    def _resize_and_pad(frame: np.ndarray, target_size: tuple) -> np.ndarray:
        """Letterbox forzato del frame per matchare aspect ratio target."""
        h, w = frame.shape[:2]
        tw, th = target_size

        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)

        x_offset = (tw - nw) // 2
        y_offset = (th - nh) // 2
        canvas[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized
        return canvas

    def show_preview(self, frame: np.ndarray, window_name: str = "Local Preview") -> None:
        """Mostra la preview locale tramite finestra cv2 standard."""
        if frame is not None:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

    def close(self) -> None:
        """Chiude la virtual camera e disattiva ogni finestra cv2."""
        if self.cam is not None:
            self.cam.close()
            self.cam = None
        cv2.destroyAllWindows()


def output_run(width: int = 1920, height: int = 1080, fps: int = 30) -> VideoOutput:
    """Crea un VideoOutput e inizializza la virtual camera."""
    vo = VideoOutput()
    vo.initialize_virtual_camera(width, height, fps)
    return vo
