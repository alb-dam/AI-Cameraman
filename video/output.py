"""Gestione dell'output video tramite dual NDI (Native + AI) e preview locale."""

import cv2
import numpy as np
from typing import Optional, Tuple

from video.ndi_output import NDISender
from app.logger import get_logger

logger = get_logger(__name__)


class VideoOutput:
    """Gestore dei due output NDI: Native (passthrough) e AI (elaborato)."""

    def __init__(self) -> None:
        """Inizializza il gestore output senza aprire i sender."""
        self.ndi_ai: Optional[NDISender] = None
        self.ndi_native: Optional[NDISender] = None
        self._native_enabled: bool = False

    def initialize_ndi(self, ai_name: str, native_name: str,
                       width: int, height: int, fps: int) -> None:
        """Inizializza entrambi i sender NDI.
        
        Args:
            ai_name: Nome NDI per la sorgente AI (es. "AI-Cameraman AI").
            native_name: Nome NDI per la sorgente nativa (es. "AI-Cameraman Native").
            width: Larghezza output in pixel.
            height: Altezza output in pixel.
            fps: Frame rate target.
        """
        self.close()

        self.ndi_ai = NDISender(ai_name, width, height, fps)
        if not self.ndi_ai.open():
            logger.error("Impossibile aprire il sender NDI AI.")
            self.ndi_ai = None

        self.ndi_native = NDISender(native_name, width, height, fps)
        if not self.ndi_native.open():
            logger.error("Impossibile aprire il sender NDI Native.")
            self.ndi_native = None

    def send_ai_frame(self, frame: np.ndarray) -> None:
        """Invia il frame elaborato dall'AI al sender NDI AI."""
        if self.ndi_ai is not None and frame is not None:
            self.ndi_ai.send_frame(frame)

    def send_native_frame(self, frame: np.ndarray) -> None:
        """Invia il frame raw/nativo al sender NDI Native (se abilitato)."""
        if self._native_enabled and self.ndi_native is not None and frame is not None:
            self.ndi_native.send_frame(frame)

    def set_native_enabled(self, enabled: bool) -> None:
        """Abilita o disabilita l'invio del frame nativo via NDI."""
        self._native_enabled = enabled
        state = "abilitato" if enabled else "disabilitato"
        logger.info(f"NDI Native output {state}.")

    @property
    def native_enabled(self) -> bool:
        """Ritorna True se l'output nativo è abilitato."""
        return self._native_enabled

    @staticmethod
    def resize_and_pad(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Letterbox del frame alla dimensione target (thread-safe, no cache)."""
        h, w = frame.shape[:2]
        tw, th = target_size

        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

        top = (th - nh) // 2
        bottom = th - nh - top
        left = (tw - nw) // 2
        right = tw - nw - left

        return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])

    def show_preview(self, frame: np.ndarray, window_name: str = "Local Preview") -> None:
        """Mostra la preview locale tramite finestra cv2 standard."""
        if frame is not None:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

    def close(self) -> None:
        """Chiude entrambi i sender NDI e le finestre cv2."""
        if self.ndi_ai is not None:
            self.ndi_ai.close()
            self.ndi_ai = None
        if self.ndi_native is not None:
            self.ndi_native.close()
            self.ndi_native = None
        cv2.destroyAllWindows()


def output_run() -> VideoOutput:
    """Crea un VideoOutput (i sender NDI verranno inizializzati dal controller)."""
    return VideoOutput()
