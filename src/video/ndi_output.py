"""Wrapper NDI Sender basato su cyndilib per l'invio di frame video via NDI."""

import numpy as np
import cv2
from fractions import Fraction
from typing import Optional

from cyndilib import Sender, VideoSendFrame, FourCC  # type: ignore

from app.logger import get_logger

logger = get_logger(__name__)


class NDISender:
    """Singolo sender NDI: gestisce un canale video NDI con nome proprio."""

    def __init__(self, name: str, width: int, height: int, fps: int) -> None:
        """Crea e apre un sender NDI.
        
        Args:
            name: Nome NDI visibile sulla rete (es. "AI-Cameraman AI").
            width: Larghezza del frame in pixel.
            height: Altezza del frame in pixel.
            fps: Frame rate target.
        """
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self._sender: Optional[Sender] = None
        self._vf: Optional[VideoSendFrame] = None
        self._frame_buffer: Optional[bytearray] = None
        self._frame_view: Optional[memoryview] = None

    def open(self) -> bool:
        """Inizializza e apre il sender NDI. Ritorna True se OK."""
        try:
            self._sender = Sender(self.name)
            
            self._vf = VideoSendFrame()
            self._vf.set_resolution(self.width, self.height)
            self._vf.set_frame_rate(Fraction(self.fps, 1))
            self._vf.set_fourcc(FourCC.BGRA)
            
            self._sender.set_video_frame(self._vf)
            
            # Pre-alloca buffer per i dati frame BGRA
            frame_size = self._vf.get_data_size()
            self._frame_buffer = bytearray(frame_size)
            self._frame_view = memoryview(self._frame_buffer)
            
            self._sender.open()
            logger.info(f"NDI Sender '{self.name}' aperto ({self.width}x{self.height} @ {self.fps}fps)")
            return True
        except Exception as e:
            logger.error(f"Errore apertura NDI Sender '{self.name}': {e}")
            self._sender = None
            return False

    def send_frame(self, frame: np.ndarray) -> None:
        """Converte BGR→BGRA e invia il frame via NDI.
        
        Args:
            frame: Frame OpenCV in formato BGR (H, W, 3) uint8.
        """
        if self._sender is None or frame is None:
            return
        
        try:
            # Converti BGR (3ch) → BGRA (4ch) aggiungendo canale alfa
            if frame.shape[2] == 3:
                bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            else:
                bgra = frame
            
            # Copia i bytes nel buffer pre-allocato
            raw = bgra.tobytes()
            if self._frame_buffer is not None:
                self._frame_buffer[:len(raw)] = raw
            
            # Invio asincrono NDI (non blocca il thread)
            if self._frame_view is not None:
                self._sender.write_video_async(self._frame_view)
        except Exception as e:
            logger.error(f"Errore invio frame NDI '{self.name}': {e}")

    def close(self) -> None:
        """Chiude il sender NDI e libera le risorse."""
        if self._sender is not None:
            try:
                self._sender.close()
                logger.info(f"NDI Sender '{self.name}' chiuso.")
            except Exception as e:
                logger.error(f"Errore chiusura NDI Sender '{self.name}': {e}")
            finally:
                self._frame_view = None
                self._frame_buffer = None
                self._vf = None
                self._sender = None
