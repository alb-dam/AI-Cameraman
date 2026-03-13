"""Gestione dell'input video da webcam o file."""

import cv2
import platform
import sys
import platform
import sys
import numpy as np
from typing import List, Tuple, Any, Optional, Union

from app.logger import get_logger

logger = get_logger(__name__)


class VideoInput:
    """Gestione dell'input video da webcam o file."""

    def __init__(self) -> None:
        self.cap: Optional[cv2.VideoCapture] = None
        
        self.current_source_type: Optional[str] = None
        self.current_source_value: Union[int, str, None] = None


    @staticmethod
    def get_available_cameras() -> List[str]:
        """Rileva le webcam disponibili su Windows e macOS."""
        sistema = platform.system()

        if sistema == 'Windows':
            try:
                from pygrabber.dshow_graph import FilterGraph
                devices: List[str] = FilterGraph().get_input_devices()  # type: ignore
                return devices
            except ImportError:
                logger.warning("Modulo pygrabber non trovato. Fallback a indici numerici base.")
                return ["Webcam 0", "Webcam 1", "Webcam 2"]
        elif sistema == 'Darwin':
            try:
                import AVFoundation  # type: ignore
                dispositivi = AVFoundation.AVCaptureDevice.devicesWithMediaType_('vide')
                return [disp.localizedName() for disp in dispositivi]
            except ImportError:
                logger.warning("Modulo AVFoundation (pyobjc) non trovato. Fallback a indici numerici.")
                return ["Webcam 0 (Mac)", "Webcam 1 (Mac)", "Webcam 2 (Mac)"]
        else:
            return ["Webcam Default", "Webcam 1"]

    def initialize_source(self, source_type: str, source_value: Union[int, str, None] = None) -> None:
        """Inizializza la sorgente video (webcam o file)."""
        self.release()
        self.current_source_type = source_type
        self.current_source_value = source_value


        if source_type == "webcam":
            cam_index = int(source_value) if source_value is not None else 0
            if sys.platform == "darwin":
                self.cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
            else:
                self.cap = cv2.VideoCapture(cam_index)
                
            if not self.cap.isOpened():
                raise RuntimeError(f"Impossibile aprire la webcam '{cam_index}'")
                
        elif source_type in ("file", "srt"):
            if source_type == "file" and source_value:
                url = str(source_value)
            elif source_type == "srt":
                port = source_value if source_value else 9999
                url = f"srt://0.0.0.0:{port}?mode=listener"
                logger.info(f"Apertura sorgente SRT: {url}")
            
            try:
                self.cap = cv2.VideoCapture(url)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Impossibile aprire la sorgente '{source_type}': {url}")
            except Exception as e:
                raise RuntimeError(f"Impossibile aprire la sorgente '{source_type}': {e}")

        else:
            raise ValueError("Configurazione sorgente non valida")

    def get_fps(self) -> float:
        """Ritorna gli FPS della sorgente. Default 30.0 se non disponibile."""
        if self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0 and fps == fps:
                return float(fps)
        return 30.0

    def get_resolution(self) -> Tuple[int, int]:
        """Ritorna la risoluzione (width, height) della sorgente."""
        if self.cap is not None:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return w, h
        return 1920, 1080

    def read_frame(self) -> Tuple[bool, Any, Optional[np.ndarray]]:
        """Legge un singolo frame dalla sorgente inizializzata e i relativi campioni audio.
        
        Returns:
            Tuple[bool, Any, Optional[np.ndarray]]: (successo, frame_bgr, audio_data)
        """
        if self.cap is not None:
            if not self.cap.isOpened():
                return False, None, None
            ret, frame = self.cap.read()
            return ret, frame, None
            
        return False, None, None

    def release(self) -> None:
        """Rilascia le risorse della sorgente video."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def reconnect(self) -> bool:
        """Tentativo di riconnessione all'ultima sorgente configurata."""
        if self.current_source_type is None:
            logger.error("Impossibile riconnettere: nessuna sorgente configurata.")
            return False
            
        logger.info(f"Tentativo di riconnessione a {self.current_source_type} ({self.current_source_value})...")
        try:
            self.initialize_source(self.current_source_type, self.current_source_value)
            return True
        except Exception as e:
            logger.error(f"Riconnessione fallita: {e}")
            return False


def input_run(source_type: str, source_value: Union[int, str, None] = None) -> VideoInput:
    """Crea un VideoInput e inizializza la sorgente."""
    vi = VideoInput()
    vi.initialize_source(source_type, source_value)
    return vi
