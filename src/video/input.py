"""Gestione dell'input video da webcam o file."""

import cv2
import platform
import sys
from typing import List, Tuple, Any, Optional, Union

from app.logger import get_logger

logger = get_logger(__name__)


class VideoInput:
    """Gestione dell'input video da webcam o file."""

    def __init__(self) -> None:
        """Inizializza la gestione dell'input senza avviare la cattura."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_source_type: Optional[str] = None
        self.current_source_value: Union[int, str, None] = None


    @staticmethod
    def get_available_cameras() -> List[str]:
        """Rileva le webcam disponibili su Windows e macOS."""
        sistema = platform.system()

        if sistema == 'Windows':
            try:
                from pygrabber.dshow_graph import FilterGraph  # type: ignore
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
        elif source_type == "file" and source_value:
            self.cap = cv2.VideoCapture(str(source_value))
        elif source_type == "srt":
            port = source_value if source_value else 9999

            srt_url = (
                f"srt://0.0.0.0:{port}"
                f"?mode=listener"
                f"&transtype=live"
                f"&latency=3000"
                f"&peerlatency=3000"
                f"&rcvbuf=16777216"
                f"&sndbuf=16777216"
                f"&pkt_size=1316"
                f"&tlpktdrop=0"
            )

            # Imposta le opzioni globali ffmpeg per evitare blocchi infiniti su OpenCV
            # timeout e rw_timeout in microsecondi a 4s (per superare latency=3000), listen_timeout a 1s
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;4000000|listen_timeout;1000000|rw_timeout;4000000"

            logger.info(f"Apertura sorgente SRT: {srt_url}")
            cap = cv2.VideoCapture(srt_url, cv2.CAP_FFMPEG)

            if isinstance(cap, cv2.VideoCapture) and cap.isOpened():
                # Minimizza la latenza lato OpenCV
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap = cap
            else:
                self.cap = cap
        else:
            raise ValueError("Configurazione sorgente non valida")

        if not self.cap.isOpened():
            raise RuntimeError(f"Impossibile aprire la sorgente '{source_type}'")

    def get_fps(self) -> float:
        """Ritorna gli FPS della sorgente. Default 30.0 se non disponibile."""
        if self.cap is None:
            return 30.0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:  # Check for NaN/invalid
            return 30.0
        return float(fps)

    def get_resolution(self) -> Tuple[int, int]:
        """Ritorna la risoluzione (width, height) della sorgente."""
        if self.cap is None:
            return 1920, 1080
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            return 1920, 1080
        return w, h

    def read_frame(self) -> Tuple[bool, Any]:
        """Legge un singolo frame dalla sorgente inizializzata."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

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
