import cv2
from typing import Generator, Any

class Video_Input:
    """Acquisisce il flusso video e lo cede come generatore."""

    def __init__(self, sorgente: Any, width: int, height: int):
        self.sorgente = sorgente
        self.width = width
        self.height = height

    def _apri_sorgente(self) -> cv2.VideoCapture:
        """Inizializza l'oggetto VideoCapture."""
        backend = cv2.CAP_DSHOW if isinstance(self.sorgente, int) else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.sorgente, backend)
        if not cap.isOpened():
            raise ValueError(f"Impossibile aprire la sorgente {self.sorgente}")
        return cap

    def _forza_risoluzione(self, cap: cv2.VideoCapture) -> None:
        """Imposta la risoluzione hardware desiderata."""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def _estrai_frame(self, cap: cv2.VideoCapture) -> Any:
        """Legge e ridimensiona un singolo frame."""
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

    def input_run(self) -> Generator[Any, None, None]:
        """Metodo coordinatore: cede sequenzialmente i frame catturati."""
        cap = self._apri_sorgente()
        if isinstance(self.sorgente, int):
            self._forza_risoluzione(cap)
        
        try:
            while True:
                frame = self._estrai_frame(cap)
                if frame is None:
                    break
                yield frame
        finally:
            cap.release()