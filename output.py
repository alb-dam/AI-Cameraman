import cv2
import pyvirtualcam
import numpy as np
from PySide6.QtCore import QThread, Signal
from input import Video_Input

class Video_Thread(QThread):
    """Thread background per l'emissione del video verso GUI e OBS."""
    
    frame_pronto_gui = Signal(np.ndarray)
    log_sistema = Signal(str)

    def __init__(self, config_manager):
        super().__init__()
        self.config = config_manager
        self.in_esecuzione = True

    def _prepara_input(self):
        """Inizializza il generatore dal modulo input."""
        sorgente = self.config.ottieni_valore("sorgente", 0)
        w = self.config.ottieni_valore("larghezza_target", 1280)
        h = self.config.ottieni_valore("altezza_target", 720)
        video_in = Video_Input(sorgente, w, h)
        return video_in.input_run(), w, h

    def _invia_a_gui(self, frame: np.ndarray) -> np.ndarray:
        """Converte il frame in RGB e lo emette per la preview GUI."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_pronto_gui.emit(frame_rgb)
        return frame_rgb

    def _invia_a_obs(self, cam: pyvirtualcam.Camera, frame_rgb: np.ndarray) -> None:
        """Invia il frame alla Virtual Camera."""
        cam.send(frame_rgb)
        cam.sleep_until_next_frame()

    def output_run(self):
        """Metodo coordinatore del thread di output."""
        try:
            generatore_frame, w, h = self._prepara_input()
            fps = self.config.ottieni_valore("fps_target", 30)
            
            with pyvirtualcam.Camera(width=w, height=h, fps=fps) as cam:
                self.log_sistema.emit(f"Output avviato: GUI + OBS VirtualCam ({w}x{h})")
                
                for frame in generatore_frame:
                    if not self.in_esecuzione:
                        break
                    
                    # 1. Invio alla GUI (Preview automatica)
                    frame_rgb = self._invia_a_gui(frame)
                    
                    # 2. Invio a OBS
                    self._invia_a_obs(cam, frame_rgb)
                    
        except Exception as e:
            self.log_sistema.emit(f"Errore in Output: {e}")
        finally:
            self.log_sistema.emit("Flusso video interrotto.")

    def run(self):
        self.output_run()

    def ferma(self):
        self.in_esecuzione = False
        self.wait()