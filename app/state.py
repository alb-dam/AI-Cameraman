"""Gestione dello stato di runtime per l'elaborazione video."""

import threading
from typing import Optional, Callable
import numpy as np


class RuntimeState:
    """Contiene lo stato mutabile globale condiviso tra i thread."""

    def __init__(self) -> None:
        self.is_running: bool = False
        self.is_outputting_to_obs: bool = False
        self.source_exhausted: bool = False
        
        self.frame_lock = threading.Lock()
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_obs_frame: Optional[np.ndarray] = None
        self.latest_debug_frame: Optional[np.ndarray] = None
        
        self.on_frame_ready: Optional[Callable[[np.ndarray], None]] = None
        self.on_log_message: Optional[Callable[[str], None]] = None
        
        self.frames_processed: int = 0
        self.start_time: float = 0.0
        self.yolo_frame_counter: int = 0
        self.current_fps: float = 0.0
        
        self.input_fps: float = 30.0
        self.output_fps: float = 30.0
        self.obs_width: int = 1920
        self.obs_height: int = 1080

    def reset_for_start(self, input_fps: float, output_fps: float, width: int, height: int, start_time: float) -> None:
        """Resetta lo stato per una nuova sessione di elaborazione."""
        self.is_running = True
        self.source_exhausted = False
        
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.obs_width = width
        self.obs_height = height
        
        with self.frame_lock:
            self.latest_raw_frame = None
            self.latest_obs_frame = None
            self.latest_debug_frame = None
            
        self.frames_processed = 0
        self.start_time = start_time

    def log(self, message: str) -> None:
        """Invia un messaggio di log tramite la callback se definita."""
        if self.on_log_message:
            self.on_log_message(message)
