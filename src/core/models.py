"""Modelli di dominio per la pipeline AI-Cameraman."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Detection:
    """Bbox grezzo prodotto dal modello AI."""
    box: Tuple[int, int, int, int]
    center: Tuple[int, int]
    conf: float


@dataclass
class TrackedObject:
    """Oggetto elaborato e stabilizzato dal tracking (Kalman)."""
    object_id: int
    center: Tuple[int, int]
    raw_box: Optional[Tuple[int, int, int, int]] = None


@dataclass
class CameraInstruction:
    """Istruzioni di regia generate dal Director."""
    cropped_frame: np.ndarray
    crop_box: Tuple[int, int, int, int]
    zoom_level: float
    smoothed_center: Tuple[int, int]


@dataclass
class ROI:
    """Regione di Interesse per maschere o ritaglio visivo."""
    polygon: Optional[np.ndarray] = None
    
    @property
    def is_valid(self) -> bool:
        """True se il poligono ha almeno 3 vertici."""
        return self.polygon is not None and len(self.polygon) >= 3


@dataclass
class FrameMetadata:
    """Metadati del loop corrente."""
    frame_id: int
    timestamp: float
    fps: float
    original_size: Tuple[int, int]


@dataclass
class DetectionResult:
    """Risultato completo di un ciclo di rilevamento + tracking."""
    players: List[TrackedObject] = field(default_factory=list)
    ball: Optional[TrackedObject] = None
    action_center: Tuple[int, int] = (0, 0)
    player_spread: float = 0.0
    raw_players: List[Detection] = field(default_factory=list)
    raw_ball: Optional[Detection] = None
