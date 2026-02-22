"""Façade che compone rilevamento, tracking e centro d'azione."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from core.yolo_model import YoloDetector
from core.tracker import KalmanTracker
from core.action_center import ActionCenterCalculator


@dataclass
class DetectionResult:
    """Risultato completo di un ciclo di rilevamento + tracking."""

    players: List[Dict[str, Any]] = field(default_factory=list)
    ball: Optional[Dict[str, Any]] = None
    action_center: Tuple[int, int] = (0, 0)
    player_spread: float = 0.0
    raw_players: List[Dict[str, Any]] = field(default_factory=list)
    raw_ball: Optional[Dict[str, Any]] = None


class Detector:
    """Façade che compone rilevamento, tracking e centro d'azione."""

    def __init__(self, model_name: str = "assets/yolo26n.pt", yolo_imgsz: int = 640, debug: bool = False) -> None:
        """Inizializza i tre sotto-moduli interni e lo stato locale."""
        self.yolo = YoloDetector(model_name)
        self.yolo_imgsz = yolo_imgsz
        self.tracker = KalmanTracker()
        self.center_calc = ActionCenterCalculator()
        self.debug = debug
        self.last_action_center: Optional[Tuple[int, int]] = None
        self.last_player_spread: float = 0.0

    def set_config(self, q_std: float, r_std: float, yolo_imgsz: int = 640) -> None:
        """Aggiorna i parametri di smoothing per il tracker e imgsz per YOLO."""
        self.tracker.set_config(q_std, r_std)
        self.yolo_imgsz = yolo_imgsz

    def process(self, frame: np.ndarray, predict_only: bool = False) -> DetectionResult:
        """Esegue rilevamento AI, tracking e computo del centro d'azione."""
        h, w = frame.shape[:2]
        frame_center = (w // 2, h // 2)

        if predict_only:
            raw_players, raw_ball = [], None
        else:
            raw_players, raw_ball = self.yolo.detect(frame, imgsz=self.yolo_imgsz)
            
        filtered_players, filtered_ball = self.tracker.update(raw_players, raw_ball, predict_only=predict_only)
        
        computed_center = self.center_calc.compute_center(filtered_players, filtered_ball)
        if computed_center is not None:
            self.last_action_center = computed_center
        
        action_center = self.last_action_center if self.last_action_center is not None else frame_center
        
        current_spread = self._compute_player_spread(filtered_players)
        if current_spread >= 0.0:
            self.last_player_spread = current_spread
        spread = self.last_player_spread

        return DetectionResult(
            players=filtered_players,
            ball=filtered_ball,
            action_center=action_center,
            player_spread=spread,
            raw_players=raw_players,
            raw_ball=raw_ball,
        )

    @staticmethod
    def _compute_player_spread(players: List[Dict[str, Any]]) -> float:
        """Calcola la dimensione massima del bounding box che racchiude tutti i giocatori."""
        if not players:
            return -1.0
        xs = [p["center"][0] for p in players]
        ys = [p["center"][1] for p in players]
        return float(max(max(xs) - min(xs), max(ys) - min(ys)))


def detector_run(model_name: str = "assets/yolo26n.pt", yolo_imgsz: int = 640, debug: bool = False) -> Detector:
    """Crea e ritorna un Detector inizializzato."""
    return Detector(model_name=model_name, yolo_imgsz=yolo_imgsz, debug=debug)
