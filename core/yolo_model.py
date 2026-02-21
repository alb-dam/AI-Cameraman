"""Inferenza YOLO isolata dal resto del codice.

Riceve un frame e restituisce bounding box grezzi. Non mantiene stato di tracking.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from app.logger import get_logger

logger = get_logger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.warning("Modulo ultralytics (YOLO) non trovato. AI disabilitata.")
    YOLO = None


class YoloDetector:
    """Inferenza YOLO: riceve un frame, restituisce bounding box grezzi."""

    PLAYER_CLASS_ID = 0
    BALL_CLASS_ID = 32

    def __init__(self, model_name: str = "assets/yolo26n.pt") -> None:
        self.model: Any = self._load_model(model_name)

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Esegue l'inferenza e ritorna (raw_players, raw_ball)."""
        if self.model is None:
            return [], None

        results = self.model.predict(source=frame, verbose=False,
                                     classes=[self.PLAYER_CLASS_ID, self.BALL_CLASS_ID])

        raw_players: List[Dict[str, Any]] = []
        raw_ball: Optional[Dict[str, Any]] = None

        for box in results[0].boxes:
            detection = self._parse_box(box)
            if detection is None:
                continue

            cls_id, entry = detection
            if cls_id == self.PLAYER_CLASS_ID:
                raw_players.append(entry)
            elif cls_id == self.BALL_CLASS_ID:
                if raw_ball is None or entry["conf"] > raw_ball["conf"]:
                    raw_ball = entry

        return raw_players, raw_ball

    def _load_model(self, model_path: str) -> Any:
        """Tenta il caricamento del modello YOLO e applica accelerazione hardware."""
        if YOLO is None:
            return None
        try:
            # Assumiamo asset spostato nella root / assets / ma il path passato 
            # conterrà se necessario la direcory. Di base usiamo il nome/path fornito.
            model = YOLO(model_path)
            import sys
            if sys.platform == "darwin":
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        model.to("mps")
                        logger.info(f"Modello caricato su acceleratore hardware MPS.")
                except ImportError:
                    pass
            return model
        except Exception as e:
            logger.error(f"Errore caricamento modello {model_path}: {e}")
            return None

    @staticmethod
    def _parse_box(box: Any) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Estrae classe, confidence e coordinate da un singolo box YOLO."""
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return cls_id, {"box": (x1, y1, x2, y2), "center": (cx, cy), "conf": conf}
