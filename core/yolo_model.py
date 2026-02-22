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

    def __init__(self, model_name: str = "assets/yolo26n.pt", ball_conf_thresh: float = 0.4) -> None:
        self.device, self.use_half = self._detect_device()
        self.model: Any = self._load_model(model_name)
        self.ball_conf_thresh: float = ball_conf_thresh

    @staticmethod
    def _detect_device() -> Tuple[str, bool]:
        """Detects the best available hardware device and whether to use half precision."""
        device = "cpu"
        use_half = False
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"
                use_half = True
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                use_half = True
        except ImportError:
            pass
        logger.info(f"YOLO configurato per usare device: {device}, precisione half: {use_half}")
        return device, use_half

    def detect(self, frame: np.ndarray, imgsz: int = 640) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Esegue l'inferenza e ritorna (raw_players, raw_ball)."""
        if self.model is None:
            return [], None

        results = self.model.predict(source=frame, imgsz=imgsz, verbose=False, half=self.use_half,
                                     device=self.device, classes=[self.PLAYER_CLASS_ID, self.BALL_CLASS_ID])

        raw_players: List[Dict[str, Any]] = []
        raw_ball: Optional[Dict[str, Any]] = None

        img_h, img_w = frame.shape[:2]
        # La palla non dovrebbe mai occupare più del 15% o 20% della dimensione minore dello schermo
        max_ball_dim = min(img_w, img_h) * 0.20

        for box in results[0].boxes:
            detection = self._parse_box(box)
            if detection is None:
                continue

            cls_id, entry = detection
            if cls_id == self.PLAYER_CLASS_ID:
                raw_players.append(entry)
            elif cls_id == self.BALL_CLASS_ID:
                if entry["conf"] < self.ball_conf_thresh:
                    continue
                
                # Verifica morfologica
                w = entry["box"][2] - entry["box"][0]
                h = entry["box"][3] - entry["box"][1]
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                # Rifiuta box troppo grandi o troppo schiacciati (hallucinazioni)
                if w > max_ball_dim or h > max_ball_dim or aspect_ratio > 3.0:
                    continue

                if raw_ball is None or entry["conf"] > raw_ball["conf"]:
                    raw_ball = entry

        return raw_players, raw_ball

    def _load_model(self, model_path: str) -> Any:
        """Tenta il caricamento del modello YOLO e applica accelerazione hardware."""
        if YOLO is None:
            return None
            
        import sys
        import os
        
        # Gestione export CoreML o ONNX platform-aware
        base_name, _ = os.path.splitext(model_path)
        optimized_path = model_path
        
        if sys.platform == "darwin":
            mlpackage_path = base_name + ".mlpackage"
            if os.path.exists(mlpackage_path):
                optimized_path = mlpackage_path
                logger.info(f"Trovato modello CoreML per macOS: {optimized_path}")
            elif os.path.exists(model_path):
                logger.info(f"Modello Ottimizzato non trovato. Esportazione automatica in CoreML per {model_path} in corso, attendere...")
                try:
                    temp_model = YOLO(model_path)
                    temp_model.export(format="coreml", nms=True)
                    if os.path.exists(mlpackage_path):
                        optimized_path = mlpackage_path
                        logger.info(f"Esportazione terminata. ORA Carico {optimized_path}!")
                except Exception as e:
                    logger.error(f"Errore durante esportazione automatica: {e}")
        else:
            onnx_path = base_name + ".onnx"
            engine_path = base_name + ".engine"
            if os.path.exists(onnx_path):
                optimized_path = onnx_path
                logger.info(f"Trovato modello ONNX per PC: {optimized_path}")
            elif os.path.exists(engine_path):
                optimized_path = engine_path
                logger.info(f"Trovato modello TensorRT per PC: {optimized_path}")
            elif os.path.exists(model_path):
                logger.info(f"Modello Ottimizzato non trovato. Esportazione automatica in ONNX per {model_path} in corso, attendere...")
                try:
                    temp_model = YOLO(model_path)
                    temp_model.export(format="onnx", opset=12, half=self.use_half, device=self.device)
                    if os.path.exists(onnx_path):
                        optimized_path = onnx_path
                        logger.info(f"Esportazione terminata. ORA Carico {optimized_path}!")
                except Exception as e:
                    logger.error(f"Errore durante esportazione automatica: {e}")
                
        try:
            model = YOLO(optimized_path)
            if optimized_path.endswith('.pt') and self.device != "cpu":
                try:
                    model.to(self.device)
                    logger.info(f"Modello caricato su acceleratore hardware {self.device}.")
                except Exception as e:
                    logger.warning(f"Impossibile spostare il modello su {self.device}: {e}")
            return model
        except Exception as e:
            logger.error(f"Errore caricamento modello {optimized_path}: {e}")
            return None

    @staticmethod
    def _parse_box(box: Any) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Estrae classe, confidence e coordinate da un singolo box YOLO."""
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return cls_id, {"box": (x1, y1, x2, y2), "center": (cx, cy), "conf": conf}
