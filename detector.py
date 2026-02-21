"""Rilevamento YOLO, tracking Kalman e calcolo centro d'azione.

Architettura interna:
    YoloDetector          – inferenza AI (YOLO), nessuno stato di tracking
    KalmanTracker         – gestione filtri di Kalman per giocatori e pallone
    ActionCenterCalculator – media ponderata del centro d'azione
    Detector              – façade pubblica che compone le tre classi
    DetectionResult       – dataclass per l'output strutturato
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.warning("Modulo ultralytics (YOLO) non trovato. AI disabilitata.")
    YOLO = None


# ── Output strutturato ──────────────────────────────────────────────────


@dataclass
class DetectionResult:
    """Risultato completo di un ciclo di rilevamento + tracking."""

    players: List[Dict[str, Any]] = field(default_factory=list)
    ball: Optional[Dict[str, Any]] = None
    action_center: Tuple[int, int] = (0, 0)
    player_spread: float = 0.0
    raw_players: List[Dict[str, Any]] = field(default_factory=list)
    raw_ball: Optional[Dict[str, Any]] = None


# ── Filtro di Kalman 2D ─────────────────────────────────────────────────


class SimpleKalman:
    """Filtro di Kalman 2D con preset di rumore configurabili."""

    def __init__(self, init_x: float, init_y: float, q_std: float = 20.0, r_std: float = 20.0) -> None:
        self.dt: float = 1.0 / 30.0
        self.x: np.ndarray = np.array([[init_x], [init_y], [0.0], [0.0]])

        self.F: np.ndarray = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.H: np.ndarray = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q: np.ndarray = np.eye(4)
        self.R: np.ndarray = np.eye(2)
        self.P: np.ndarray = np.eye(4) * 10.0

    def set_config(self, q_std: float, r_std: float) -> None:
        """Applica i nuovi valori di Process Noise (Q) e Measurement Noise (R)."""
        self.Q = np.eye(4) * q_std
        self.R = np.eye(2) * r_std

    def predict(self) -> Tuple[float, float]:
        """Predizione di stato per il prossimo step temporale."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, z_x: float, z_y: float) -> Tuple[float, float]:
        """Aggiornamento (correzione) dello stato con una nuova misurazione."""
        Z = np.array([[z_x], [z_y]])
        y = Z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)
        return float(self.x[0, 0]), float(self.x[1, 0])


# ── 1. Rilevamento AI ───────────────────────────────────────────────────


class YoloDetector:
    """Inferenza YOLO: riceve un frame, restituisce bounding box grezzi."""

    PLAYER_CLASS_ID = 0
    BALL_CLASS_ID = 32

    def __init__(self, model_name: str = "yolo26m.pt") -> None:
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

    # ── Private ──

    def _load_model(self, model_path: str) -> Any:
        """Tenta il caricamento del modello YOLO."""
        if YOLO is None:
            return None
        try:
            model = YOLO(model_path)
            import sys
            if sys.platform == "darwin":
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        model.to("mps")
                        logger.info(f"Modello {model_path} caricato su acceleratore hardware MPS.")
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


# ── 2. Tracking Kalman ──────────────────────────────────────────────────


class KalmanTracker:
    """Gestione dei filtri di Kalman per giocatori e pallone."""

    MAX_MATCH_DISTANCE = 150.0

    def __init__(self, q_std: float = 20.0, r_std: float = 20.0) -> None:
        self.q_std: float = q_std
        self.r_std: float = r_std
        self.players_kalman: Dict[int, SimpleKalman] = {}
        self.ball_kalman: Optional[SimpleKalman] = None
        self.next_player_id: int = 0

    def set_config(self, q_std: float, r_std: float) -> None:
        """Aggiorna i parametri di smoothing per tutti i tracker attivi."""
        self.q_std = q_std
        self.r_std = r_std
        if self.ball_kalman:
            self.ball_kalman.set_config(q_std, r_std)
        for k in self.players_kalman.values():
            k.set_config(q_std, r_std)

    def update(
        self,
        raw_players: List[Dict[str, Any]],
        raw_ball: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Associa le rilevazioni grezze ai tracker e ritorna posizioni filtrate."""
        filtered_ball = self._update_ball(raw_ball)
        filtered_players = self._update_players(raw_players)
        return filtered_players, filtered_ball

    # ── Private ──

    def _update_ball(self, raw_ball: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Aggiorna il tracking del singolo pallone."""
        if raw_ball:
            cx, cy = raw_ball["center"]
            if self.ball_kalman is None:
                self.ball_kalman = SimpleKalman(cx, cy, q_std=self.q_std, r_std=self.r_std)
            self.ball_kalman.predict()
            up_x, up_y = self.ball_kalman.update(cx, cy)
            return {"center": (int(up_x), int(up_y)), "raw_box": raw_ball["box"]}

        if self.ball_kalman:
            pred_x, pred_y = self.ball_kalman.predict()
            return {"center": (int(pred_x), int(pred_y)), "raw_box": None}

        return None

    def _update_players(self, raw_players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggiorna il tracking dei giocatori usando nearest-neighbor matching."""
        filtered: List[Dict[str, Any]] = []
        new_kalman: Dict[int, SimpleKalman] = {}

        for rp in raw_players:
            cx, cy = rp["center"]
            matched_id = self._match_nearest(cx, cy)

            if matched_id != -1:
                k = self.players_kalman.pop(matched_id)
                up_x, up_y = k.update(cx, cy)
                new_kalman[matched_id] = k
                filtered.append({"id": matched_id, "center": (int(up_x), int(up_y)), "raw_box": rp["box"]})
            else:
                k = SimpleKalman(cx, cy, q_std=self.q_std, r_std=self.r_std)
                new_kalman[self.next_player_id] = k
                filtered.append({"id": self.next_player_id, "center": (cx, cy), "raw_box": rp["box"]})
                self.next_player_id += 1

        self.players_kalman = new_kalman
        return filtered

    def _match_nearest(self, cx: float, cy: float) -> int:
        """Trova il tracker Kalman più vicino, entro la soglia massima."""
        best_id = -1
        best_dist = self.MAX_MATCH_DISTANCE

        for pid, k in self.players_kalman.items():
            kx, ky = k.predict()
            dist = float(np.hypot(cx - kx, cy - ky))
            if dist < best_dist:
                best_id, best_dist = pid, dist

        return best_id


# ── 3. Calcolo centro d'azione ───────────────────────────────────────────


class ActionCenterCalculator:
    """Calcola il centro d'azione come media ponderata giocatori + pallone."""

    PLAYER_WEIGHT = 1.0
    BALL_WEIGHT = 3.0

    @classmethod
    def compute_center(
        cls,
        players: List[Dict[str, Any]],
        ball: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[int, int]]:
        """Ritorna il baricentro ponderato dell'azione sul campo. Ritorna None se vuoto."""
        if not players and not ball:
            return None

        total_w = 0.0
        wx, wy = 0.0, 0.0

        for p in players:
            cx, cy = p["center"]
            wx += cx * cls.PLAYER_WEIGHT
            wy += cy * cls.PLAYER_WEIGHT
            total_w += cls.PLAYER_WEIGHT

        if ball:
            bx, by = ball["center"]
            wx += bx * cls.BALL_WEIGHT
            wy += by * cls.BALL_WEIGHT
            total_w += cls.BALL_WEIGHT

        if total_w > 0:
            return (int(wx / total_w), int(wy / total_w))
        return None


# ── Façade pubblica ──────────────────────────────────────────────────────


class Detector:
    """Façade che compone rilevamento, tracking e centro d'azione.

    API pubblica invariata rispetto alla versione precedente, ad eccezione di `set_config`:
        detector_run(frame) -> DetectionResult
        set_config(q_std, r_std)
    """

    def __init__(self, model_name: str = "yolo26n.pt", debug: bool = False) -> None:
        """Inizializza i tre sotto-moduli interni e lo stato locale."""
        self.yolo = YoloDetector(model_name)
        self.tracker = KalmanTracker()
        self.center_calc = ActionCenterCalculator()
        self.debug = debug
        self.last_action_center: Optional[Tuple[int, int]] = None
        self.last_player_spread: float = 0.0

    def set_config(self, q_std: float, r_std: float) -> None:
        """Aggiorna i parametri di smoothing per il tracker."""
        self.tracker.set_config(q_std, r_std)

    def process(self, frame: np.ndarray) -> DetectionResult:
        """Esegue rilevamento AI, tracking e computo del centro d'azione."""
        h, w = frame.shape[:2]
        frame_center = (w // 2, h // 2)

        raw_players, raw_ball = self.yolo.detect(frame)
        filtered_players, filtered_ball = self.tracker.update(raw_players, raw_ball)
        
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


def detector_run(model_name: str = "yolo26n.pt", debug: bool = False) -> Detector:
    """Crea e ritorna un Detector inizializzato."""
    return Detector(model_name=model_name, debug=debug)
