"""Vision module: rileva e traccia giocatori e pallone.

Architettura interna:
    SimpleKalman    - Filtro di Kalman 2D matematico di base.
    KalmanTracker   - Classifica e associa i rilevamenti frame-by-frame.
    Detector        - Façade che compone YoloDetector, KalmanTracker e centro d'azione.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable

from core.yolo_model import YoloDetector
from core.tracking import ActionCenterCalculator
from core.models import Detection, TrackedObject, DetectionResult


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
        self.time_since_update: int = 0

    def set_config(self, q_std: float, r_std: float) -> None:
        self.Q = np.eye(4) * q_std
        self.R = np.eye(2) * r_std

    def predict(self) -> Tuple[float, float]:
        self.time_since_update += 1
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, z_x: float, z_y: float) -> Tuple[float, float]:
        self.time_since_update = 0
        Z = np.array([[z_x], [z_y]])
        y = Z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Inversione analitica 2×2: ~5-10x più veloce di np.linalg.inv per matrici piccole
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        if abs(det) < 1e-12:
            det = 1e-12  # Fallback numerico per evitare divisione per zero
        S_inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
        K = np.dot(np.dot(self.P, self.H.T), S_inv)
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)
        return float(self.x[0, 0]), float(self.x[1, 0])


class KalmanTracker:
    """Classifica e traccia le bbox grezze usando multiple istanze Kalman."""

    MAX_MATCH_DISTANCE = 150.0

    def __init__(self, q_std: float = 20.0, r_std: float = 20.0) -> None:
        self.q_std: float = q_std
        self.r_std: float = r_std
        self.players_kalman: Dict[int, SimpleKalman] = {}
        self.ball_kalman: Optional[SimpleKalman] = None
        self.next_player_id: int = 0

    def set_config(self, q_std: float, r_std: float) -> None:
        self.q_std = q_std
        self.r_std = r_std
        if self.ball_kalman:
            self.ball_kalman.set_config(q_std, r_std)
        for k in self.players_kalman.values():
            k.set_config(q_std, r_std)

    def update(
        self,
        raw_players: List[Detection],
        raw_ball: Optional[Detection],
        predict_only: bool = False
    ) -> Tuple[List[TrackedObject], Optional[TrackedObject]]:
        self._predict_all_models()
        if predict_only:
            filtered_ball = self._update_ball_predict_only()
            filtered_players = self._update_players_predict_only()
        else:
            filtered_ball = self._update_ball(raw_ball)
            filtered_players = self._update_players(raw_players)
            
        return filtered_players, filtered_ball

    def _update_ball_predict_only(self) -> Optional[TrackedObject]:
        if self.ball_kalman:
            pred_x, pred_y = self.ball_kalman.x[0, 0], self.ball_kalman.x[1, 0]
            return TrackedObject(object_id=-1, center=(int(pred_x), int(pred_y)), raw_box=None)
        return None

    def _update_players_predict_only(self) -> List[TrackedObject]:
        filtered: List[TrackedObject] = []
        for pid, k in self.players_kalman.items():
            pred_x, pred_y = k.x[0, 0], k.x[1, 0]
            filtered.append(TrackedObject(object_id=pid, center=(int(pred_x), int(pred_y)), raw_box=None))
        return filtered

    def _predict_all_models(self) -> None:
        if self.ball_kalman:
            self.ball_kalman.predict()
        for k in self.players_kalman.values():
            k.predict()

    def _update_ball(self, raw_ball: Optional[Detection]) -> Optional[TrackedObject]:
        if raw_ball:
            cx, cy = raw_ball.center
            if self.ball_kalman is None:
                self.ball_kalman = SimpleKalman(cx, cy, q_std=self.q_std, r_std=self.r_std)
            up_x, up_y = self.ball_kalman.update(cx, cy)
            return TrackedObject(object_id=-1, center=(int(up_x), int(up_y)), raw_box=raw_ball.box)

        if self.ball_kalman:
            pred_x, pred_y = float(self.ball_kalman.x[0, 0]), float(self.ball_kalman.x[1, 0])
            return TrackedObject(object_id=-1, center=(int(pred_x), int(pred_y)), raw_box=None)

        return None

    def _update_players(self, raw_players: List[Detection]) -> List[TrackedObject]:
        filtered: List[TrackedObject] = []
        new_kalman: Dict[int, SimpleKalman] = {}

        for rp in raw_players:
            cx, cy = rp.center
            matched_id = self._match_nearest(cx, cy)

            if matched_id != -1:
                k = self.players_kalman.pop(matched_id)
                up_x, up_y = k.update(cx, cy)
                new_kalman[matched_id] = k
                filtered.append(TrackedObject(object_id=matched_id, center=(int(up_x), int(up_y)), raw_box=rp.box))
            else:
                k = SimpleKalman(cx, cy, q_std=self.q_std, r_std=self.r_std)
                new_kalman[self.next_player_id] = k
                filtered.append(TrackedObject(object_id=self.next_player_id, center=(cx, cy), raw_box=rp.box))
                self.next_player_id += 1

        for pid, k in self.players_kalman.items():
            if k.time_since_update < 30:
                new_kalman[pid] = k

        self.players_kalman = new_kalman
        return filtered

    def _match_nearest(self, cx: float, cy: float) -> int:
        best_id = -1
        best_dist = self.MAX_MATCH_DISTANCE

        for pid, k in self.players_kalman.items():
            kx, ky = float(k.x[0, 0]), float(k.x[1, 0])
            dist = float(np.hypot(cx - kx, cy - ky))
            if dist < best_dist:
                best_id, best_dist = pid, dist

        return best_id


class OutlierFilter:
    """Filtra giocatori outlier usando la Median Absolute Deviation (MAD).

    I giocatori la cui coordinata X è troppo distante dalla mediana del gruppo
    vengono esclusi dal calcolo del baricentro e dello spread, evitando che
    figure lontane (allenatore, raccattapalle) influenzino l'inquadratura.
    """

    DEFAULT_K = 3.0  # Moltiplicatore MAD: 3.0 ≈ ampio margine statistico

    @classmethod
    def filter_outliers(
        cls,
        players: List[TrackedObject],
        k: float = DEFAULT_K,
    ) -> List[TrackedObject]:
        """Rimuove outlier spaziali dalla lista giocatori.

        Algoritmo:
            1. Calcola mediana X di tutti i centri
            2. Calcola MAD = mediana(|xi - mediana|)
            3. Scarta i giocatori con |xi - mediana| > k * MAD
            4. Safeguard: se restano < 2 giocatori, ritorna la lista originale

        Args:
            players: giocatori tracciati.
            k: moltiplicatore per la soglia MAD.

        Returns:
            Lista filtrata (o originale se il filtro è troppo aggressivo).
        """
        if len(players) < 3:
            return players

        xs = np.array([p.center[0] for p in players], dtype=np.float64)
        median_x = float(np.median(xs))
        deviations = np.abs(xs - median_x)
        mad = float(np.median(deviations))

        # Se MAD ≈ 0, i giocatori sono quasi tutti nella stessa posizione X → nessun outlier
        if mad < 1.0:
            return players

        threshold = k * mad
        filtered = [p for p, dev in zip(players, deviations) if dev <= threshold]

        # Safeguard: non lasciare mai meno di 2 giocatori
        if len(filtered) < 2:
            return players

        return filtered


class Detector:
    """Façade che compone rilevamento, tracking e centro d'azione."""

    def __init__(self, model_name: str = "assets/yoloe-26s-seg.pt", yolo_imgsz: int = 640, debug: bool = False) -> None:
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

    def process(
        self,
        frame: np.ndarray,
        predict_only: bool = False,
        detection_filter: Optional[Callable[[List[Detection]], List[Detection]]] = None,
    ) -> DetectionResult:
        """Esegue rilevamento AI, tracking e computo del centro d'azione.

        Args:
            frame: frame BGR da analizzare.
            predict_only: se True, salta YOLO e usa solo predizione Kalman.
            detection_filter: callback opzionale per filtrare le detection grezze
                prima del tracking (es. filtro piedi ROI). Firma: (List[Detection]) -> List[Detection].
        """
        h, w = frame.shape[:2]
        frame_center = (w // 2, h // 2)

        if predict_only:
            raw_players: List[Detection] = []
            raw_ball = None
        else:
            raw_players, raw_ball = self.yolo.detect(frame, imgsz=self.yolo_imgsz)
            # Filtro post-detection (es. piedi dentro/fuori ROI)
            if detection_filter is not None:
                raw_players = detection_filter(raw_players)

        filtered_players, filtered_ball = self.tracker.update(raw_players, raw_ball, predict_only=predict_only)
        
        # Filtro outlier: rimuove giocatori spazialmente isolati (es. allenatore lontano)
        valid_players = OutlierFilter.filter_outliers(filtered_players)
        
        computed_center = self.center_calc.compute_center(valid_players, filtered_ball)
        if computed_center is not None:
            self.last_action_center = computed_center
        
        action_center = self.last_action_center if self.last_action_center is not None else frame_center
        
        current_spread = self._compute_player_spread(valid_players, w, h)
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
    def _compute_player_spread(players: List[TrackedObject], frame_width: int, frame_height: int) -> float:
        """Calcola la dimensione massima del bounding box che racchiude tutti i giocatori, come percentuale (0-1) della diagonale del frame."""
        if not players:
            return -1.0
        xs = [p.center[0] for p in players]
        ys = [p.center[1] for p in players]
        spread_px = float(max(max(xs) - min(xs), max(ys) - min(ys)))
        reference_length = float(np.hypot(frame_width, frame_height))
        if reference_length == 0:
             return 0.0
        return spread_px / reference_length


def detector_run(model_name: str = "assets/yoloe-26s-seg.pt", yolo_imgsz: int = 640, debug: bool = False) -> Detector:
    """Crea e ritorna un Detector inizializzato."""
    return Detector(model_name=model_name, yolo_imgsz=yolo_imgsz, debug=debug)
