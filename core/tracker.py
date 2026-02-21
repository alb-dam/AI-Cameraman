"""Gestione dei filtri di Kalman per giocatori e pallone."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


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
        self.Q = np.eye(4) * q_std
        self.R = np.eye(2) * r_std

    def predict(self) -> Tuple[float, float]:
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, z_x: float, z_y: float) -> Tuple[float, float]:
        Z = np.array([[z_x], [z_y]])
        y = Z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
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
        raw_players: List[Dict[str, Any]],
        raw_ball: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        filtered_ball = self._update_ball(raw_ball)
        filtered_players = self._update_players(raw_players)
        return filtered_players, filtered_ball

    def _update_ball(self, raw_ball: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
        best_id = -1
        best_dist = self.MAX_MATCH_DISTANCE

        for pid, k in self.players_kalman.items():
            kx, ky = k.predict()
            dist = float(np.hypot(cx - kx, cy - ky))
            if dist < best_dist:
                best_id, best_dist = pid, dist

        return best_id
