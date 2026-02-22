"""Regia virtuale: pan, tilt, zoom dinamico/fisso e ritaglio.

Architettura interna:
    ActionCenterCalculator - calcolo centro d'azione basato su rilevamenti
    ValueSmoother      – smoothing dello zoom (scalare)
    PointSmoother      – smoothing temporale del centro inquadrato (2D)
    CameraStrategy     – calcolo zoom fisso + dinamico
    Director           – façade pubblica che compone le classi precedenti

Nessuna dipendenza da YOLO o dal modulo detector.
Input richiesti: frame, action_center, player_spread.
"""

import numpy as np
from typing import List, Tuple, Optional

from app.logger import get_logger
from core.models import CameraInstruction, TrackedObject
from core.geometry import GeometryService

logger = get_logger(__name__)


class ActionCenterCalculator:
    """Calcola il centro d'azione come media ponderata giocatori + pallone."""

    PLAYER_WEIGHT = 1.0
    BALL_WEIGHT = 3.0

    @classmethod
    def compute_center(
        cls,
        players: List[TrackedObject],
        ball: Optional[TrackedObject],
    ) -> Optional[Tuple[int, int]]:
        """Ritorna il baricentro ponderato dell'azione sul campo. Ritorna None se vuoto."""
        if not players and not ball:
            return None

        total_w = 0.0
        wx, wy = 0.0, 0.0

        for p in players:
            cx, cy = p.center
            wx += cx * cls.PLAYER_WEIGHT
            wy += cy * cls.PLAYER_WEIGHT
            total_w += cls.PLAYER_WEIGHT

        if ball:
            bx, by = ball.center
            wx += bx * cls.BALL_WEIGHT
            wy += by * cls.BALL_WEIGHT
            total_w += cls.BALL_WEIGHT

        if total_w > 0:
            return (int(wx / total_w), int(wy / total_w))
        return None


class ValueSmoother:
    """Applica uno smoothing esponenziale a un valore scalare."""

    def __init__(self, smoothing_factor: float = 0.10, initial_value: float = 1.0) -> None:
        self.smoothing_factor = smoothing_factor
        self.current_value = initial_value

    def smooth(self, target_value: float) -> float:
        """Ammorbidisce la transizione verso il valore target."""
        self.current_value += (target_value - self.current_value) * self.smoothing_factor
        return self.current_value


class PointSmoother:
    """Applica uno smoothing esponenziale a una coordinata 2D."""

    def __init__(self, smoothing_factor: float = 0.05) -> None:
        self.smoothing_factor = smoothing_factor
        self.current_point: Optional[Tuple[float, float]] = None

    def smooth(self, target_point: Tuple[float, float]) -> Tuple[float, float]:
        """Ammorbidisce la transizione verso il punto target."""
        current = self.current_point
        if current is None:
            self.current_point = target_point
            return target_point

        cx, cy = current
        sx = cx + (target_point[0] - cx) * self.smoothing_factor
        sy = cy + (target_point[1] - cy) * self.smoothing_factor
        
        self.current_point = (sx, sy)
        return (sx, sy)


class DeadzoneFilter:
    """Filtro che ignora spostamenti inferiori a una certa soglia (deadzone) per annullare il micro-jitter."""

    def __init__(self, threshold: float = 20.0) -> None:
        self.threshold = threshold
        self.stable_point: Optional[Tuple[float, float]] = None

    def filter(self, target_point: Tuple[float, float]) -> Tuple[float, float]:
        """Se il punto si muove meno della soglia, ritorna il punto precedente."""
        stable = self.stable_point
        if stable is None:
            self.stable_point = target_point
            return target_point
            
        spx, spy = stable
        dx = target_point[0] - spx
        dy = target_point[1] - spy
        dist = float(np.hypot(dx, dy))
        
        if dist < self.threshold:
            return stable
            
        self.stable_point = target_point
        return target_point


class ScalarDeadzoneFilter:
    """Filtro che ignora variazioni scalari inferiori a una certa soglia (deadzone) per annullare il jitter (es. zoom hunting)."""

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold
        self.stable_value: Optional[float] = None

    def filter(self, target_value: float) -> float:
        """Se il valore varia meno della soglia, ritorna il valore precedente."""
        stable = self.stable_value
        if stable is None:
            self.stable_value = target_value
            return target_value
            
        diff = abs(target_value - stable)
        
        if diff < self.threshold:
            return stable
            
        self.stable_value = target_value
        return target_value


class CameraStrategy:
    """Strategia di calcolo dello zoom (fisso vs dinamico) dato uno spread di giocatori."""

    def __init__(self) -> None:
        self.fixed_zoom: float = 1.0         # 1.0x = nessun zoom
        self.dynamic_intensity: float = 0.0  # 0.0 = disattivato
        self._MAX_SPREAD: float = 1000.0     # spread massimo (più alto = attesa prima di zoomare)
        self._DYNAMIC_SCALE: float = 1.0     # moltiplicatore massimo del bonus dinamico

    def set_config(self, fixed_zoom_percent: float, dynamic_zoom_percent: float) -> None:
        """Traduce le percentuali UI (0-100) in valori interni."""
        self.fixed_zoom = 1.0 + (fixed_zoom_percent / 50.0)
        self.dynamic_intensity = dynamic_zoom_percent / 100.0

    def compute_target_zoom(self, player_spread: float) -> float:
        """Calcola lo zoom target (fisso + bonus dinamico basato sullo spread)."""
        if self.dynamic_intensity == 0.0 or player_spread < 0:
            return self.fixed_zoom

        spread_norm = float(np.clip((self._MAX_SPREAD - player_spread) / self._MAX_SPREAD, 0.0, 1.0))
        dynamic_bonus = spread_norm * self.dynamic_intensity * self._DYNAMIC_SCALE
        return float(self.fixed_zoom + dynamic_bonus)


class Director:
    """Regia virtuale: compone CameraStrategy, filtri di smoothing e GeometryService."""

    def __init__(self) -> None:
        self.camera_strategy = CameraStrategy()
        self.zoom_smoother = ValueSmoother(smoothing_factor=0.05, initial_value=1.0)
        self.zoom_deadzone = ScalarDeadzoneFilter(threshold=0.1) # 5% di deadzone sullo zoom target
        self.deadzone_filter = DeadzoneFilter(threshold=25.0)
        self.pan_tilt_smoother = PointSmoother(smoothing_factor=0.03) # Diminuito per maggiore stabilità

    def set_config(
        self, 
        fixed_zoom_percent: float, 
        dynamic_zoom_percent: float,
        max_spread: float = 1000.0,
        dynamic_scale: float = 1.0,
        zoom_smoothing: float = 0.05,
        zoom_deadzone: float = 0.1,
        pan_tilt_deadzone: float = 25.0,
        pan_tilt_smoothing: float = 0.03
    ) -> None:
        """Imposta i parametri dalla UI e configura filtri e deadzone."""
        self.camera_strategy.set_config(fixed_zoom_percent, dynamic_zoom_percent)
        self.camera_strategy._MAX_SPREAD = max_spread
        self.camera_strategy._DYNAMIC_SCALE = dynamic_scale
        
        self.zoom_smoother.smoothing_factor = zoom_smoothing
        self.zoom_deadzone.threshold = zoom_deadzone
        self.deadzone_filter.threshold = pan_tilt_deadzone
        self.pan_tilt_smoother.smoothing_factor = pan_tilt_smoothing

    def process(
        self, frame: np.ndarray,
        action_center: Tuple[int, int],
        player_spread: float
    ) -> CameraInstruction:
        """Esegue la regia sul frame corrente."""
        target_pt = (float(action_center[0]), float(action_center[1]))
        
        # 1. Filtro Deadzone per rimuovere il tremolio microscopico
        stable_target = self.deadzone_filter.filter(target_pt)
        # 2. Addolcimento per i movimenti più lenti e decisi
        smoothed = self.pan_tilt_smoother.smooth(stable_target)

        raw_target_zoom = self.camera_strategy.compute_target_zoom(player_spread)
        stable_target_zoom = self.zoom_deadzone.filter(raw_target_zoom)
        current_zoom = self.zoom_smoother.smooth(stable_target_zoom)

        h, w = frame.shape[:2]
        center_int = (int(smoothed[0]), int(smoothed[1]))
        crop_box = GeometryService.calculate_crop_region(center_int, current_zoom, w, h)

        x1, y1, x2, y2 = crop_box
        cropped_frame = frame[y1:y2, x1:x2]

        return CameraInstruction(
            cropped_frame=cropped_frame,
            crop_box=crop_box,
            zoom_level=current_zoom,
            smoothed_center=center_int,
        )


def director_run() -> Director:
    """Crea e ritorna un Director inizializzato."""
    return Director()
