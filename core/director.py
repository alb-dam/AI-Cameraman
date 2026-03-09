"""Regia virtuale: pan, tilt, zoom dinamico/fisso e ritaglio.

Architettura interna:
    ValueSmoother      – smoothing dello zoom (scalare)
    PointSmoother      – smoothing temporale del centro inquadrato (2D)
    CameraStrategy     – calcolo zoom fisso + dinamico
    Director           – façade pubblica che compone le classi precedenti

Nessuna dipendenza da YOLO o dal modulo detector.
Input richiesti: frame, action_center, player_spread.

NOTA: ActionCenterCalculator è stato spostato in core/tracking.py
"""

import numpy as np
from typing import Tuple, Optional
import logging

from core.models import CameraInstruction
from core.geometry import GeometryService

logger = logging.getLogger(__name__)



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
    """Filtro che si comporta come un 'guinzaglio' (leash): ignora i micromovimenti all'interno della deadzone, ma segue in modo fluido quando il target esce dalla soglia."""

    def __init__(self, threshold_pct: float = 0.05) -> None:
        self.threshold_pct = threshold_pct
        self.stable_point: Optional[Tuple[float, float]] = None

    def filter(self, target_point: Tuple[float, float], reference_length: float) -> Tuple[float, float]:
        """Se il punto esce dalla soglia, il centro stabile viene 'trascinato' lungo il perimetro della deadzone."""
        stable = self.stable_point
        if stable is None:
            self.stable_point = target_point
            return target_point
            
        spx, spy = stable
        dx = target_point[0] - spx
        dy = target_point[1] - spy
        dist = float(np.hypot(dx, dy))
        
        threshold_px = self.threshold_pct * reference_length
        if dist <= threshold_px:
            return stable
            
        # Comportamento "Leash": trasciniamo il punto stabile così che la distanza dal target sia esattamente threshold_px
        ratio = threshold_px / dist
        new_spx = target_point[0] - dx * ratio
        new_spy = target_point[1] - dy * ratio
        
        new_point = (new_spx, new_spy)
        self.stable_point = new_point
        return new_point


class ScalarDeadzoneFilter:
    """Filtro a guinzaglio scalare: annulla lo zoom hunting all'interno della soglia, ma segue dolcemente all'esterno."""

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold
        self.stable_value: Optional[float] = None

    def filter(self, target_value: float) -> float:
        """Trascina il valore stabile mantenendo una distanza massima pari alla soglia."""
        stable = self.stable_value
        if stable is None:
            self.stable_value = target_value
            return target_value
            
        diff = abs(target_value - stable)
        
        if diff <= self.threshold:
            return stable
            
        # Al di fuori della deadzone, agganciamo esattamente il target
        # (invece di usare un leash) per evitare che lo zoom rimanga permanentemente sfalsato.
        self.stable_value = target_value
        return target_value


class CameraStrategy:
    """Strategia di calcolo dello zoom (fisso vs dinamico) dato uno spread di giocatori."""

    def __init__(self) -> None:
        self.fixed_zoom: float = 1.0         # 1.0x = nessun zoom
        self.dynamic_intensity: float = 0.0  # 0.0 = disattivato
        self._MAX_SPREAD: float = 1000.0     # spread massimo (più alto = attesa prima di zoomare)
        self._DYNAMIC_SCALE: float = 1.0     # moltiplicatore massimo del bonus dinamico

    def set_config(
        self,
        fixed_zoom_percent: float,
        dynamic_zoom_percent: float,
        max_spread: float = 1000.0,
        dynamic_scale: float = 1.0
    ) -> None:
        """Traduce le percentuali UI (0-100) in valori interni."""
        self.fixed_zoom = 1.0 + (fixed_zoom_percent / 50.0)
        self.dynamic_intensity = dynamic_zoom_percent / 100.0
        self._MAX_SPREAD = max_spread
        self._DYNAMIC_SCALE = dynamic_scale

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
        
        self.base_pan_tilt_deadzone: float = 0.05
        self.base_pan_tilt_smoothing: float = 0.03
        
        self.deadzone_filter = DeadzoneFilter(threshold_pct=self.base_pan_tilt_deadzone)
        self.pan_tilt_smoother = PointSmoother(smoothing_factor=self.base_pan_tilt_smoothing) # Diminuito per maggiore stabilità

    def set_config(
        self, 
        fixed_zoom_percent: float, 
        dynamic_zoom_percent: float,
        max_spread: float = 1000.0,
        dynamic_scale: float = 1.0,
        zoom_smoothing: float = 0.05,
        zoom_deadzone: float = 0.1,
        pan_tilt_deadzone: float = 0.05,
        pan_tilt_smoothing: float = 0.03
    ) -> None:
        """Imposta i parametri dalla UI e configura filtri e deadzone."""
        self.camera_strategy.set_config(
            fixed_zoom_percent, dynamic_zoom_percent,
            max_spread=max_spread, dynamic_scale=dynamic_scale
        )
        
        self.zoom_smoother.smoothing_factor = zoom_smoothing
        self.zoom_deadzone.threshold = zoom_deadzone
        
        self.base_pan_tilt_deadzone = pan_tilt_deadzone
        self.base_pan_tilt_smoothing = pan_tilt_smoothing
        
        # Le applichiamo come default iniziale
        self.deadzone_filter.threshold_pct = self.base_pan_tilt_deadzone
        self.pan_tilt_smoother.smoothing_factor = self.base_pan_tilt_smoothing

    def process(
        self, frame: np.ndarray,
        action_center: Tuple[int, int],
        player_spread: float
    ) -> CameraInstruction:
        """Esegue la regia sul frame corrente."""
        target_pt = (float(action_center[0]), float(action_center[1]))
        h, w = frame.shape[:2]
        reference_length = float(np.hypot(w, h))

        # 1. Calcolo dello zoom attuale
        raw_target_zoom = self.camera_strategy.compute_target_zoom(player_spread)
        stable_target_zoom = self.zoom_deadzone.filter(raw_target_zoom)
        current_zoom = self.zoom_smoother.smooth(stable_target_zoom)
        
        # 2. Modulazione dinamica della sensibilità di Pan/Tilt in base allo zoom
        # Più zoom = movimenti più lenti e deadzone più ridotta (per reagire in fretta ma dolcemente)
        dynamic_pan_smoothing = self.base_pan_tilt_smoothing / current_zoom
        dynamic_pan_deadzone = self.base_pan_tilt_deadzone / current_zoom
        
        self.pan_tilt_smoother.smoothing_factor = dynamic_pan_smoothing
        self.deadzone_filter.threshold_pct = dynamic_pan_deadzone
        
        # 3. Filtro Deadzone per rimuovere il tremolio microscopico
        stable_target = self.deadzone_filter.filter(target_pt, reference_length)
        # 4. Addolcimento per i movimenti più lenti e decisi
        smoothed = self.pan_tilt_smoother.smooth(stable_target)

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
