"""Regia virtuale: pan, tilt, zoom dinamico/fisso e ritaglio.

Architettura interna:
    ZoomManager        – calcolo zoom fisso + dinamico con smoothing
    PanTiltController  – smoothing temporale del centro inquadrato
    CropCalculator     – calcolo e clamping della regione di ritaglio
    Director           – façade pubblica che compone le tre classi

Nessuna dipendenza da YOLO o dal modulo detector.
Input richiesti: frame, action_center, player_spread.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

from app.logger import get_logger

logger = get_logger(__name__)


# ── 1. Gestione Zoom ────────────────────────────────────────────────────


class ZoomManager:
    """Calcola il livello di zoom combinando componente fissa e dinamica."""

    _MAX_SPREAD = 1500.0   # spread massimo → nessun bonus dinamico
    _DYNAMIC_SCALE = 0.5   # moltiplicatore massimo del bonus dinamico

    def __init__(self) -> None:
        self.fixed_zoom: float = 1.0         # 1.0x = nessun zoom
        self.dynamic_intensity: float = 0.0  # 0.0 = disattivato
        self.current_zoom: float = 1.0
        self._zoom_smoothing: float = 0.10   # fattore smoothing zoom

    def set_config(self, fixed_zoom_percent: float, dynamic_zoom_percent: float) -> None:
        """Traduce le percentuali UI (0-100) in valori interni."""
        self.fixed_zoom = 1.0 + (fixed_zoom_percent / 50.0)
        self.dynamic_intensity = dynamic_zoom_percent / 100.0

    def compute_target_zoom(self, player_spread: float) -> float:
        """Calcola lo zoom target (fisso + bonus dinamico)."""
        if self.dynamic_intensity == 0.0 or player_spread < 0:
            return self.fixed_zoom

        spread_norm = np.clip((self._MAX_SPREAD - player_spread) / self._MAX_SPREAD, 0.0, 1.0)
        dynamic_bonus = spread_norm * self.dynamic_intensity * self._DYNAMIC_SCALE
        return self.fixed_zoom + dynamic_bonus

    def smooth_transition(self, target_zoom: float) -> float:
        """Ammorbidisce la transizione verso il target zoom."""
        self.current_zoom += (target_zoom - self.current_zoom) * self._zoom_smoothing
        return self.current_zoom


# ── 2. Pan / Tilt ────────────────────────────────────────────────────────


class PanTiltController:
    """Gestisce lo smoothing temporale del centro inquadrato (pan + tilt)."""

    def __init__(self, smoothing_factor: float = 0.05) -> None:
        self.smoothed_center: Optional[Tuple[float, float]] = None
        self.smoothing_factor: float = smoothing_factor

    def compute_offset(self, action_center: Tuple[int, int]) -> Tuple[float, float]:
        """Aggiorna e ritorna il centro smoothato."""
        target_x, target_y = float(action_center[0]), float(action_center[1])

        if self.smoothed_center is None:
            result = (target_x, target_y)
        else:
            sx = self.smoothed_center[0] + (target_x - self.smoothed_center[0]) * self.smoothing_factor
            sy = self.smoothed_center[1] + (target_y - self.smoothed_center[1]) * self.smoothing_factor
            result = (sx, sy)

        self.smoothed_center = result
        return result


# ── 3. Calcolo Crop ──────────────────────────────────────────────────────


class CropCalculator:
    """Calcola la regione di ritaglio e clampa ai bordi del frame (stateless)."""

    @staticmethod
    def calculate_crop_region(
        center: Tuple[int, int], zoom: float, frame_w: int, frame_h: int
    ) -> Tuple[int, int, int, int]:
        """Genera le coordinate di ritaglio (x1, y1, x2, y2)."""
        crop_w = int(frame_w / zoom)
        crop_h = int(frame_h / zoom)
        cx, cy = center

        x1 = cx - crop_w // 2
        y1 = cy - crop_h // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        return CropCalculator.clamp_to_frame(x1, y1, x2, y2, crop_w, crop_h, frame_w, frame_h)

    @staticmethod
    def clamp_to_frame(
        x1: int, y1: int, x2: int, y2: int,
        crop_w: int, crop_h: int, frame_w: int, frame_h: int
    ) -> Tuple[int, int, int, int]:
        """Clampa le coordinate ai bordi del frame, preservando la dimensione del crop."""
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x1 + crop_w)
        y2 = min(frame_h, y1 + crop_h)

        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)

        return (x1, y1, x2, y2)


# ── Façade pubblica ──────────────────────────────────────────────────────


class Director:
    """Regia virtuale: compone ZoomManager, PanTiltController e CropCalculator."""

    def __init__(self) -> None:
        self.zoom_manager = ZoomManager()
        self.pan_tilt = PanTiltController()
        self.crop_calc = CropCalculator()

    def set_config(self, fixed_zoom_percent: float, dynamic_zoom_percent: float) -> None:
        """Imposta le percentuali di zoom dalla UI. Delega a ZoomManager."""
        self.zoom_manager.set_config(fixed_zoom_percent, dynamic_zoom_percent)

    def process(
        self, frame: np.ndarray,
        action_center: Tuple[int, int],
        player_spread: float
    ) -> Dict[str, Any]:
        """Esegue la regia sul frame corrente."""
        smoothed = self.pan_tilt.compute_offset(action_center)

        target_zoom = self.zoom_manager.compute_target_zoom(player_spread)
        current_zoom = self.zoom_manager.smooth_transition(target_zoom)

        h, w = frame.shape[:2]
        center_int = (int(smoothed[0]), int(smoothed[1]))
        crop_box = self.crop_calc.calculate_crop_region(center_int, current_zoom, w, h)

        x1, y1, x2, y2 = crop_box
        cropped_frame = frame[y1:y2, x1:x2]

        return {
            "cropped_frame": cropped_frame,
            "crop_box": crop_box,
            "zoom_level": current_zoom,
            "smoothed_center": center_int,
        }


def director_run() -> Director:
    """Crea e ritorna un Director inizializzato."""
    return Director()
