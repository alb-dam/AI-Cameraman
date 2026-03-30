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



class VirtualPTZModel:
    """Modello matematico unificato per la gestione fluida e cinematica del PTZ Virtuale.
    Integra Deadzone (Leash) e Smoothing per Pan, Tilt e Zoom in un'unica cascata di equazioni.
    """

    def __init__(self) -> None:
        # Stato Corrente (Pan, Tilt, Zoom)
        self.current_x: Optional[float] = None
        self.current_y: Optional[float] = None
        self.current_z: float = 1.0  # Zoom
        
        # Leash State (Punto stabile ancorato dalla deadzone)
        self.stable_x: Optional[float] = None
        self.stable_y: Optional[float] = None
        self.stable_z: float = 1.0

        # Parametri Base Regia
        self.fixed_zoom: float = 1.0
        self.dynamic_intensity: float = 0.0
        
        # Variabili Interpolate dal Controller (valori di default sicuri)
        self.max_spread: float = 0.6
        self.dynamic_scale: float = 1.5
        self.zoom_smoothing: float = 0.1
        self.zoom_deadzone: float = 0.1
        self.pan_tilt_deadzone: float = 0.1
        self.pan_tilt_smoothing: float = 0.1

    def set_config(
        self,
        fixed_zoom_percent: float,
        dynamic_zoom_percent: float,
        max_spread: float = 0.6,
        dynamic_scale: float = 1.5,
        zoom_smoothing: float = 0.1,
        zoom_deadzone: float = 0.1,
        pan_tilt_deadzone: float = 0.1,
        pan_tilt_smoothing: float = 0.1
    ) -> None:
        """Applica i parametri interpolati dal controller ai coefficienti del PTZ."""
        self.fixed_zoom = 1.0 + (fixed_zoom_percent / 50.0)
        self.dynamic_intensity = dynamic_zoom_percent / 100.0
        
        self.max_spread = max_spread
        self.dynamic_scale = dynamic_scale
        self.zoom_smoothing = zoom_smoothing
        self.zoom_deadzone = zoom_deadzone
        self.pan_tilt_deadzone = pan_tilt_deadzone
        self.pan_tilt_smoothing = pan_tilt_smoothing

    def step(self, target_x: float, target_y: float, player_spread: float, reference_length: float) -> Tuple[float, float, float]:
        """Esegue un tick del modello matematico e ritorna (pan_x, tilt_y, zoom_z) correnti."""
        
        # --- Equazione 1: Calcolo Target Zoom (Fisso + Dinamico) ---
        if self.dynamic_intensity > 0.0 and player_spread >= 0:
            # Normalizziamo lo spread (giocatori distanti = meno bonus, vicini = più bonus)
            spread_norm = float(np.clip((self.max_spread - player_spread) / self.max_spread, 0.0, 1.0))
            dynamic_bonus = spread_norm * self.dynamic_intensity * self.dynamic_scale
            target_z = self.fixed_zoom + dynamic_bonus
        else:
            target_z = self.fixed_zoom

        # Inizializzazione lazy al primo frame utile
        if self.current_x is None or self.stable_x is None or self.stable_y is None or self.current_y is None:
            self.current_x = target_x
            self.stable_x = target_x
            self.current_y = target_y
            self.stable_y = target_y
            self.current_z = target_z
            self.stable_z = target_z

        cx: float = float(self.current_x)
        cy: float = float(self.current_y)
        sx: float = float(self.stable_x)
        sy: float = float(self.stable_y)

        # --- Equazione 2: Zoom Deadzone ---
        diff_z = abs(target_z - self.stable_z)
        if diff_z > self.zoom_deadzone:
            # Agganciamo esattamente il target z fuori deadzone
            self.stable_z = target_z

        # --- Equazione 3: Zoom Smoothing ---
        self.current_z += (self.stable_z - self.current_z) * self.zoom_smoothing

        # --- Equazione 4: Derivazione Parametri Spaziali scalati dallo Zoom ---
        # Più la telecamera fa zoom, più l'inquadratura è ristretta. Dunque:
        # A) Diminuiamo la deadzone consentita così la telecamera reagisce prima
        # B) Diminuiamo lo smoothing perché i movimenti veloci sembreranno amplificati.
        dynamic_pt_deadzone = self.pan_tilt_deadzone / self.current_z
        dynamic_pt_smoothing = self.pan_tilt_smoothing / self.current_z

        # --- Equazione 5: Pan/Tilt Deadzone (Filtro Elastico / Leash) ---
        dx = target_x - sx
        dy = target_y - sy
        dist = float(np.hypot(dx, dy))
        
        # Convertiamo la deadzone percentuale in pixel rispetto alla diagonale dell'inquadratura
        threshold_px = dynamic_pt_deadzone * reference_length

        if dist > threshold_px:
            # Guinzaglio: spostiamo il punto stabile in direzione del target, mantenendolo a distanza 'threshold_px'
            ratio = threshold_px / dist
            sx = target_x - dx * ratio
            sy = target_y - dy * ratio
            self.stable_x = sx
            self.stable_y = sy

        # --- Equazione 6: Pan/Tilt Smoothing ---
        cx += (sx - cx) * dynamic_pt_smoothing
        cy += (sy - cy) * dynamic_pt_smoothing
        
        self.current_x = cx
        self.current_y = cy

        return (cx, cy, self.current_z)


class Director:
    """Regia virtuale: wrapper per eseguire VirtualPTZModel e convertire in crop box."""

    def __init__(self) -> None:
        self.ptz_model = VirtualPTZModel()

    def set_config(
        self, 
        fixed_zoom_percent: float, 
        dynamic_zoom_percent: float,
        max_spread: float = 0.6,
        dynamic_scale: float = 1.5,
        zoom_smoothing: float = 0.1,
        zoom_deadzone: float = 0.1,
        pan_tilt_deadzone: float = 0.1,
        pan_tilt_smoothing: float = 0.1
    ) -> None:
        """Imposta i parametri dalla UI nel modello matematico unificato."""
        self.ptz_model.set_config(
            fixed_zoom_percent, dynamic_zoom_percent,
            max_spread=max_spread, dynamic_scale=dynamic_scale,
            zoom_smoothing=zoom_smoothing, zoom_deadzone=zoom_deadzone,
            pan_tilt_deadzone=pan_tilt_deadzone, pan_tilt_smoothing=pan_tilt_smoothing
        )

    def process(
        self, frame: np.ndarray,
        action_center: Tuple[int, int],
        player_spread: float
    ) -> CameraInstruction:
        """Esegue la regia sul frame corrente applicando l'algoritmo matematico e ritornando il crop."""
        h, w = frame.shape[:2]
        reference_length = float(np.hypot(w, h))
        
        # Esecuzione del Modello Matematico Unico PTZ
        cx, cy, current_zoom = self.ptz_model.step(
            float(action_center[0]), 
            float(action_center[1]), 
            player_spread, 
            reference_length
        )
        
        center_int = (int(cx), int(cy))
        # Generazione Crop Virtuale tramite modulo di Geometria
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
