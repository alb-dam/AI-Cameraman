"""Pipeline di elaborazione frame: Input -> Detection -> Directing -> Output."""

from typing import Tuple, Any
import numpy as np

from config.settings import SettingsManager
from core.models import FrameMetadata, DetectionResult
from core.interfaces import IRuntimeState, IROIManager, IDetector, IDirector, IOverlay


class FramePipeline:
    """Gestisce il flusso computazionale sincrono (YOLO, Tracking, Regia)."""

    def __init__(
        self, 
        settings_manager: SettingsManager, 
        state: IRuntimeState, 
        roi_manager: IROIManager,
        detector: IDetector,
        director: IDirector,
        overlay: IOverlay
    ) -> None:
        self.settings = settings_manager
        self.state = state
        self.roi_manager = roi_manager
        
        self.detector = detector
        self.director = director
        self.overlay = overlay
        
        # Dirty flags per evitare di ri-applicare config ad ogni frame
        self._last_detector_config_version: int = -1
        self._last_director_config_version: int = -1

    def run_inference(self, frame: np.ndarray) -> DetectionResult:
        """Esegue YOLO (o skip pass) e ritorna det_out."""
        self._maybe_apply_detector_config()
        
        interval = self.settings.get("yolo_inference_interval")
        run_full_inference = (self.state.yolo_frame_counter % interval) == 0
        self.state.yolo_frame_counter += 1

        ai_frame = self.roi_manager.apply_roi(frame)
        
        det_out = self.detector.process(ai_frame, predict_only=not run_full_inference)
        return det_out

    def run_tracking_and_directing(self, frame: np.ndarray, det_out: Any, metadata: FrameMetadata) -> Tuple[np.ndarray, np.ndarray]:
        """Esegue il tracking (Kalman) e la regia per produrre l'output."""
        self._maybe_apply_director_config()
        
        dir_out = self.director.process(frame, det_out.action_center, det_out.player_spread)
        obs_frame = dir_out.cropped_frame

        # Genera debug frame: overlay su video (se preview attiva) o su sfondo nero (se solo debug)
        if self.settings.get("debug_mode"):
            if self.state.is_preview_enabled:
                debug_frame = frame.copy()
            else:
                debug_frame = np.zeros_like(frame)
            self.overlay.draw(debug_frame, det_out, dir_out, self.roi_manager, metadata)
        elif self.state.is_preview_enabled:
            debug_frame = frame
        else:
            debug_frame = None

        return obs_frame, debug_frame

    def _maybe_apply_detector_config(self) -> None:
        """Applica config YOLO/Kalman solo se i settings sono cambiati (dirty flag)."""
        current_version = self.settings.get_version()
        if current_version == self._last_detector_config_version:
            return
        self._last_detector_config_version = current_version
        self._apply_detector_config()

    def _maybe_apply_director_config(self) -> None:
        """Applica config regia solo se i settings sono cambiati (dirty flag)."""
        current_version = self.settings.get_version()
        if current_version == self._last_director_config_version:
            return
        self._last_director_config_version = current_version
        self._apply_director_config()

    def _apply_detector_config(self) -> None:
        """Applica gli aggiornamenti dinamici in tempo reale per YOLO/Kalman dai settings."""
        # Filtro sempre il più reattivo possibile
        q_std = 100.0
        r_std = 0.01

        self.detector.set_config(q_std, r_std, int(self.settings.get("yolo_imgsz")))

    def _apply_director_config(self) -> None:
        """Applica gli aggiornamenti dinamici in tempo reale per la regia dai settings."""
        deadzone_pct = float(self.settings.get("director_deadzone_preset_percent")) / 100.0
        inertia_pct = float(self.settings.get("director_inertia_preset_percent")) / 100.0

        # Interpola i valori Tolleranza (Deadzone)
        z_dz_min = float(self.settings.get("director_zoom_deadzone_min"))
        z_dz_max = float(self.settings.get("director_zoom_deadzone_max"))
        zoom_deadzone = z_dz_min + (z_dz_max - z_dz_min) * deadzone_pct

        pt_dz_min = float(self.settings.get("director_pan_tilt_deadzone_min"))
        pt_dz_max = float(self.settings.get("director_pan_tilt_deadzone_max"))
        pan_tilt_deadzone = pt_dz_min + (pt_dz_max - pt_dz_min) * deadzone_pct

        spread_min = float(self.settings.get("director_max_spread_min"))
        spread_max = float(self.settings.get("director_max_spread_max"))
        max_spread = spread_min + (spread_max - spread_min) * deadzone_pct
        
        # Interpola i valori Velocità (Inerzia/Smoothing)
        z_sm_min = float(self.settings.get("director_zoom_smoothing_min"))
        z_sm_max = float(self.settings.get("director_zoom_smoothing_max"))
        zoom_smoothing = z_sm_min + (z_sm_max - z_sm_min) * inertia_pct

        pt_sm_min = float(self.settings.get("director_pan_tilt_smoothing_min"))
        pt_sm_max = float(self.settings.get("director_pan_tilt_smoothing_max"))
        pan_tilt_smoothing = pt_sm_min + (pt_sm_max - pt_sm_min) * inertia_pct

        scale_min = float(self.settings.get("director_dynamic_scale_min"))
        scale_max = float(self.settings.get("director_dynamic_scale_max"))
        dynamic_scale = scale_min + (scale_max - scale_min) * inertia_pct

        self.director.set_config(
            float(self.settings.get("fixed_zoom_percent")),
            float(self.settings.get("dynamic_zoom_percent")),
            max_spread=max_spread,
            dynamic_scale=dynamic_scale,
            zoom_smoothing=zoom_smoothing,
            zoom_deadzone=zoom_deadzone,
            pan_tilt_deadzone=pan_tilt_deadzone,
            pan_tilt_smoothing=pan_tilt_smoothing
        )
