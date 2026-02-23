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
        percent = float(self.settings.get("kalman_preset_percent")) / 100.0
        
        q_smooth = float(self.settings.get("kalman_q_smooth"))
        r_smooth = float(self.settings.get("kalman_r_smooth"))
        q_reactive = float(self.settings.get("kalman_q_reactive"))
        r_reactive = float(self.settings.get("kalman_r_reactive"))

        q_std = q_smooth + (q_reactive - q_smooth) * percent
        r_std = r_smooth + (r_reactive - r_smooth) * percent

        self.detector.set_config(q_std, r_std, int(self.settings.get("yolo_imgsz")))

    def _apply_director_config(self) -> None:
        """Applica gli aggiornamenti dinamici in tempo reale per la regia dai settings."""
        self.director.set_config(
            float(self.settings.get("fixed_zoom_percent")),
            float(self.settings.get("dynamic_zoom_percent")),
            max_spread=float(self.settings.get("director_max_spread")),
            dynamic_scale=float(self.settings.get("director_dynamic_scale")),
            zoom_smoothing=float(self.settings.get("director_zoom_smoothing")),
            zoom_deadzone=float(self.settings.get("director_zoom_deadzone")),
            pan_tilt_deadzone=float(self.settings.get("director_pan_tilt_deadzone")),
            pan_tilt_smoothing=float(self.settings.get("director_pan_tilt_smoothing"))
        )
