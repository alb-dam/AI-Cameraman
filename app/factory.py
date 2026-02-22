"""Factory per l'Inversion of Control: istanzia e inietta le dipendenze."""

import os
from app.logger import get_logger
from config.settings import SettingsManager

# Import dei Protocolli
from core.interfaces import IController

# Import delle implementazioni concrete
from app.state import RuntimeState
from core.roi import ROIManager, roi_manager_run
from app.pipeline import FramePipeline
from app.controller import ApplicationController
from core.performance import PerformanceMonitor

from video.input import VideoInput
from video.output import VideoOutput
from video.overlay import DebugOverlay
from core.vision import detector_run
from core.director import director_run

logger = get_logger(__name__)

class AppFactory:
    """Costruisce l'applicazione e risolve le dipendenze."""
    
    @staticmethod
    def _create_roi_manager(settings: SettingsManager) -> ROIManager:
        last_roi = settings.get("last_roi_path")
        if last_roi and os.path.exists(last_roi):
            return roi_manager_run(last_roi)
        return ROIManager()

    @staticmethod
    def create_controller(settings: SettingsManager) -> 'IController':
        """Costruisce i componenti core e ritorna il controller assemblato."""
        # 1. Crea stati e manager condivisi
        state = RuntimeState()
        roi_manager = AppFactory._create_roi_manager(settings)
        
        # 2. Crea componenti Core e Video
        # Detector
        model_name = settings.get("yolo_model")
        yolo_imgsz = settings.get("yolo_imgsz")
        debug = bool(settings.get("debug_mode"))
        detector = detector_run(model_name=model_name, yolo_imgsz=yolo_imgsz, debug=debug)
        
        # Director
        director = director_run()
        
        # Overlay
        overlay = DebugOverlay()
        
        # Video I/O
        video_input = VideoInput()
        video_output = VideoOutput()
        
        output_fps = settings.get("output_fps")
        obs_width = settings.get("output_width")
        obs_height = settings.get("output_height")
        video_output.initialize_virtual_camera(obs_width, obs_height, int(output_fps))

        perf_monitor = PerformanceMonitor(settings)

        # 3. Assembla Pipeline e Controller
        pipeline = FramePipeline(
            settings_manager=settings,
            state=state,
            roi_manager=roi_manager,
            detector=detector,
            director=director,
            overlay=overlay
        )
        
        controller = ApplicationController(
            settings_manager=settings,
            state=state,
            pipeline=pipeline,
            video_input=video_input,
            video_output=video_output,
            perf_monitor=perf_monitor
        )

        return controller
