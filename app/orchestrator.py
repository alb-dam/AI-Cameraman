"""Orchestratore puro: coordinazione Input → Detector → Director → Output.

Nessuna logica YOLO, nessuna logica zoom, nessun disegno CV2.
Solo inizializzazione moduli, gestione thread e dispatch frame.
"""

import os
import threading
import time
from typing import Tuple, Optional, Callable, List
import numpy as np

from app.logger import get_logger
from config.settings import SettingsManager
from video.input import VideoInput
from video.output import VideoOutput
from core.detector import detector_run, DetectionResult
from core.director import director_run
from video.overlay import DebugOverlay
from app.roi_manager import roi_manager_run, ROIManager

logger = get_logger(__name__)


class Orchestrator:
    """Orchestratore puro: coordina i moduli senza logica di elaborazione."""

    def __init__(self, settings_manager: SettingsManager) -> None:
        """Inizializza tutti i sotto-moduli. Nessun I/O qui."""
        self.settings_manager: SettingsManager = settings_manager
        
        # Moduli funzionali
        self.video_input: VideoInput = VideoInput()
        self.video_output: VideoOutput = VideoOutput()
        self.detector = detector_run(model_name=self.settings_manager.get("yolo_model"))
        self.director = director_run()
        self.overlay: DebugOverlay = DebugOverlay()
        self.roi_manager: ROIManager = self._load_initial_roi()

        # Stato threading
        self.is_running, self.is_outputting_to_obs, self.source_exhausted = False, False, False
        self.reader_thread = self.processor_thread = self.output_thread = None
        self.frame_lock = threading.Lock()
        self.latest_raw_frame = self.latest_obs_frame = self.latest_debug_frame = None

        self.on_frame_ready: Optional[Callable[[np.ndarray], None]] = None
        self.on_log_message: Optional[Callable[[str], None]] = None

        self.frames_processed, self.start_time = 0, 0.0
        self.current_fps = 0.0
        
        self.input_fps = 30.0
        self.output_fps = self.settings_manager.get("output_fps")
        self.obs_width = self.settings_manager.get("output_width")
        self.obs_height = self.settings_manager.get("output_height")

        # Init virtual camera una sola volta
        self.video_output.initialize_virtual_camera(self.obs_width, self.obs_height, int(self.output_fps))

    # ── Entry point pubblico ────────────────────────────────────────────

    def orchestrator_run(self) -> None:
        """Entry point pubblico: avvia l'elaborazione video."""
        self.start_processing()

    # ── Controllo elaborazione ──────────────────────────────────────────

    def start_processing(self) -> None:
        """Avvia l'elaborazione del video in thread separati."""
        if self.is_running:
            self.stop_processing()

        source_type = self.settings_manager.get("source_type")
        source_path = self.settings_manager.get("source_path")

        try:
            self.video_input.initialize_source(source_type, source_path)
            label = f"file {source_path}" if source_type == "file" else "webcam"
            self._log(f"Sorgente inizializzata: {label}")
        except Exception as e:
            self._log(f"Errore inizializzazione sorgente: {e}")
            return

        self.input_fps = self.video_input.get_fps()
        if self.input_fps <= 0:
            self.input_fps = 30.0

        self.output_fps = self.settings_manager.get("output_fps")
        self.obs_width = self.settings_manager.get("output_width")
        self.obs_height = self.settings_manager.get("output_height")

        self.video_output.initialize_virtual_camera(self.obs_width, self.obs_height, int(self.output_fps))

        self.is_running = True
        self.source_exhausted = False
        
        with self.frame_lock:
            self.latest_raw_frame = None
            self.latest_obs_frame = None
            self.latest_debug_frame = None

        self.frames_processed = 0
        self.start_time = time.time()

        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.processor_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.output_thread = threading.Thread(target=self._output_loop, daemon=True)

        self.reader_thread.start()
        self.processor_thread.start()
        self.output_thread.start()
        
        self._log(f"Elaborazione avviata (IN: {self.input_fps:.1f} FPS, OUT: {self.output_fps} FPS @ {self.obs_width}x{self.obs_height}).")

    def stop_processing(self) -> None:
        """Ferma l'elaborazione del video."""
        if not self.is_running:
            return

        self._log("Chiusura elaborazione in corso...")
        self.is_running = False

        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2.0)
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=2.0)

        self.video_input.release()
        self._log("Elaborazione chiusa in sicurezza.")

    def close_all(self) -> None:
        """Chiusura totale in fase di exit application."""
        self.stop_processing()
        self.video_output.close()

    # ── Controllo OBS ───────────────────────────────────────────────────

    def start_obs_output(self) -> None:
        """Avvia l'invio del flusso video a OBS."""
        if not self.is_running:
            self._log("Errore: impossibile avviare l'elaborazione OBS senza sorgente.")
            return
        self.is_outputting_to_obs = True
        self._log("Trasmissione verso OBS avviata.")

    def stop_obs_output(self) -> None:
        """Ferma l'invio del flusso video a OBS."""
        self.is_outputting_to_obs = False
        self._log("Trasmissione verso OBS fermata.")

    # ── Bridge ROI (Delega Esposta Direttamente) ────────────────

    # L'accesso manuale avverrà tramite self.roi_manager dalla GUI.

    # ── 1. Reader Loop ─────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        is_file = self.settings_manager.get("source_type") == "file"
        frame_duration = 1.0 / self.input_fps
        empty_frames_count = 0

        while self.is_running:
            loop_start = time.time()

            ret, frame = self.video_input.read_frame()
            if not ret:
                empty_frames_count += 1
                if empty_frames_count > 30:
                    self._log("Flusso video interrotto (>30 frame vuoti).")
                    self.source_exhausted = True
                    break
                time.sleep(0.03)
                continue

            empty_frames_count = 0
            
            with self.frame_lock:
                self.latest_raw_frame = frame.copy()

            if is_file:
                sleep_time = frame_duration - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.001)

    # ── 2. Processing Loop ──────────────────────────────────────────────

    def _processing_loop(self) -> None:
        last_processed_frame_id = id(None)
        
        while self.is_running:
            if self.source_exhausted:
                break
                
            with self.frame_lock:
                raw_frame = self.latest_raw_frame
                
            if raw_frame is None or id(raw_frame) == last_processed_frame_id:
                time.sleep(0.005)
                continue
                
            last_processed_frame_id = id(raw_frame)
            
            try:
                # Scaliamo in proporzione (letterbox) il raw_frame alla risoluzione di target 
                # PRIMA di passarlo a YOLO, Tracking e Regia per risparmiare moltissime risorse CPU/RAM.
                working_frame = VideoOutput._resize_and_pad(
                    raw_frame, (self.obs_width, self.obs_height)
                )

                obs_frame, debug_frame = self._process_frame(working_frame)
                with self.frame_lock:
                    self.latest_obs_frame = obs_frame
                    self.latest_debug_frame = debug_frame
                    
                self._update_fps()
            except Exception as e:
                self._log(f"Errore durante processamento frame: {e}")

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._apply_config_updates()

        ai_frame = self.roi_manager.apply_roi(frame.copy())
        det_out = self.detector.process(ai_frame)
        dir_out = self.director.process(frame, det_out.action_center, det_out.player_spread)

        obs_frame = dir_out.get("cropped_frame", frame)

        debug_frame = frame.copy()
        if self.settings_manager.get("debug_mode"):
            self.overlay.draw(debug_frame, det_out, dir_out, self.roi_manager, self.current_fps)

        return obs_frame, debug_frame

    # ── 3. Output Loop ──────────────────────────────────────────────────

    def _output_loop(self) -> None:
        frame_duration = 1.0 / self.output_fps
        
        while self.is_running:
            if self.source_exhausted:
                break
                
            loop_start = time.time()

            with self.frame_lock:
                obs_frame = self.latest_obs_frame
                debug_frame = self.latest_debug_frame
            
            if obs_frame is not None and debug_frame is not None:
                self._dispatch_outputs(debug_frame, obs_frame)

            sleep_time = frame_duration - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ── Metodi interni di supporto ──────────────────────────────────────

    def _apply_config_updates(self) -> None:
        percent = self.settings_manager.get("kalman_preset_percent") / 100.0
        
        q_smooth = self.settings_manager.get("kalman_q_smooth")
        r_smooth = self.settings_manager.get("kalman_r_smooth")
        q_reactive = self.settings_manager.get("kalman_q_reactive")
        r_reactive = self.settings_manager.get("kalman_r_reactive")

        q_std = q_smooth + (q_reactive - q_smooth) * percent
        r_std = r_smooth + (r_reactive - r_smooth) * percent

        self.detector.set_config(q_std, r_std)
        self.director.set_config(
            self.settings_manager.get("fixed_zoom_percent"),
            self.settings_manager.get("dynamic_zoom_percent")
        )

    def _dispatch_outputs(self, debug_frame: np.ndarray, obs_frame: np.ndarray) -> None:
        if self.on_frame_ready:
            self.on_frame_ready(debug_frame)

        if self.is_outputting_to_obs:
            self.video_output.send_frame(obs_frame)

    def _update_fps(self) -> None:
        self.frames_processed += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.current_fps = self.frames_processed / elapsed
            self.frames_processed = 0
            self.start_time = time.time()

    def _load_initial_roi(self) -> ROIManager:
        last_roi = self.settings_manager.get("last_roi_path")
        if last_roi and os.path.exists(last_roi):
            return roi_manager_run(last_roi)
        return ROIManager()

    def _log(self, message: str) -> None:
        logger.info(f"[Orchestrator] {message}")
        if self.on_log_message:
            self.on_log_message(message)
