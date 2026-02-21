"""Orchestratore puro: coordinazione Input → Detector → Director → Output.

Nessuna logica YOLO, nessuna logica zoom, nessun disegno CV2.
Solo inizializzazione moduli, gestione thread e dispatch frame.
"""

import os
import threading
import time
import logging

logger = logging.getLogger(__name__)

import numpy as np
from typing import Tuple, Optional, Callable, List

from config import ConfigManager
from input import VideoInput
from output import VideoOutput
from detector import detector_run, DetectionResult
from director import director_run
from overlay import DebugOverlay
from roi import ROIManager, roi_run


class Backend:
    """Orchestratore puro: coordina i moduli senza logica di elaborazione."""

    # ── Inizializzazione ────────────────────────────────────────────────

    def __init__(self, config_manager: ConfigManager) -> None:
        """Inizializza tutti i sotto-moduli. Nessun I/O qui."""
        self.config: ConfigManager = config_manager

        # Moduli funzionali
        self.video_input: VideoInput = VideoInput()
        self.video_output: VideoOutput = VideoOutput()
        self.detector = detector_run(model_name=self.config.get("yolo_model"))
        self.director = director_run()
        self.overlay: DebugOverlay = DebugOverlay()
        self.roi_manager: ROIManager = self._load_initial_roi()

        # Stato threading
        self.is_running: bool = False
        self.is_outputting_to_obs: bool = False
        
        self.reader_thread: Optional[threading.Thread] = None
        self.processor_thread: Optional[threading.Thread] = None
        self.output_thread: Optional[threading.Thread] = None

        self.frame_lock = threading.Lock()
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_obs_frame: Optional[np.ndarray] = None
        self.latest_debug_frame: Optional[np.ndarray] = None
        self.source_exhausted: bool = False

        # Callback verso la UI (impostati dal Frontend)
        self.on_frame_ready: Optional[Callable[[np.ndarray], None]] = None
        self.on_log_message: Optional[Callable[[str], None]] = None

        # Metriche FPS
        self.frames_processed: int = 0
        self.start_time: float = 0.0
        self.current_fps: float = 0.0
        self.target_fps: float = 30.0

        # Dimensioni output OBS
        self.obs_width: int = 1920
        self.obs_height: int = 1080

        # Init virtual camera una sola volta
        self.video_output.initialize_virtual_camera(self.obs_width, self.obs_height, 30)

    # ── Entry point pubblico ────────────────────────────────────────────

    def backend_run(self) -> None:
        """Entry point pubblico: avvia l'elaborazione video."""
        self.start_processing()

    # ── Controllo elaborazione ──────────────────────────────────────────

    def start_processing(self) -> None:
        """Avvia l'elaborazione del video in thread separati."""
        if self.is_running:
            self.stop_processing()

        source_type = self.config.get("source_type")
        source_path = self.config.get("source_path")

        try:
            self.video_input.initialize_source(source_type, source_path)
            label = f"file {source_path}" if source_type == "file" else "webcam"
            self._log(f"Sorgente inizializzata: {label}")
        except Exception as e:
            self._log(f"Errore inizializzazione sorgente: {e}")
            return

        self.target_fps = self.video_input.get_fps()
        if self.target_fps <= 0:
            self.target_fps = 30.0

        # Reinizializza virtual camera con i nuovi FPS nativi
        self.video_output.initialize_virtual_camera(self.obs_width, self.obs_height, int(self.target_fps))

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
        
        self._log(f"Elaborazione avviata ({self.target_fps} FPS nativi).")


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

    # ── Bridge ROI (encapsulation per il Frontend) ──────────────────────

    def load_roi(self, filepath: str) -> None:
        """Carica una ROI da file JSON."""
        self.roi_manager.load_roi(filepath)

    def save_roi(self, filepath: str) -> None:
        """Salva la ROI corrente su file JSON."""
        self.roi_manager.save_roi(filepath)

    def start_roi_selection(self) -> None:
        """Avvia la modalità editing ROI."""
        self.roi_manager.start_roi_selection()

    def finalize_roi(self) -> None:
        """Finalizza la ROI corrente."""
        self.roi_manager.finalize_roi()

    def add_roi_point(self, norm_x: float, norm_y: float) -> None:
        """Aggiunge un punto normalizzato alla ROI in editing."""
        self.roi_manager.add_point(norm_x, norm_y)

    @property
    def is_roi_editing(self) -> bool:
        """True se il ROI manager è in modalità editing."""
        return self.roi_manager.editing_mode

    @property
    def roi_points(self) -> List[Tuple[float, float]]:
        """Ritorna i punti ROI correnti."""
        return self.roi_manager.roi_points

    # ── 1. Reader Loop ─────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        """Loop di lettura al framerate nativo."""
        is_file = self.config.get("source_type") == "file"
        frame_duration = 1.0 / self.target_fps
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
                # Se è un file lo limitiamo al framerate corretto
                sleep_time = frame_duration - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                # Per la webcam una micropausa per non affaticare la CPU inutilmente
                time.sleep(0.001)

    # ── 2. Processing Loop ──────────────────────────────────────────────

    def _processing_loop(self) -> None:
        """Prende l'ultimo frame disponibile ed elabora l'AI de-accoppiato."""
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
                obs_frame, debug_frame = self._process_frame(raw_frame)
                with self.frame_lock:
                    self.latest_obs_frame = obs_frame
                    self.latest_debug_frame = debug_frame
                    
                self._update_fps()
            except Exception as e:
                self._log(f"Errore durante processamento frame: {e}")

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pipeline: config → detect → direct → overlay debug."""
        self._apply_config_updates()

        # AI pipeline: ROI → Detector → Director
        ai_frame = self.roi_manager.apply_roi(frame.copy())
        det_out = self.detector.process(ai_frame)
        dir_out = self.director.process(frame, det_out.action_center, det_out.player_spread)

        obs_frame = dir_out.get("cropped_frame", frame)

        # Debug overlay (se attivo)
        debug_frame = frame.copy()
        if self.config.get("debug_mode"):
            self.overlay.draw(debug_frame, det_out, dir_out, self.roi_manager, self.current_fps)

        return obs_frame, debug_frame

    # ── 3. Output Loop ──────────────────────────────────────────────────

    def _output_loop(self) -> None:
        """Manda l'output alle interfacce utente costantemente al framerate nativo."""
        frame_duration = 1.0 / self.target_fps
        
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
        """Aggiorna parametri on-the-fly dagli slider di configurazione."""
        # Interpolazione lineare per i parametri del filtro di Kalman
        percent = self.config.get("kalman_preset_percent") / 100.0
        
        q_smooth = self.config.get("kalman_q_smooth")
        r_smooth = self.config.get("kalman_r_smooth")
        q_reactive = self.config.get("kalman_q_reactive")
        r_reactive = self.config.get("kalman_r_reactive")

        q_std = q_smooth + (q_reactive - q_smooth) * percent
        r_std = r_smooth + (r_reactive - r_smooth) * percent

        self.detector.set_config(q_std, r_std)
        self.director.set_config(
            self.config.get("fixed_zoom_percent"),
            self.config.get("dynamic_zoom_percent")
        )

    def _dispatch_outputs(self, debug_frame: np.ndarray, obs_frame: np.ndarray) -> None:
        """Invia frame alla UI e/o alla virtual camera OBS."""
        if self.on_frame_ready:
            self.on_frame_ready(debug_frame)

        if self.is_outputting_to_obs:
            self.video_output.send_frame(obs_frame)

    def _update_fps(self) -> None:
        """Calcola gli FPS effettivi dell'elaborazione AI."""
        self.frames_processed += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.current_fps = self.frames_processed / elapsed
            self.frames_processed = 0
            self.start_time = time.time()

    def _load_initial_roi(self) -> ROIManager:
        """Carica la ROI dal percorso salvato in configurazione."""
        last_roi = self.config.get("last_roi_path")
        if last_roi and os.path.exists(last_roi):
            return roi_run(last_roi)
        return ROIManager()

    def _log(self, message: str) -> None:
        """Invia messaggi di log al frontend (se connesso) e alla console."""
        logger.info(f"[Backend] {message}")
        if self.on_log_message:
            self.on_log_message(message)

