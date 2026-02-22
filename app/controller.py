"""Controller dell'applicazione: Threading (4 stadi), Lifecycle e I/O Video."""

import threading
import time
from typing import Optional


from core.models import FrameMetadata
from core.thread_manager import DropFrameQueue, WorkerThread
from config.settings import SettingsManager
from core.interfaces import IRuntimeState, IPipeline, IVideoInput, IVideoOutput, IPerformanceMonitor
from app.logger import get_logger

logger = get_logger(__name__)


class ApplicationController:
    """Gestisce il ciclo di vita e i 4 thread per l'I/O video e AI."""

    def __init__(
        self, 
        settings_manager: SettingsManager, 
        state: IRuntimeState, 
        pipeline: IPipeline,
        video_input: IVideoInput,
        video_output: IVideoOutput,
        perf_monitor: IPerformanceMonitor
    ) -> None:
        self.settings = settings_manager
        self.state = state
        self.pipeline = pipeline

        self.video_input = video_input
        self.video_output = video_output
        self.perf_monitor = perf_monitor

        self.stop_event = threading.Event()
        
        # Le code DropFrame per garantire latenza minima ed evitare accumulo
        self.q_capture_to_inference = DropFrameQueue(maxsize=2)
        self.q_inference_to_tracking = DropFrameQueue(maxsize=2)
        self.q_tracking_to_render = DropFrameQueue(maxsize=2)

        self.capture_thread: Optional[WorkerThread] = None
        self.inference_thread: Optional[WorkerThread] = None
        self.tracking_thread: Optional[WorkerThread] = None
        self.render_thread: Optional[WorkerThread] = None

        self._empty_frames_count = 0
        self._capture_frame_id = 0

    @property
    def roi_manager(self) -> 'IROIManager':
        return self.pipeline.roi_manager

    def start(self) -> None:
        if self.state.is_running:
            self.stop()

        source_type = self.settings.get("source_type")
        source_path = self.settings.get("source_path")

        try:
            self.video_input.initialize_source(source_type, source_path)
            label = f"file {source_path}" if source_type == "file" else "webcam"
            self._log(f"Sorgente inizializzata: {label}")
        except Exception as e:
            self._log(f"Errore inizializzazione sorgente: {e}")
            return

        input_fps = self.video_input.get_fps()
        if input_fps <= 0:
            input_fps = 30.0

        output_fps = self.settings.get("output_fps")
        obs_width = self.settings.get("output_width")
        obs_height = self.settings.get("output_height")

        self.video_output.initialize_virtual_camera(obs_width, obs_height, int(output_fps))

        self.state.reset_for_start(input_fps, output_fps, obs_width, obs_height, time.time())
        self.stop_event.clear()
        
        self.q_capture_to_inference.clear()
        self.q_inference_to_tracking.clear()
        self.q_tracking_to_render.clear()
        self._empty_frames_count = 0
        self._capture_frame_id = 0

        self.capture_thread = WorkerThread("CaptureThread", self._capture_loop, self.stop_event, error_callback=self._log_error)
        self.inference_thread = WorkerThread("InferenceThread", self._inference_loop, self.stop_event, error_callback=self._log_error)
        self.tracking_thread = WorkerThread("TrackingThread", self._tracking_loop, self.stop_event, error_callback=self._log_error)
        self.render_thread = WorkerThread("RenderThread", self._render_loop, self.stop_event, error_callback=self._log_error)

        self.capture_thread.start()
        self.inference_thread.start()
        self.tracking_thread.start()
        self.render_thread.start()
        
        self._log(f"Elaborazione avviata (IN: {input_fps:.1f} FPS, OUT: {output_fps} FPS @ {obs_width}x{obs_height}).")
        self._log("Pipeline multithread a 4 stadi attiva (C -> I -> T -> R).")

    def stop(self) -> None:
        if not self.state.is_running:
            return

        self._log("Chiusura elaborazione in corso...")
        self.state.is_running = False
        self.stop_event.set()

        threads = [self.capture_thread, self.inference_thread, self.tracking_thread, self.render_thread]
        for t in threads:
            if t and t.is_alive():
                t.join(timeout=2.0)

        try:
            self.video_input.release()
        except Exception as e:
            self._log(f"Errore durante il rilascio di video_input: {e}")
        finally:
            self._log("Elaborazione chiusa in sicurezza.")

    def close_all(self) -> None:
        try:
            self.stop()
        finally:
            try:
                self.video_output.close()
            except Exception as e:
                self._log(f"Errore durante la chiusura di video_output: {e}")

    def start_obs_output(self) -> bool:
        if not self.state.is_running:
            self._log("Errore: impossibile avviare l'elaborazione OBS senza sorgente.")
            return False
        if getattr(self.video_output, "cam", None) is None:
            self._log("Errore: Virtual Camera non inizializzata. Controllare driver OBS.")
            return False
            
        self.state.is_outputting_to_obs = True
        self._log("Trasmissione verso OBS avviata.")
        return True

    def stop_obs_output(self) -> None:
        self.state.is_outputting_to_obs = False
        self._log("Trasmissione verso OBS fermata.")

    def _log(self, message: str) -> None:
        logger.info(f"[Controller] {message}")
        self.state.log(message)

    def _log_error(self, e: Exception) -> None:
        self._log(f"Errore CRITICO in un thread: {e}. Arresto pipeline.")
        self.stop_event.set()

    def _capture_loop(self) -> None:
        """Thread 1: Legge frame, check source, push on Queue 1."""
        if self.state.source_exhausted:
            self.stop_event.set()
            return
            
        is_file = self.settings.get("source_type") == "file"
        frame_duration = 1.0 / self.state.input_fps

        loop_start = time.time()
        ret, frame = self.video_input.read_frame()
        if not ret:
            self._empty_frames_count += 1
            if self._empty_frames_count > 30:
                if is_file:
                    self._log("Flusso video interrotto (>30 frame vuoti). Fine del file.")
                    self.state.source_exhausted = True
                    self.stop_event.set()
                else:
                    self._log("Segnale perso (>30 frame vuoti). Tentativo di riconnessione...")
                    time.sleep(1.0)
                    if hasattr(self.video_input, 'reconnect') and self.video_input.reconnect():
                        self._log("Riconnessione avvenuta con successo!")
                        self._empty_frames_count = 0
                    else:
                        self._log("Riconnessione fallita. Nuovo tentativo al prossimo ciclo.")
                return
            time.sleep(0.03)
            return

        self._empty_frames_count = 0
        
        # GIL rende gli assegnamenti di reference atomici — no lock needed
        self.state.latest_raw_frame = frame
            
        self._capture_frame_id += 1
        frame_id = self._capture_frame_id
        capture_time = time.time()
        
        self.perf_monitor.mark_capture(frame_id)
            
        # Push to inference
        self.q_capture_to_inference.put((frame, frame_id, capture_time))

        if is_file:
            sleep_time = frame_duration - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
        else:
            time.sleep(0.001)

    def _inference_loop(self) -> None:
        """Thread 2: Resize + Inference YOLO (Queue 1 -> Queue 2)."""
        raw_frame, frame_id, capture_time = self.q_capture_to_inference.get(timeout=0.1)
        
        try:
            self.perf_monitor.mark_inference(frame_id)
            
            # Resize a risoluzione OBS affinché le coordinate
            # calcolate da YOLO combacino con quelle della Regia.
            working_frame = self.video_output._resize_and_pad(
                raw_frame, (self.state.obs_width, self.state.obs_height)
            )
            
            det_out = self.pipeline.run_inference(working_frame)
            self.q_inference_to_tracking.put((working_frame, det_out, frame_id, capture_time))
        except Exception as e:
            self._log(f"Errore in Inferenza: {e}")

    def _tracking_loop(self) -> None:
        """Thread 3: Tracking Kalman e Regia (Queue 2 -> Queue 3)."""
        working_frame, det_out, frame_id, capture_time = self.q_inference_to_tracking.get(timeout=0.1)
        
        try:
            metadata = FrameMetadata(
                frame_id=frame_id,
                timestamp=capture_time,
                fps=self.state.current_fps,
                original_size=(working_frame.shape[1], working_frame.shape[0])
            )

            obs_frame, debug_frame = self.pipeline.run_tracking_and_directing(working_frame, det_out, metadata)
            
            # GIL rende gli assegnamenti di reference atomici — no lock needed
            self.state.latest_obs_frame = obs_frame
            self.state.latest_debug_frame = debug_frame
                
            self.q_tracking_to_render.put((obs_frame, debug_frame, metadata))
        except Exception as e:
            self._log(f"Errore in Tracking/Directing: {e}")

    def _render_loop(self) -> None:
        """Thread 4: Rendering GUI e output OBS (Consume from Queue 3)."""
        frame_duration = 1.0 / self.state.output_fps
        loop_start = time.time()
        
        obs_frame, debug_frame, metadata = self.q_tracking_to_render.get(timeout=0.1)
        
        self.perf_monitor.mark_render(metadata.frame_id, metadata.timestamp)
        
        if self.state.is_outputting_to_obs:
            self.video_output.send_frame(obs_frame)
            # Skip GUI preview refresh to optimize performance
        elif self.state.on_frame_ready:
            self.state.on_frame_ready(debug_frame)

        self._update_fps()

        sleep_time = frame_duration - (time.time() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    def _update_fps(self) -> None:
        self.state.frames_processed += 1
        elapsed = time.time() - self.state.start_time
        if elapsed > 1.0:
            self.state.current_fps = self.state.frames_processed / elapsed
            self.state.frames_processed = 0
            self.state.start_time = time.time()
