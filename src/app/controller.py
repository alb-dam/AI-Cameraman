"""Controller dell'applicazione: Threading (4 stadi), Lifecycle e I/O Video."""

import threading
import time
from typing import Optional

import numpy as np


from core.models import FrameMetadata
from core.thread_manager import DropFrameQueue, WorkerThread
from config.settings import SettingsManager
from core.interfaces import IRuntimeState, IPipeline, IVideoInput, IVideoOutput, IPerformanceMonitor, IROIManager
from video.output import VideoOutput
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
        # Usiamo maxsize alto (es. 30) per evitare che i picchi computazionali di YOLO (ogni N frame)
        # causino il drop dei frame antecedenti all'inferenza. Il drop distruggerebbe il pacing del tracker
        # e causerebbe microscatti. Il drop finale avverrà solo se il render loop sarà troppo lento.
        self.q_capture_to_inference = DropFrameQueue(maxsize=30)
        self.q_inference_to_tracking = DropFrameQueue(maxsize=30)
        self.q_tracking_to_render = DropFrameQueue(maxsize=30)

        self.capture_thread: Optional[WorkerThread] = None
        self.inference_thread: Optional[WorkerThread] = None
        self.tracking_thread: Optional[WorkerThread] = None
        self.render_thread: Optional[WorkerThread] = None

        self._empty_frames_count = 0
        self._capture_frame_id = 0
        self._last_reconnect_time: float = 0.0
        self._last_srt_shape: Optional[tuple] = None
        
        # Logging debounce flags
        self._log_file_exhausted: bool = False
        self._srt_reconnecting: bool = False
        self._camera_reconnecting: bool = False
        self._ai_output_error_logged: bool = False
        self._ai_ndi_missing_logged: bool = False
        self._native_output_error_logged: bool = False

    @property
    def roi_manager(self) -> 'IROIManager':
        return self.pipeline.roi_manager

    def generate_roi(self) -> None:
        """Avvia la raccolta di 60 frame equidistanti in 2 minuti e genera la ROI automatica."""
        if not self.state.is_running:
            self._log("Errore: impossibile generare ROI senza sorgente attiva.")
            return

        thread = threading.Thread(target=self._generate_roi_worker, daemon=True)
        thread.start()

    def _generate_roi_worker(self) -> None:
        """Worker thread: raccoglie 60 frame equidistanti in 2 minuti e genera la ROI."""
        import time as _time

        target_frames = 60
        duration_seconds = 20  # 3 minuti
        # Intervallo tra campionamenti: 1 frame ogni 2 secondi
        sample_interval = duration_seconds / target_frames  # = 2.0 secondi

        self._log(f"Genera ROI: raccolta di {target_frames} frame in {duration_seconds}s "
                  f"(1 frame ogni {sample_interval:.1f}s)...")

        collected_frames = []
        for i in range(target_frames):
            if self.stop_event.is_set():
                self._log("Genera ROI: interrotta (pipeline fermata).")
                return

            frame = self.state.latest_raw_frame
            if frame is not None:
                collected_frames.append(frame.copy())

            # Aspetta il prossimo campionamento (tranne l'ultimo)
            if i < target_frames - 1:
                _time.sleep(sample_interval)

        if len(collected_frames) < 2:
            self._log("Genera ROI: frame insufficienti raccolti. Verifica la sorgente video.")
            return

        self._log(f"Genera ROI: {len(collected_frames)} frame raccolti. Avvio segmentazione YOLOE...")
        success = self.roi_manager.generate_roi_from_video(collected_frames, "tmp/roi.json")

        if success:
            self.settings.set("last_roi_path", "tmp/roi.json")
            self._log("ROI generata automaticamente e salvata in tmp/roi.json.")
        else:
            self._log("Genera ROI: fallita. Controllare i log per dettagli.")

    def start(self) -> None:
        if self.state.is_running:
            self.stop()

        source_type = self.settings.get("source_type")
        source_path = self.settings.get("source_path")
        
        self.stop_event.clear()
        
        # Start initialization in a background thread to prevent GUI freeze (especially with SRT)
        init_thread = threading.Thread(
            target=self._async_initialize, 
            args=(source_type, source_path),
            daemon=True
        )
        init_thread.start()

    def _async_initialize(self, source_type: str, source_path: str) -> None:
        """Inizializzazione bloccante eseguita in background."""
        try:
            self._log(f"Inizializzazione sorgente in corso ({source_type})...")
            self.video_input.initialize_source(source_type, source_path)
            if self.stop_event.is_set():
                self.video_input.release()
                return
                
            label = f"file {source_path}" if source_type == "file" else source_type
            self._log(f"Sorgente inizializzata: {label}")
        except Exception as e:
            self._log(f"Avviso: {e}. La pipeline tenterà la riconnessione automatica.")
            # Rimuoviamo il return: vogliamo che i thread si avviino comunque, 
            # così il _capture_loop invocherà _handle_empty_frame e gestirà i retry all'infinito!
            
        if self.stop_event.is_set():
            return

        input_fps = self.video_input.get_fps()
        if input_fps <= 0:
            input_fps = 30.0

        obs_width, obs_height = self.video_input.get_resolution()

        # Inizializza i sender NDI (AI + Native)
        ndi_ai_name = self.settings.get("ndi_ai_name")
        ndi_native_name = self.settings.get("ndi_native_name")
        self.video_output.initialize_ndi(ndi_ai_name, ndi_native_name,
                                         obs_width, obs_height, int(input_fps))

        self.state.reset_for_start(input_fps, obs_width, obs_height, time.time())
        
        self.q_capture_to_inference.clear()
        self.q_inference_to_tracking.clear()
        self.q_tracking_to_render.clear()
        self._empty_frames_count = 0
        self._capture_frame_id = 0
        
        # Pacing accumulator per timer accurato in caso di riproduzione file
        self._render_start_time = time.time()
        self._render_frame_count = 0
        self._capture_start_time = time.time()
        self._capture_frame_count = 0

        self.capture_thread = WorkerThread("CaptureThread", self._capture_loop, self.stop_event, error_callback=self._log_error)
        self.inference_thread = WorkerThread("InferenceThread", self._inference_loop, self.stop_event, error_callback=self._log_error)
        self.tracking_thread = WorkerThread("TrackingThread", self._tracking_loop, self.stop_event, error_callback=self._log_error)
        self.render_thread = WorkerThread("RenderThread", self._render_loop, self.stop_event, error_callback=self._log_error)

        self.capture_thread.start()
        self.inference_thread.start()
        self.tracking_thread.start()
        self.render_thread.start()
        
        self._log(f"Elaborazione avviata (IN/OUT: {input_fps:.1f} FPS @ {obs_width}x{obs_height}).")
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

    def start_ai_output(self) -> bool:
        if getattr(self, "_ai_output_error_logged", False) is False and not self.state.is_running:
            self._log("Errore: impossibile avviare NDI AI senza sorgente.")
            self._ai_output_error_logged = True
            return False
        elif not self.state.is_running:
            return False

        if getattr(self.video_output, "ndi_ai", None) is None:
            if not getattr(self, "_ai_ndi_missing_logged", False):
                self._log("Errore: NDI AI Sender non inizializzato.")
                self._ai_ndi_missing_logged = True
            return False
            
        self._ai_output_error_logged = False
        self._ai_ndi_missing_logged = False
            
        if self.state.is_outputting_ai:
            return True
            
        self.state.is_outputting_ai = True
        self._log("Trasmissione NDI AI avviata.")
        return True

    def stop_ai_output(self) -> None:
        if not self.state.is_outputting_ai:
            return
        self.state.is_outputting_ai = False
        self._log("Trasmissione NDI AI fermata.")

    def start_native_output(self) -> None:
        if not self.state.is_running:
            if not getattr(self, "_native_output_error_logged", False):
                self._log("Errore: impossibile avviare NDI Native senza sorgente.")
                self._native_output_error_logged = True
            return
        
        self._native_output_error_logged = False
            
        if self.state.is_outputting_native:
            return
            
        self.state.is_outputting_native = True
        self.video_output.set_native_enabled(True)
        self._log("Trasmissione NDI Native avviata.")

    def stop_native_output(self) -> None:
        if not self.state.is_outputting_native:
            return
        self.state.is_outputting_native = False
        self.video_output.set_native_enabled(False)
        self._log("Trasmissione NDI Native fermata.")

    def _log(self, message: str) -> None:
        logger.info(f"[Controller] {message}")
        self.state.log(message)

    def _log_error(self, e: Exception) -> None:
        self._log(f"Errore CRITICO in un thread: {e}. Arresto pipeline.")
        self.stop_event.set()

    def _capture_loop(self) -> None:
        """Thread 1: Legge frame, check source, push on Queue 1. Invia anche il frame nativo via NDI."""
        if self.state.source_exhausted:
            self.stop_event.set()
            return
            
        is_file = self.settings.get("source_type") == "file"
        frame_duration = 1.0 / self.state.input_fps

        loop_start = time.time()
        ret, frame = self.video_input.read_frame()
        if not ret:
            self._handle_empty_frame(is_file)
            return

        # Rilevamento cambi di risoluzione mid-stream (es. cambio orientamento)
        if self.settings.get("source_type") == "srt":
            if self._last_srt_shape is None:
                self._last_srt_shape = frame.shape
            elif self._last_srt_shape != frame.shape:
                self._log(f"Rilevato cambio RTP in-stream ({self._last_srt_shape} -> {frame.shape}). Adatto la pipeline...")
                self._last_srt_shape = frame.shape

        self._empty_frames_count = 0
        
        # GIL rende gli assegnamenti di reference atomici — no lock needed
        self.state.latest_raw_frame = frame

        self._send_native_passthrough(frame)
            
        self._capture_frame_id += 1
        frame_id = self._capture_frame_id
        capture_time = time.time()
        
        self.perf_monitor.mark_capture(frame_id)
            
        # Push to inference
        self.q_capture_to_inference.put((frame, frame_id, capture_time))

        if is_file:
            # Pacing perfetto ad accumulatore anche per il capture loop, per evitare
            # di leggere il file file velocemente ed esaurire le DropFrameQueue.
            # Questo assicura che in media si viaggi a 30fps (reali), limitando il rischio
            # che la queue si riempia di 30 frame e ne droppi sfasando il timecode.
            expected_time = getattr(self, "_capture_start_time", time.time()) + getattr(self, "_capture_frame_count", 0) * frame_duration
            now = time.time()
            sleep_time = expected_time - now
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -1.0:
                self._capture_start_time = now - getattr(self, "_capture_frame_count", 0) * frame_duration
                
            self._capture_frame_count = getattr(self, "_capture_frame_count", 0) + 1
        else:
            time.sleep(0.001)

    def _handle_empty_frame(self, is_file: bool) -> None:
        """Gestisce i frame vuoti: stop su file esaurito, riconnessione su sorgente live persa."""
        self._empty_frames_count += 1
        is_srt = self.settings.get("source_type") == "srt"
        
        # Per SRT applichiamo threshold 0 (riavvio istantaneo) per evitare che FFmpeg accetti una
        # nuova connessione sullo stesso socket, incasinando il decoder dopo un cambio orientamento.
        threshold = 0 if is_srt else 30
        
        if self._empty_frames_count > threshold:
            if is_file:
                if not getattr(self, "_log_file_exhausted", False):
                    self._log("Flusso video interrotto (> frame vuoti). Fine del file.")
                    self._log_file_exhausted = True
                self.state.source_exhausted = True
                self.stop_event.set()
            elif is_srt:
                now = time.time()
                # Cooldown ridotto per mantenere la porta in ascolto il più a lungo possibile.
                cooldown = 0.5
                if now - self._last_reconnect_time >= cooldown:
                    self._last_reconnect_time = time.time()
                    if not getattr(self, "_srt_reconnecting", False):
                        self._log("Segnale SRT perso o orientamento cambiato. Riapertura immediata listener...")
                        self._srt_reconnecting = True
                    
                    if hasattr(self.video_input, 'reconnect') and self.video_input.reconnect():
                        self._log("Listener SRT riaperto e segnale ripristinato con successo.")
                        self._empty_frames_count = 0
                        self._srt_reconnecting = False
                        self._last_srt_shape = None
                else:
                    time.sleep(0.05)  # attesa ridotta non bloccante
            else:
                if not getattr(self, "_camera_reconnecting", False):
                    self._log("Segnale perso (>30 frame vuoti). Tentativi continui di riconnessione in background...")
                    self._camera_reconnecting = True
                time.sleep(1.0)
                if hasattr(self.video_input, 'reconnect') and self.video_input.reconnect():
                    self._log("Riconnessione sorgente avvenuta con successo!")
                    self._empty_frames_count = 0
                    self._camera_reconnecting = False
            return
        time.sleep(0.03)

    def _send_native_passthrough(self, frame: np.ndarray) -> None:
        """Invia il frame nativo (passthrough) via NDI se l'output nativo è abilitato."""
        if self.state.is_outputting_native:
            native_frame = VideoOutput.resize_and_pad(
                frame, (self.state.obs_width, self.state.obs_height)
            )
            self.video_output.send_native_frame(native_frame)

    def _inference_loop(self) -> None:
        """Thread 2: Inference YOLO alla risoluzione nativa (Queue 1 -> Queue 2).
        
        Non viene eseguito resize prima dell'inferenza: YOLO, tracking e director
        lavorano tutti sul frame nativo della sorgente. Il resize a risoluzione NDI
        avviene una sola volta nel render thread (resize_and_pad finale).
        """
        raw_frame, frame_id, capture_time = self.q_capture_to_inference.get(timeout=0.1)
        
        try:
            self.perf_monitor.mark_inference(frame_id)
            
            # Lavoriamo alla risoluzione nativa della sorgente.
            # Questo garantisce che il crop dello zoom avvenga con il massimo
            # dettaglio disponibile, specialmente con sorgenti >= risoluzione output.
            working_frame = raw_frame
            
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
        """Thread 4: Rendering GUI e output NDI AI (Consume from Queue 3)."""
        obs_frame, debug_frame, metadata = self.q_tracking_to_render.get(timeout=0.1)
        
        self.perf_monitor.mark_render(metadata.frame_id, metadata.timestamp)
        
        # NDI AI output (indipendente dalla preview)
        if self.state.is_outputting_ai:
            ai_frame = VideoOutput.resize_and_pad(
                obs_frame, (self.state.obs_width, self.state.obs_height)
            )
            self.video_output.send_ai_frame(ai_frame)

        # GUI preview (indipendente dall'output NDI)
        if self.state.on_frame_ready and debug_frame is not None:
            self.state.on_frame_ready(debug_frame)

        self._update_fps()

        # Pacing perfetto per i file video: assorbe tutto il jitter (OS/YOLO) e restituisce
        # una scansione fluida alla GUI, emulando il sync dell'NDI quando spento.
        if self.settings.get("source_type") == "file":
            frame_duration = 1.0 / self.state.input_fps
            expected_time = getattr(self, "_render_start_time", time.time()) + getattr(self, "_render_frame_count", 0) * frame_duration
            now = time.time()
            sleep_time = expected_time - now
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -1.0:
                # Se è rimasto troppo indietro (es. blocco manuale o freeze), resetta il clock
                self._render_start_time = now - getattr(self, "_render_frame_count", 0) * frame_duration
                
            self._render_frame_count = getattr(self, "_render_frame_count", 0) + 1

    def _update_fps(self) -> None:
        self.state.frames_processed += 1
        elapsed = time.time() - self.state.start_time
        if elapsed > 1.0:
            self.state.current_fps = self.state.frames_processed / elapsed
            self.state.frames_processed = 0
            self.state.start_time = time.time()
