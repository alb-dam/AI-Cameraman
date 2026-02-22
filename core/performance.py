"""Modulo per il monitoraggio delle performance della pipeline."""

import time
import logging
from collections import deque
from core.interfaces import IPerformanceMonitor
from config.settings import SettingsManager
from app.logger import get_logger

logger = get_logger(__name__)

class PerformanceMonitor(IPerformanceMonitor):
    """Calcola FPS per stage e latenza end-to-end senza bloccare la pipeline."""

    def __init__(self, settings: SettingsManager) -> None:
        self.settings = settings
        self.enabled = bool(self.settings.get("enable_performance_monitor"))
        
        # Buffer per i tempi: usiamo deques con limite fisso per memoria O(1)
        self._max_history = 60
        self._capture_times = deque(maxlen=self._max_history)
        self._inference_times = deque(maxlen=self._max_history)
        self._render_times = deque(maxlen=self._max_history)
        
        # Latenza end-to-end dell'ultimo frame elaborato
        self._last_latency = 0.0
        
        # Gestione del log periodico
        self._last_log_time = time.time()
        self._log_interval = 2.0  # Log statistics every 2 seconds

    def mark_capture(self, frame_id: int) -> None:
        if not self.enabled: return
        self._capture_times.append(time.time())

    def mark_inference(self, frame_id: int) -> None:
        if not self.enabled: return
        self._inference_times.append(time.time())

    def mark_render(self, frame_id: int, capture_time: float) -> None:
        if not self.enabled: return
        now = time.time()
        self._render_times.append(now)
        
        if capture_time > 0:
            self._last_latency = (now - capture_time) * 1000.0  # in ms
            
        if now - self._last_log_time >= self._log_interval:
            self.log_stats()
            self._last_log_time = now

    def log_stats(self) -> None:
        if not self.enabled: return
        
        cap_fps = self._calculate_fps(self._capture_times)
        inf_fps = self._calculate_fps(self._inference_times)
        rnd_fps = self._calculate_fps(self._render_times)
        
        # Determinazione collo di bottiglia semplice (se i valori sono > 0)
        fps_dict = {"Capture": cap_fps, "Inference": inf_fps, "Render": rnd_fps}
        bottleneck = min(fps_dict, key=lambda k: fps_dict[k]) if any(v > 0 for v in fps_dict.values()) else "N/A"
        
        logger.info(
            f"[PerformanceMonitor] Latency: {self._last_latency:.1f}ms | "
            f"Cap FPS: {cap_fps:.1f} | Inf FPS: {inf_fps:.1f} | Rnd FPS: {rnd_fps:.1f} | "
            f"Bottleneck: {bottleneck}"
        )

    def _calculate_fps(self, times_deque: deque) -> float:
        """Calcola gli FPS basandosi sull'intervallo temporale nella deque."""
        if len(times_deque) < 2:
            return 0.0
        
        # Tempo dal primo elemento della storia all'ultimo inserito
        time_diff = times_deque[-1] - times_deque[0]
        if time_diff <= 0:
            return 0.0
            
        return (len(times_deque) - 1) / time_diff
