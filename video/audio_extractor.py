"""Modulo per l'estrazione asincrona dell'audio tramite ffmpeg in subprocess."""

import subprocess
import threading
import numpy as np
import queue
from typing import Optional

from app.logger import get_logger

logger = get_logger(__name__)


class AudioExtractorThread(threading.Thread):
    """Estrae l'audio da un URL usando ffmpeg in un thread separato e lo mette in una coda.
    
    Usa il flag `-re` per leggere i file alla velocità nativa (real-time),
    evitando che ffmpeg decodifichi tutta la traccia audio in pochi secondi.
    """

    def __init__(self, source_url: str, is_srt: bool = False):
        super().__init__(name="AudioExtractor", daemon=True)
        self.source_url = source_url
        self.is_srt = is_srt
        
        # Coda per i chunk audio — abbastanza grande da coprire qualche secondo di buffer
        # ma non infinita per evitare memory leak su stream lunghi
        self.audio_queue: queue.Queue = queue.Queue(maxsize=600)
        
        self.running = False
        self.process: Optional[subprocess.Popen] = None
        
        # Audio params richiesti da NDI
        self.sample_rate = 48000
        self.channels = 2
        self.bytes_per_sample = 4  # float32
        
        # Chunk da 1600 samples → ~33ms a 48kHz (≈30fps di video)
        self.samples_per_chunk = 1600
        self.chunk_size_bytes = self.samples_per_chunk * self.channels * self.bytes_per_sample

    def run(self):
        self.running = True
        
        # Costruisci il comando ffmpeg
        input_args = []
        
        if self.is_srt:
            # Per SRT non usiamo -re, lo stream arriva già in tempo reale
            input_args = ["-i", f"{self.source_url}?mode=listener&transtype=live"]
        else:
            # Per file locali: -re forza la lettura in tempo reale
            # Senza -re, ffmpeg decodifica l'intera traccia in ~2 secondi
            input_args = ["-re", "-i", self.source_url]

        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",
            "-y",
        ] + input_args + [
            "-vn",                         # Scarta video
            "-acodec", "pcm_f32le",        # Float 32 bit little endian
            "-ar", str(self.sample_rate),   # 48000 Hz
            "-ac", str(self.channels),      # Stereo
            "-f", "f32le",                 # Formato RAW
            "pipe:1"                       # Output su stdout
        ]
        
        logger.info(f"[AudioExtractor] Avvio ffmpeg: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.chunk_size_bytes * 4  # Buffer ragionevole
            )
            
            # Thread separato per leggere stderr e loggare eventuali errori ffmpeg
            stderr_thread = threading.Thread(
                target=self._read_stderr, daemon=True
            )
            stderr_thread.start()
            
            while self.running and self.process and self.process.stdout:
                raw_data = self.process.stdout.read(self.chunk_size_bytes)
                
                if not raw_data:
                    break
                
                if len(raw_data) < self.chunk_size_bytes:
                    # Chunk parziale a fine file — padding a zero per evitare reshape error
                    raw_data = raw_data + b'\x00' * (self.chunk_size_bytes - len(raw_data))
                    
                # Convertiamo in numpy array float32
                audio_np = np.frombuffer(raw_data, dtype=np.float32)
                
                # Reshape: NDI vuole (canali, campioni) es. (2, 1600)
                # ffmpeg outputta interlacciato L R L R ...
                audio_np = audio_np.reshape(-1, self.channels).T
                
                # Contiguous in memoria (Cython/C++ wrappers lo richiedono)
                audio_np = np.ascontiguousarray(audio_np)
                
                try:
                    # Bloccare brevemente se la coda è piena — il video loop fa da pacemaker
                    self.audio_queue.put(audio_np, block=True, timeout=0.5)
                except queue.Full:
                    # Se il consumer è troppo lento, scartiamo il chunk più vecchio
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.audio_queue.put_nowait(audio_np)
                
        except Exception as e:
            logger.error(f"[AudioExtractor] Errore: {e}")
            
        finally:
            logger.info("[AudioExtractor] Thread terminato.")

    def _read_stderr(self):
        """Legge stderr di ffmpeg e logga eventuali errori/warning."""
        try:
            if self.process and self.process.stderr:
                for line in self.process.stderr:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if decoded:
                        logger.warning(f"[AudioExtractor/ffmpeg] {decoded}")
        except Exception:
            pass

    def get_next_chunk(self, block: bool = False, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Recupera il prossimo chunk audio disponibile. Non blocca di default."""
        try:
            return self.audio_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Ferma il thread e fa terminare il processo ffmpeg."""
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
