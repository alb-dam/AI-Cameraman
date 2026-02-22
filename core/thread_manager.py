"""Gestore dei thread e code thread-safe per la pipeline video."""

import queue
import threading
from typing import Any, Callable, Optional


class DropFrameQueue:
    """Una coda thread-safe che implementa una drop-frame policy.
    
    Se la coda raggiunge la dimensione massima (default: 2), l'inserimento
    di un nuovo elemento comporta l'eliminazione automatica di quello più vecchio,
    garantendo che venga elaborato solo il dato più recente e prevenendo backlog.
    """

    def __init__(self, maxsize: int = 2) -> None:
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, item: Any) -> None:
        """Inserisce un elemento, rimuovendo il più vecchio se la coda è piena."""
        # Se la coda è piena, cerchiamo di liberare spazio
        while True:
            try:
                self.q.put_nowait(item)
                break
            except queue.Full:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass

    def get(self, timeout: Optional[float] = None) -> Any:
        """Recupera l'elemento più vecchio in coda."""
        return self.q.get(timeout=timeout)

    def empty(self) -> bool:
        """Ritorna True se la coda è vuota."""
        return self.q.empty()

    def clear(self) -> None:
        """Svuota completamente la coda."""
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break


class WorkerThread(threading.Thread):
    """Thread worker generico con gestione dello spegnimento pulito ed eccezioni."""

    def __init__(
        self,
        name: str,
        target_func: Callable[[], None],
        stop_event: threading.Event,
        error_callback: Optional[Callable[[Exception], None]] = None
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.target_func = target_func
        self.stop_event = stop_event
        self.error_callback = error_callback

    def run(self) -> None:
        """Esegue il target function finché non viene segnalato lo stop_event."""
        while not self.stop_event.is_set():
            try:
                self.target_func()
            except queue.Empty:
                # Normale durante i timeout delle code
                pass
            except Exception as e:
                if self.error_callback:
                    self.error_callback(e)
                else:
                    print(f"Errore non gestito nel thread {self.name}: {e}")
                
                # Se l'errore è grave, potrebbe essere necessario interrompere il thread,
                # ma per ora lo manteniamo vivo e logghiamo. 
                # Se lo sviluppatore vuole fermarsi, dovrebbe impostare lo stop_event nel callback.
