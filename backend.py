from PySide6.QtCore import QObject, Signal, Slot
from config import App_Config
from output import Video_Thread
from pygrabber.dshow_graph import FilterGraph

class Controller_Video(QObject):
    """Gestisce la logica di business e connette Frontend, Config e Output."""
    
    # Segnali verso la GUI
    invia_frame_gui = Signal(object)
    invia_log_gui = Signal(str)

    def __init__(self):
        super().__init__()
        self.config = App_Config()
        self.config.config_run()
        self.thread_video = None

    def _ottieni_dispositivi(self) -> list:
        """Utility hardware per trovare le webcam."""
        return FilterGraph().get_input_devices()

    def _avvia_flusso_automatico(self):
        """Avvia la preview e la VirtualCam automaticamente."""
        if self.thread_video and self.thread_video.isRunning():
            self.thread_video.ferma()

        self.thread_video = Video_Thread(self.config)
        self.thread_video.frame_pronto_gui.connect(self.invia_frame_gui.emit)
        self.thread_video.log_sistema.connect(self.invia_log_gui.emit)
        self.thread_video.start()

    @Slot(int, str)
    def imposta_sorgente(self, indice: int, path: str = ""):
        """Riceve la selezione della sorgente e avvia automaticamente la preview."""
        sorgente = path if path else indice
        self.config.aggiorna_valore("sorgente", sorgente)
        self.invia_log_gui.emit(f"Sorgente impostata: {sorgente}. Avvio preview in corso...")
        self._avvia_flusso_automatico()

    @Slot()
    def esegui_elaborazione_complessa(self):
        """
        [PLACEHOLDER] Questa è la funzione non ancora definita.
        Si attiverà alla pressione del tasto 'Avvia Elaborazione'.
        """
        # Esempio: aggiorna il config per dire al thread di applicare filtri
        stato_attuale = self.config.ottieni_valore("in_elaborazione", False)
        nuovo_stato = not stato_attuale
        self.config.aggiorna_valore("in_elaborazione", nuovo_stato)
        
        messaggio = "Avviata" if nuovo_stato else "Fermata"
        self.invia_log_gui.emit(f"[*] Funzione di Elaborazione Complessa: {messaggio}")

    def backend_run(self) -> list:
        """Metodo coordinatore per l'avvio logico. Ritorna le webcam disponibili."""
        return self._ottieni_dispositivi()