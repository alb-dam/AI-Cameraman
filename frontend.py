import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QComboBox, QPushButton, QLabel, QTextEdit, QSizePolicy)
from PySide6.QtCore import Qt, Slot, Signal, QObject
from PySide6.QtGui import QImage, QPixmap

# -------------------------------------------------------------------
# CLASSE FRONTEND
# -------------------------------------------------------------------

class Pannello_Controllo_Video:
    """
    Interfaccia Utente (View).
    Gestisce il layout a due colonne: Sinistra (Sorgenti) e Destra (Preview, Controlli, Log).
    """

    def __init__(self, controller):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.window = QMainWindow()
        self.controller = controller

    # ==========================================
    # 1. FUNZIONI DI COSTRUZIONE UI (Atomiche)
    # ==========================================

    def _imposta_finestra_principale(self) -> QHBoxLayout:
        """Configura la finestra e restituisce il layout principale a DUE COLONNE (Orizzontale)."""
        self.window.setWindowTitle("Pannello Video Modulare - Layout a Colonne")
        self.window.setMinimumSize(1280, 720) # Finestra più larga per ospitare le colonne
        
        central_widget = QWidget()
        self.window.setCentralWidget(central_widget)
        
        # Layout principale orizzontale (Sinistra / Destra)
        return QHBoxLayout(central_widget)

    def _costruisci_colonna_sinistra(self, layout_principale: QHBoxLayout):
        """Crea la colonna di sinistra contenente SOLO il menu a tendina."""
        lay_sinistra = QVBoxLayout()
        lay_sinistra.setAlignment(Qt.AlignmentFlag.AlignTop) # Spinge gli elementi in alto
        
        self.combo_sorgenti = QComboBox()
        self.combo_sorgenti.addItem("Seleziona una sorgente...")
        self.combo_sorgenti.setFixedWidth(250)
        self.combo_sorgenti.setStyleSheet("padding: 5px; font-size: 14px;")
        
        lay_sinistra.addWidget(self.combo_sorgenti)
        
        # Aggiungo il layout di sinistra al layout principale
        layout_principale.addLayout(lay_sinistra)

    def _costruisci_colonna_destra(self, layout_principale: QHBoxLayout):
        """Crea la colonna di destra assemblando preview, pulsante e log."""
        lay_destra = QVBoxLayout()
        
        # Costruisco i sub-componenti della colonna destra
        self._costruisci_preview(lay_destra)
        self._costruisci_pulsante_avvio_ferma(lay_destra)
        self._costruisci_log(lay_destra)
        
        # Aggiungo il layout di destra al layout principale
        layout_principale.addLayout(lay_destra)

    def _costruisci_preview(self, layout_genitore: QVBoxLayout):
        """Genera il riquadro della telecamera a grandezza fissa."""
        lay_preview = QHBoxLayout()
        
        self.lbl_preview = QLabel("In attesa di sorgente...")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet("background-color: #111; color: #aaa; border: 2px solid #444; font-size: 18px;")
        self.lbl_preview.setFixedSize(960, 540) # Dimensioni fisse richieste
        
        # Centra la preview orizzontalmente nel suo spazio
        lay_preview.addStretch()
        lay_preview.addWidget(self.lbl_preview)
        lay_preview.addStretch()
        
        layout_genitore.addLayout(lay_preview)

    def _costruisci_pulsante_avvio_ferma(self, layout_genitore: QVBoxLayout):
        """Genera il pulsante per avviare o fermare l'elaborazione."""
        lay_pulsante = QHBoxLayout()
        
        self.btn_elabora = QPushButton("Avvia/Ferma Elaborazione")
        self.btn_elabora.setFixedSize(300, 45)
        self.btn_elabora.setStyleSheet("font-weight: bold; font-size: 15px; background-color: #2b5797; color: white;")
        
        # Centra il pulsante orizzontalmente
        lay_pulsante.addStretch()
        lay_pulsante.addWidget(self.btn_elabora)
        lay_pulsante.addStretch()
        
        lay_pulsante.setContentsMargins(0, 15, 0, 15) # Margini sopra e sotto il pulsante
        layout_genitore.addLayout(lay_pulsante)

    def _costruisci_log(self, layout_genitore: QVBoxLayout):
            """Genera l'area di testo per i log di sistema con font-size esplicito per evitare warning."""
            self.txt_log = QTextEdit()
            self.txt_log.setReadOnly(True)
            
            # FIX: Aggiunto 'font-size: 11pt;' per evitare l'errore QFont::setPointSize
            self.txt_log.setStyleSheet(
                "font-family: Consolas, 'Courier New', monospace; "
                "font-size: 11pt; "
                "background-color: #f5f5f5;"
            )
            self.txt_log.setPlaceholderText("Log degli eventi di sistema...")
            
            layout_genitore.addWidget(self.txt_log)

    # ==========================================
    # 2. SEGNALI E AGGIORNAMENTO
    # ==========================================

    def _connetti_segnali(self):
        """Collega le azioni utente ai comandi del controller, e viceversa."""
        self.combo_sorgenti.currentIndexChanged.connect(
            lambda idx: self.controller.imposta_sorgente(idx - 1) if idx > 0 else None
        )
        self.btn_elabora.clicked.connect(self.controller.esegui_elaborazione_complessa)

        self.controller.invia_frame_gui.connect(self._aggiorna_preview)
        self.controller.invia_log_gui.connect(self.txt_log.append)

    @Slot(object)
    def _aggiorna_preview(self, frame_rgb):
        """Dipinge il frame sulla QLabel mantenendo l'aspect ratio."""
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.lbl_preview.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.lbl_preview.size(), Qt.AspectRatioMode.KeepAspectRatio
        ))

    # ==========================================
    # 3. METODO COORDINATORE
    # ==========================================

    def gui_run(self):
        """
        Metodo principale (Runner).
        Assembla le colonne, connette la logica e avvia l'interfaccia.
        """
        # 1. Imposto il layout radice a due colonne
        layout_principale = self._imposta_finestra_principale()
        
        # 2. Assemblo la colonna di sinistra e poi quella di destra
        self._costruisci_colonna_sinistra(layout_principale)
        self._costruisci_colonna_destra(layout_principale)
        
        # 3. Connetto i segnali UI-Backend
        self._connetti_segnali()
        
        # 4. Inizializzo i dati
        dispositivi = self.controller.backend_run()
        self.combo_sorgenti.addItems(dispositivi)
        
        # 5. Avvio il loop
        self.window.show()
        sys.exit(self.app.exec())


