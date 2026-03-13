"""Pannello di controllo sinistro: sorgente, debug, slider AI, ROI."""

from typing import Any, Dict, List, Tuple, Union

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QComboBox,
                               QCheckBox, QSlider, QLabel, QGroupBox)
from PySide6.QtCore import Qt, Signal


class ControlPanel(QWidget):
    """Colonna sinistra con tutti i controlli dell'applicazione.

    Emette segnali puri, non conosce Backend né ConfigManager.
    """

    # ── Segnali ─────────────────────────────────────────────────────────

    source_changed = Signal(object)          # (source_type, source_path)
    file_requested = Signal()
    debug_toggled = Signal(bool)
    preview_toggled = Signal(bool)
    native_toggled = Signal(bool)
    fixed_zoom_changed = Signal(int)
    dynamic_zoom_changed = Signal(int)
    kalman_changed = Signal(int)
    load_roi_requested = Signal()
    create_roi_requested = Signal()
    save_roi_requested = Signal()
    generate_roi_requested = Signal()

    # ── Inizializzazione ────────────────────────────────────────────────

    def __init__(self, parent: QWidget = None) -> None:
        """Crea il layout e tutti i widget del pannello controlli."""
        super().__init__(parent)
        self._setup_ui()
        self._connect_internal_signals()

    # ── Setup UI ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        """Costruisce il layout verticale con tutti i gruppi di controlli."""
        layout = QVBoxLayout(self)
        group = QGroupBox("Controlli")
        panel = QVBoxLayout()
        group.setLayout(panel)
        layout.addWidget(group)

        self._setup_source_controls(panel)
        self._setup_ai_sliders(panel)
        self._setup_roi_controls(panel)
        self._setup_output_toggles(panel)
        panel.addStretch()

    def _setup_source_controls(self, panel: QVBoxLayout) -> None:
        """Menu a tendina sorgente video e pulsante file."""
        panel.addWidget(QLabel("Sorgente Video:"))
        self.source_combo = QComboBox()
        panel.addWidget(self.source_combo)

        self.file_btn = QPushButton("Apri File Video")
        self.file_btn.setVisible(False)
        panel.addWidget(self.file_btn)

    def _setup_output_toggles(self, panel: QVBoxLayout) -> None:
        """Gruppo checkbox per Debug, Preview e NDI Nativo."""
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        output_group.setLayout(output_layout)
        panel.addWidget(output_group)

        self.debug_cb = QCheckBox("Modalità Debug (Mostra FPS)")
        output_layout.addWidget(self.debug_cb)

        self.preview_cb = QCheckBox("Mostra Preview")
        self.preview_cb.setChecked(True)
        self.preview_cb.setToolTip("Disattiva per migliorare le performance durante lo streaming")
        output_layout.addWidget(self.preview_cb)

        self.native_cb = QCheckBox("NDI Output Nativo")
        self.native_cb.setToolTip("Attiva/disattiva l'invio del frame nativo (passthrough) via NDI")
        output_layout.addWidget(self.native_cb)

    def _setup_ai_sliders(self, panel: QVBoxLayout) -> None:
        """Slider per zoom fisso, zoom dinamico e preset Kalman."""
        panel.addWidget(QLabel("Zoom Fisso (1x - 3x)"))
        self.fixed_zoom_slider = QSlider(Qt.Horizontal)
        self.fixed_zoom_slider.setRange(0, 100)
        panel.addWidget(self.fixed_zoom_slider)

        panel.addWidget(QLabel("Intensità Zoom Dinamico"))
        self.dynamic_zoom_slider = QSlider(Qt.Horizontal)
        self.dynamic_zoom_slider.setRange(0, 100)
        panel.addWidget(self.dynamic_zoom_slider)

        panel.addWidget(QLabel("Reattività Kalman (Lento -> Reattivo)"))
        self.kalman_slider = QSlider(Qt.Horizontal)
        self.kalman_slider.setRange(0, 100)
        self.kalman_slider.setTickPosition(QSlider.NoTicks)
        panel.addWidget(self.kalman_slider)

    def _setup_roi_controls(self, panel: QVBoxLayout) -> None:
        """Pulsanti per gestione area ROI."""
        roi_group = QGroupBox("Gestione Area (ROI)")
        roi_layout = QVBoxLayout()
        roi_group.setLayout(roi_layout)
        panel.addWidget(roi_group)

        self.btn_load_roi = QPushButton("Carica ROI")
        self.btn_create_roi = QPushButton("Crea ROI")
        self.btn_generate_roi = QPushButton("Genera ROI")
        self.btn_save_roi = QPushButton("Salva ROI")
        roi_layout.addWidget(self.btn_load_roi)
        roi_layout.addWidget(self.btn_create_roi)
        roi_layout.addWidget(self.btn_generate_roi)
        roi_layout.addWidget(self.btn_save_roi)

    # ── Connessioni interne ─────────────────────────────────────────────

    def _connect_internal_signals(self) -> None:
        """Collega i widget interni ai segnali pubblici del pannello."""
        self.source_combo.currentIndexChanged.connect(self._on_source_index_changed)
        self.file_btn.clicked.connect(self.file_requested.emit)
        self.debug_cb.toggled.connect(self.debug_toggled.emit)
        self.preview_cb.toggled.connect(self.preview_toggled.emit)
        self.native_cb.toggled.connect(self.native_toggled.emit)
        self.fixed_zoom_slider.valueChanged.connect(self.fixed_zoom_changed.emit)
        self.dynamic_zoom_slider.valueChanged.connect(self.dynamic_zoom_changed.emit)
        self.kalman_slider.valueChanged.connect(self.kalman_changed.emit)
        self.btn_load_roi.clicked.connect(self.load_roi_requested.emit)
        self.btn_create_roi.clicked.connect(self.create_roi_requested.emit)
        self.btn_save_roi.clicked.connect(self.save_roi_requested.emit)
        self.btn_generate_roi.clicked.connect(self.generate_roi_requested.emit)

    def _on_source_index_changed(self, index: int) -> None:
        """Emette source_changed con i dati associati all'item selezionato."""
        data = self.source_combo.itemData(index)
        if data is not None:
            self.source_changed.emit(data)

    # ── API pubblica ────────────────────────────────────────────────────

    def populate_sources(self, cameras: Union[Dict[int, str], List[str]],
                         saved_type: str, saved_path: str) -> None:
        """Popola il combobox sorgente con le webcam disponibili.

        Args:
            cameras: dict {index: name} oppure list di nomi.
            saved_type: tipo sorgente salvato in config ("webcam" o "file").
            saved_path: percorso sorgente salvato in config.
        """
        self.source_combo.blockSignals(True)
        self.source_combo.clear()

        idx_to_select = 0

        if isinstance(cameras, dict):
            for i, cam_name in cameras.items():
                self.source_combo.addItem(f"{cam_name}", userData=("webcam", i))
                if saved_type == "webcam" and str(i) == str(saved_path):
                    idx_to_select = self.source_combo.count() - 1
        else:
            for i, cam_name in enumerate(cameras):
                self.source_combo.addItem(f"{cam_name}", userData=("webcam", i))
                if saved_type == "webcam" and str(i) == str(saved_path):
                    idx_to_select = self.source_combo.count() - 1

        self.source_combo.addItem("Ricevitore SRT (Porta 9999)", userData=("srt", 9999))
        if saved_type == "srt":
            idx_to_select = self.source_combo.count() - 1

        self.source_combo.addItem("File Video...", userData=("file", saved_path))
        if saved_type == "file":
            idx_to_select = self.source_combo.count() - 1
            self.file_btn.setVisible(True)

        self.source_combo.setCurrentIndex(idx_to_select)
        self.source_combo.blockSignals(False)

    def restore_config(self, debug: bool, fixed_zoom: int,
                       dynamic_zoom: int, kalman: int) -> None:
        """Ripristina i valori salvati della configurazione nei widget.

        Args:
            debug: stato checkbox debug.
            fixed_zoom: valore slider zoom fisso (0-100).
            dynamic_zoom: valore slider zoom dinamico (0-100).
            kalman: valore slider preset Kalman (0-100).
        """
        self.debug_cb.setChecked(debug)
        self.fixed_zoom_slider.setValue(fixed_zoom)
        self.dynamic_zoom_slider.setValue(dynamic_zoom)
        self.kalman_slider.setValue(kalman)

    def set_file_button_visible(self, visible: bool) -> None:
        """Mostra/nasconde il pulsante di selezione file."""
        self.file_btn.setVisible(visible)


def control_panel_run(parent: QWidget = None) -> ControlPanel:
    """Crea e ritorna un'istanza di ControlPanel."""
    return ControlPanel(parent)
