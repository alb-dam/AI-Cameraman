"""Pannello di controllo compatto: sorgente, slider AI, ROI, output.

Layout a griglia orizzontale, ottimizzato per finestre piccole (640×360).
"""

from typing import Dict, List, Union, Optional

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QComboBox, QCheckBox, QSlider, QLabel,
                               QGridLayout, QSizePolicy)
from PySide6.QtCore import Qt, Signal


class ControlPanel(QWidget):
    """Pannello controlli compatto con layout a righe orizzontali.

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
    deadzone_changed = Signal(int)
    inertia_changed = Signal(int)
    load_roi_requested = Signal()
    create_roi_requested = Signal()
    save_roi_requested = Signal()
    generate_roi_requested = Signal()

    # ── Inizializzazione ────────────────────────────────────────────────

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Crea il layout e tutti i widget del pannello controlli."""
        super().__init__(parent)
        self._setup_ui()
        self._connect_internal_signals()

    # ── Setup UI ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        """Costruisce il layout compatto a righe orizzontali."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Riga 1: Sorgente + Output checkboxes
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(QLabel("Sorgente:"))
        self.source_combo = QComboBox()
        self.source_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # type: ignore
        row1.addWidget(self.source_combo)
        self.file_btn = QPushButton("Apri File")
        self.file_btn.setVisible(False)
        row1.addWidget(self.file_btn)

        row1.addSpacing(12)
        self.debug_cb = QCheckBox("Debug")
        self.debug_cb.setToolTip("Modalità Debug (Mostra FPS)")
        row1.addWidget(self.debug_cb)
        self.preview_cb = QCheckBox("Preview")
        self.preview_cb.setChecked(True)
        self.preview_cb.setToolTip("Disattiva per migliorare le performance durante lo streaming")
        row1.addWidget(self.preview_cb)
        self.native_cb = QCheckBox("NDI Nativo")
        self.native_cb.setToolTip("Attiva/disattiva l'invio del frame nativo (passthrough) via NDI")
        row1.addWidget(self.native_cb)
        layout.addLayout(row1)

        # Riga 2: Slider (griglia 2×2)
        slider_grid = QGridLayout()
        slider_grid.setSpacing(4)

        slider_grid.addWidget(QLabel("Zoom Fisso:"), 0, 0)
        self.fixed_zoom_slider = QSlider(Qt.Horizontal)  # type: ignore
        self.fixed_zoom_slider.setRange(0, 100)
        slider_grid.addWidget(self.fixed_zoom_slider, 0, 1)

        slider_grid.addWidget(QLabel("Zoom Dinamico:"), 0, 2)
        self.dynamic_zoom_slider = QSlider(Qt.Horizontal)  # type: ignore
        self.dynamic_zoom_slider.setRange(0, 100)
        slider_grid.addWidget(self.dynamic_zoom_slider, 0, 3)

        slider_grid.addWidget(QLabel("Tolleranza:"), 1, 0)
        self.deadzone_slider = QSlider(Qt.Horizontal)  # type: ignore
        self.deadzone_slider.setRange(0, 100)
        slider_grid.addWidget(self.deadzone_slider, 1, 1)

        slider_grid.addWidget(QLabel("Velocità Regia:"), 1, 2)
        self.inertia_slider = QSlider(Qt.Horizontal)  # type: ignore
        self.inertia_slider.setRange(0, 100)
        slider_grid.addWidget(self.inertia_slider, 1, 3)

        # Le colonne slider si espandono, le label no
        slider_grid.setColumnStretch(1, 1)
        slider_grid.setColumnStretch(3, 1)
        layout.addLayout(slider_grid)

        # Riga 3: ROI buttons compatti
        roi_row = QHBoxLayout()
        roi_row.setSpacing(4)
        roi_row.addWidget(QLabel("ROI:"))
        self.btn_load_roi = QPushButton("Carica")
        self.btn_create_roi = QPushButton("Crea")
        self.btn_generate_roi = QPushButton("Genera")
        self.btn_save_roi = QPushButton("Salva")
        roi_row.addWidget(self.btn_load_roi)
        roi_row.addWidget(self.btn_create_roi)
        roi_row.addWidget(self.btn_generate_roi)
        roi_row.addWidget(self.btn_save_roi)
        roi_row.addStretch()
        layout.addLayout(roi_row)

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
        self.deadzone_slider.valueChanged.connect(self.deadzone_changed.emit)
        self.inertia_slider.valueChanged.connect(self.inertia_changed.emit)
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
                       dynamic_zoom: int, deadzone: int,
                       inertia: int) -> None:
        """Ripristina i valori salvati della configurazione nei widget.

        Args:
            debug: stato checkbox debug.
            fixed_zoom: valore slider zoom fisso (0-100).
            dynamic_zoom: valore slider zoom dinamico (0-100).
            deadzone: valore slider deadzone (0-100).
            inertia: valore slider inerzia (0-100).
        """
        self.debug_cb.setChecked(debug)
        self.fixed_zoom_slider.setValue(fixed_zoom)
        self.dynamic_zoom_slider.setValue(dynamic_zoom)
        self.deadzone_slider.setValue(deadzone)
        self.inertia_slider.setValue(inertia)

    def set_file_button_visible(self, visible: bool) -> None:
        """Mostra/nasconde il pulsante di selezione file."""
        self.file_btn.setVisible(visible)


def control_panel_run(parent: Optional[QWidget] = None) -> ControlPanel:
    """Crea e ritorna un'istanza di ControlPanel."""
    return ControlPanel(parent)
