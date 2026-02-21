"""Finestra principale: orchestratore puro tra pannelli UI e Backend.

Nessuna logica AI, nessuna logica di elaborazione.
Solo segnali, callback e coordinazione.
"""

import sys

import numpy as np

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QHBoxLayout, QVBoxLayout, QGroupBox,
                               QFileDialog, QPushButton)
from PySide6.QtCore import Signal, QObject
from PySide6.QtGui import QCloseEvent

from config import ConfigManager
from backend import Backend
from panels import control_panel_run, preview_panel_run, log_panel_run


class UIBridge(QObject):
    """Bridge thread-safe tra thread del Backend e il thread Main (GUI)."""
    frame_signal: Signal = Signal(np.ndarray)
    log_signal: Signal = Signal(str)


class MainWindow(QMainWindow):
    """Orchestratore GUI: compone i pannelli e collega segnali al Backend."""

    # ── Inizializzazione ────────────────────────────────────────────────

    def __init__(self, backend: Backend, config: ConfigManager) -> None:
        """Inizializza la finestra principale con backend e configurazione."""
        super().__init__()
        self.backend: Backend = backend
        self.config: ConfigManager = config
        self.bridge = UIBridge()

        self.setWindowTitle("Video Stream App")
        self.resize(1000, 600)

        self._setup_ui()
        self._connect_signals()

    # ── Setup UI ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        """Compone il layout a due colonne con i pannelli separati."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Colonna sinistra: controlli
        self.control_panel = control_panel_run()
        main_layout.addWidget(self.control_panel, stretch=1)

        # Colonna destra: preview + start/stop + log
        right_group = QGroupBox("Preview e Log")
        right_layout = QVBoxLayout()
        right_group.setLayout(right_layout)
        main_layout.addWidget(right_group, stretch=3)

        self.preview_panel = preview_panel_run()
        right_layout.addWidget(self.preview_panel)

        self._setup_start_stop_buttons(right_layout)

        self.log_panel = log_panel_run()
        right_layout.addWidget(self.log_panel)

    def _setup_start_stop_buttons(self, layout: QVBoxLayout) -> None:
        """Crea i pulsanti Avvia/Ferma sotto la preview."""
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Avvia Elaborazione")
        self.stop_btn = QPushButton("Ferma Elaborazione")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

    # ── Connessione segnali ─────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """Collega tutti i segnali dei pannelli ai callback di coordinazione."""
        self._connect_control_panel_signals()
        self._connect_preview_panel_signals()
        self._connect_start_stop_signals()
        self._connect_backend_bridge()
        self._populate_sources()
        self._restore_config_to_ui()

        # Avvia auto-preview al lancio applicazione
        self.backend.start_processing()

    def _connect_control_panel_signals(self) -> None:
        """Collega i segnali del ControlPanel ai callback."""
        cp = self.control_panel
        cp.source_changed.connect(self._on_source_changed)
        cp.file_requested.connect(self._on_file_select)
        cp.debug_toggled.connect(self._on_debug_toggled)
        cp.fixed_zoom_changed.connect(lambda v: self.config.set("fixed_zoom_percent", v))
        cp.dynamic_zoom_changed.connect(lambda v: self.config.set("dynamic_zoom_percent", v))
        cp.kalman_changed.connect(lambda v: self.config.set("kalman_preset_percent", v))
        cp.load_roi_requested.connect(self._on_load_roi)
        cp.create_roi_requested.connect(self._on_create_roi)
        cp.save_roi_requested.connect(self._on_save_roi)

    def _connect_preview_panel_signals(self) -> None:
        """Collega i segnali del PreviewPanel ai callback ROI."""
        self.preview_panel.roi_point_added.connect(self._on_roi_point_added)
        self.preview_panel.roi_finalized.connect(self._on_roi_finalized)

    def _connect_start_stop_signals(self) -> None:
        """Collega i pulsanti start/stop."""
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)

    def _connect_backend_bridge(self) -> None:
        """Configura il bridge thread-safe tra Backend e pannelli UI."""
        self.backend.on_frame_ready = self.bridge.frame_signal.emit
        self.backend.on_log_message = self.bridge.log_signal.emit
        self.bridge.frame_signal.connect(self._on_frame_received)
        self.bridge.log_signal.connect(self.log_panel.append_message)

    # ── Popolamento e ripristino configurazione ─────────────────────────

    def _populate_sources(self) -> None:
        """Trova le webcam disponibili e popola il ControlPanel."""
        from input import VideoInput
        available_cams = VideoInput.get_available_cameras()
        saved_type = self.config.get("source_type")
        saved_path = self.config.get("source_path")
        self.control_panel.populate_sources(available_cams, saved_type, saved_path)

    def _restore_config_to_ui(self) -> None:
        """Ripristina i valori salvati in configurazione nei pannelli."""
        self.control_panel.restore_config(
            debug=self.config.get("debug_mode"),
            fixed_zoom=self.config.get("fixed_zoom_percent"),
            dynamic_zoom=self.config.get("dynamic_zoom_percent"),
            kalman=self.config.get("kalman_preset_percent"),
        )

    # ── Callback: sorgente video ────────────────────────────────────────

    def _on_source_changed(self, source_data: tuple) -> None:
        """Gestisce il cambio sorgente video dal menu a tendina."""
        if not source_data:
            return

        source_type, source_path = source_data
        is_file = (source_type == "file")
        self.control_panel.set_file_button_visible(is_file)

        self.config.set("source_type", source_type)
        if not is_file:
            self.config.set("source_path", str(source_path))

        self.backend.start_processing()

    def _on_file_select(self) -> None:
        """Apre il dialogo per selezionare un file video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona File Video", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if not file_path:
            return
        self.config.set("source_path", file_path)
        self.log_panel.append_message(f"File video selezionato: {file_path}")
        self.backend.start_processing()

    def _on_debug_toggled(self, checked: bool) -> None:
        """Attiva/disattiva la modalità debug."""
        self.config.set("debug_mode", checked)

    # ── Callback: start/stop ────────────────────────────────────────────

    def _on_start(self) -> None:
        """Avvia l'invio a OBS."""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.backend.start_obs_output()

    def _on_stop(self) -> None:
        """Ferma l'invio a OBS."""
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.backend.stop_obs_output()

    # ── Callback: ROI ───────────────────────────────────────────────────

    def _on_load_roi(self) -> None:
        """Carica una ROI da file JSON."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona ROI", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        self.backend.load_roi(file_path)
        self.config.set("last_roi_path", file_path)
        self.log_panel.append_message(f"ROI caricata: {file_path}")

    def _on_create_roi(self) -> None:
        """Avvia la modalità editing ROI."""
        self.backend.start_roi_selection()
        self.preview_panel.set_roi_editing(True)
        self.log_panel.append_message(
            "Modalità editing ROI avviata. Clicca sulla preview per aggiungere punti. "
            "Click destro per terminare."
        )

    def _on_save_roi(self) -> None:
        """Salva la ROI corrente su file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salva ROI", "roi.json", "JSON Files (*.json)"
        )
        if not file_path:
            return
        self.backend.save_roi(file_path)
        self.config.set("last_roi_path", file_path)
        self.log_panel.append_message(f"ROI salvata: {file_path}")

    def _on_roi_point_added(self, norm_x: float, norm_y: float) -> None:
        """Aggiunge un punto alla ROI in editing."""
        self.backend.add_roi_point(norm_x, norm_y)

    def _on_roi_finalized(self) -> None:
        """Finalizza la ROI corrente e disattiva l'editing."""
        self.backend.finalize_roi()
        self.preview_panel.set_roi_editing(False)
        self.log_panel.append_message("ROI finalizzata.")

    # ── Callback: frame dal backend ─────────────────────────────────────

    def _on_frame_received(self, frame: np.ndarray) -> None:
        """Riceve un frame dal backend, applica overlay ROI e aggiorna preview."""
        if frame is None:
            return

        if self.backend.is_roi_editing:
            self.preview_panel.draw_roi_overlay(frame, self.backend.roi_points)

        self.preview_panel.update_frame(frame)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def closeEvent(self, event: QCloseEvent) -> None:
        """Assicura l'arresto del backend alla chiusura dell'interfaccia."""
        self.backend.stop_processing()
        super().closeEvent(event)


def frontend_run(backend: Backend, config: ConfigManager) -> None:
    """Crea la QApplication, la finestra MainWindow e avvia il loop Qt."""
    app = QApplication(sys.argv)
    window = MainWindow(backend, config)
    window.show()
    sys.exit(app.exec())
