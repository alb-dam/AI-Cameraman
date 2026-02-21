"""Finestra principale: orchestratore puro tra pannelli UI e Orchestrator.

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

from config.settings import SettingsManager
from app.orchestrator import Orchestrator
from gui.panels import control_panel_run, preview_panel_run, log_panel_run


class UIBridge(QObject):
    """Bridge thread-safe tra thread dell'Orchestrator e il thread Main (GUI)."""
    frame_signal: Signal = Signal(np.ndarray)
    log_signal: Signal = Signal(str)


class MainWindow(QMainWindow):
    """Orchestratore GUI: compone i pannelli e collega segnali all'Orchestrator."""

    def __init__(self, orchestrator: Orchestrator, settings: SettingsManager) -> None:
        super().__init__()
        self.orchestrator: Orchestrator = orchestrator
        self.settings: SettingsManager = settings
        self.bridge = UIBridge()

        self.setWindowTitle("AI-Cameraman")
        self.resize(1000, 600)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        self.control_panel = control_panel_run()
        main_layout.addWidget(self.control_panel, stretch=1)

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
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Avvia Elaborazione OBS")
        self.stop_btn = QPushButton("Ferma Elaborazione OBS")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

    def _connect_signals(self) -> None:
        self._connect_control_panel_signals()
        self._connect_preview_panel_signals()
        self._connect_start_stop_signals()
        self._connect_orchestrator_bridge()
        self._populate_sources()
        self._restore_config_to_ui()

        # Avvia auto-preview generale ma non OBS output al lancio
        self.orchestrator.start_processing()

    def _connect_control_panel_signals(self) -> None:
        cp = self.control_panel
        cp.source_changed.connect(self._on_source_changed)
        cp.file_requested.connect(self._on_file_select)
        cp.debug_toggled.connect(self._on_debug_toggled)
        cp.fixed_zoom_changed.connect(lambda v: self.settings.set("fixed_zoom_percent", float(v)))
        cp.dynamic_zoom_changed.connect(lambda v: self.settings.set("dynamic_zoom_percent", float(v)))
        cp.kalman_changed.connect(lambda v: self.settings.set("kalman_preset_percent", float(v)))
        cp.load_roi_requested.connect(self._on_load_roi)
        cp.create_roi_requested.connect(self._on_create_roi)
        cp.save_roi_requested.connect(self._on_save_roi)

    def _connect_preview_panel_signals(self) -> None:
        self.preview_panel.roi_point_added.connect(self._on_roi_point_added)
        self.preview_panel.roi_finalized.connect(self._on_roi_finalized)

    def _connect_start_stop_signals(self) -> None:
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)

    def _connect_orchestrator_bridge(self) -> None:
        self.orchestrator.on_frame_ready = self.bridge.frame_signal.emit
        self.orchestrator.on_log_message = self.bridge.log_signal.emit
        self.bridge.frame_signal.connect(self._on_frame_received)
        self.bridge.log_signal.connect(self.log_panel.append_message)

    def _populate_sources(self) -> None:
        from video.input import VideoInput
        available_cams = VideoInput.get_available_cameras()
        saved_type = self.settings.get("source_type")
        saved_path = self.settings.get("source_path")
        self.control_panel.populate_sources(available_cams, saved_type, saved_path)

    def _restore_config_to_ui(self) -> None:
        self.control_panel.restore_config(
            debug=bool(self.settings.get("debug_mode")),
            fixed_zoom=int(self.settings.get("fixed_zoom_percent")),
            dynamic_zoom=int(self.settings.get("dynamic_zoom_percent")),
            kalman=int(self.settings.get("kalman_preset_percent")),
        )

    def _on_source_changed(self, source_data: tuple) -> None:
        if not source_data:
            return
        source_type, source_path = source_data
        is_file = (source_type == "file")
        self.control_panel.set_file_button_visible(is_file)

        self.settings.set("source_type", source_type)
        if not is_file:
            self.settings.set("source_path", str(source_path))

        self.orchestrator.start_processing()

    def _on_file_select(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona File Video", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if not file_path:
            return
        self.settings.set("source_path", file_path)
        self.log_panel.append_message(f"File video selezionato: {file_path}")
        self.orchestrator.start_processing()

    def _on_debug_toggled(self, checked: bool) -> None:
        self.settings.set("debug_mode", checked)

    def _on_start(self) -> None:
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.orchestrator.start_obs_output()

    def _on_stop(self) -> None:
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.orchestrator.stop_obs_output()

    def _on_load_roi(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona ROI", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        self.orchestrator.roi_manager.load_roi(file_path)
        self.settings.set("last_roi_path", file_path)
        self.log_panel.append_message(f"ROI caricata: {file_path}")

    def _on_create_roi(self) -> None:
        self.orchestrator.roi_manager.start_roi_selection()
        self.preview_panel.set_roi_editing(True)
        self.log_panel.append_message(
            "Modalità editing ROI avviata. Clicca sulla preview per aggiungere punti. "
            "Click destro per terminare."
        )

    def _on_save_roi(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salva ROI", "roi.json", "JSON Files (*.json)"
        )
        if not file_path:
            return
        self.orchestrator.roi_manager.save_roi(file_path)
        self.settings.set("last_roi_path", file_path)
        self.log_panel.append_message(f"ROI salvata: {file_path}")

    def _on_roi_point_added(self, norm_x: float, norm_y: float) -> None:
        self.orchestrator.roi_manager.add_point(norm_x, norm_y)

    def _on_roi_finalized(self) -> None:
        self.orchestrator.roi_manager.finalize_roi()
        self.preview_panel.set_roi_editing(False)
        self.log_panel.append_message("ROI finalizzata.")

    def _on_frame_received(self, frame: np.ndarray) -> None:
        if frame is None:
            return

        if self.orchestrator.roi_manager.editing_mode:
            self.preview_panel.draw_roi_overlay(frame, self.orchestrator.roi_manager.roi_points)

        self.preview_panel.update_frame(frame)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.orchestrator.close_all()
        super().closeEvent(event)


def main_window_run(orchestrator: Orchestrator, settings: SettingsManager) -> None:
    """Crea la QApplication, la finestra MainWindow e avvia il loop Qt."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    window = MainWindow(orchestrator, settings)
    window.show()
    sys.exit(app.exec())
