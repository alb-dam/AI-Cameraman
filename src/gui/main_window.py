"""Finestra principale: orchestratore puro tra pannelli UI e Orchestrator.

Nessuna logica AI, nessuna logica di elaborazione.
Solo segnali, callback e coordinazione.
"""

import sys
import numpy as np

import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QHBoxLayout, QVBoxLayout,
                               QFileDialog, QPushButton)
from PySide6.QtCore import Signal, QObject
from PySide6.QtGui import QCloseEvent, QIcon

from config.settings import SettingsManager
from core.interfaces import IController
from gui.panels import control_panel_run, preview_panel_run, log_panel_run
from video.input import VideoInput


class UIBridge(QObject):
    """Bridge thread-safe tra thread dell'Orchestrator e il thread Main (GUI)."""
    frame_signal: Signal = Signal(np.ndarray)
    log_signal: Signal = Signal(str)


class MainWindow(QMainWindow):
    """Orchestratore GUI: compone i pannelli e collega segnali all'Orchestrator."""

    def __init__(self, controller: IController, settings: SettingsManager) -> None:
        super().__init__()
        self.controller = controller
        self.settings: SettingsManager = settings
        self.bridge = UIBridge()

        from core.paths import get_assets_path
        icon_path = os.path.join(get_assets_path(), "logo.png")

        self.setWindowTitle("AI-Cameraman")
        # Su macOS lasciamo che il Dock usi l'icona nativa .icns del bundle .app
        if sys.platform != "darwin" and os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.resize(640, 621)
        self.setMinimumSize(640, 360)  # 16:9 aspect ratio minimo

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Preview video – occupa tutto lo spazio disponibile
        self.preview_panel = preview_panel_run()
        main_layout.addWidget(self.preview_panel, stretch=1)

        # Pulsanti azione
        self._setup_action_buttons(main_layout)

        # Controlli compatti
        self.control_panel = control_panel_run()
        main_layout.addWidget(self.control_panel)

        # Log in basso
        self.log_panel = log_panel_run()
        main_layout.addWidget(self.log_panel)

    def _setup_action_buttons(self, layout: QVBoxLayout) -> None:
        """Crea i pulsanti Avvia/Ferma Elaborazione sotto la preview."""
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Avvia Elaborazione")
        self.stop_btn = QPushButton("Ferma Elaborazione")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        
        layout.addLayout(btn_layout)

    def _connect_signals(self) -> None:
        self._connect_control_panel_signals()
        self._connect_preview_panel_signals()
        self._connect_action_signals()
        self._connect_orchestrator_bridge()
        self._populate_sources()
        self._restore_config_to_ui()

        # Avvia auto-preview generale ma non NDI output al lancio
        self.controller.start()

    def _connect_control_panel_signals(self) -> None:
        cp = self.control_panel
        cp.source_changed.connect(self._on_source_changed)
        cp.file_requested.connect(self._on_file_select)
        cp.debug_toggled.connect(self._on_debug_toggled)
        cp.fixed_zoom_changed.connect(lambda v: self.settings.set("fixed_zoom_percent", float(v), save_to_disk=False))
        cp.dynamic_zoom_changed.connect(lambda v: self.settings.set("dynamic_zoom_percent", float(v), save_to_disk=False))
        cp.deadzone_changed.connect(lambda v: self.settings.set("director_deadzone_preset_percent", float(v), save_to_disk=False))
        cp.inertia_changed.connect(lambda v: self.settings.set("director_inertia_preset_percent", float(v), save_to_disk=False))
        cp.load_roi_requested.connect(self._on_load_roi)
        cp.create_roi_requested.connect(self._on_create_roi)
        cp.save_roi_requested.connect(self._on_save_roi)
        cp.generate_roi_requested.connect(self._on_generate_roi)
        cp.preview_toggled.connect(self._on_preview_toggled)
        cp.native_toggled.connect(self._on_native_toggled)

    def _connect_preview_panel_signals(self) -> None:
        self.preview_panel.roi_point_added.connect(self._on_roi_point_added)
        self.preview_panel.roi_finalized.connect(self._on_roi_finalized)

    def _connect_action_signals(self) -> None:
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)

    def _connect_orchestrator_bridge(self) -> None:
        self.controller.state.on_frame_ready = self.bridge.frame_signal.emit
        self.controller.state.on_log_message = self.bridge.log_signal.emit
        self.bridge.frame_signal.connect(self._on_frame_received)
        self.bridge.log_signal.connect(self.log_panel.append_message)

    def _populate_sources(self) -> None:
        available_cams = VideoInput.get_available_cameras()
        saved_type = self.settings.get("source_type")
        saved_path = self.settings.get("source_path")
        self.control_panel.populate_sources(available_cams, saved_type, saved_path)

    def _restore_config_to_ui(self) -> None:
        self.control_panel.restore_config(
            debug=bool(self.settings.get("debug_mode")),
            fixed_zoom=int(self.settings.get("fixed_zoom_percent")),
            dynamic_zoom=int(self.settings.get("dynamic_zoom_percent")),
            deadzone=int(self.settings.get("director_deadzone_preset_percent")),
            inertia=int(self.settings.get("director_inertia_preset_percent")),
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

        self.controller.start()

    def _on_file_select(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona File Video", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if not file_path:
            return
        self.settings.set("source_path", file_path)
        self.log_panel.append_message(f"File video selezionato: {file_path}")
        self.controller.start()

    def _on_debug_toggled(self, checked: bool) -> None:
        self.settings.set("debug_mode", checked)
        self.settings.set("enable_performance_monitor", checked)

    def _on_start(self) -> None:
        if self.controller.start_ai_output():
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.log_panel.append_message("Elaborazione NDI AI avviata.")
        else:
            self.log_panel.append_message("Errore: impossibile avviare NDI AI (verificare NDI Runtime).")

    def _on_stop(self) -> None:
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.controller.stop_ai_output()

    def _on_preview_toggled(self, checked: bool) -> None:
        self.controller.state.is_preview_enabled = checked
        status = "attivata" if checked else "disattivata"
        self.log_panel.append_message(f"Preview {status}.")

    def _on_native_toggled(self, checked: bool) -> None:
        if checked:
            self.controller.start_native_output()
            self.log_panel.append_message("NDI Output Nativo attivato.")
        else:
            self.controller.stop_native_output()
            self.log_panel.append_message("NDI Output Nativo disattivato.")

    def _on_load_roi(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona ROI", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        if self.controller.roi_manager.load_roi(file_path):
            self.settings.set("last_roi_path", file_path)
            self.log_panel.append_message(f"ROI caricata: {file_path}")
        else:
            self.log_panel.append_message(f"Errore durante il caricamento della ROI: {file_path}")

    def _on_create_roi(self) -> None:
        self.controller.roi_manager.start_roi_selection()
        self.preview_panel.set_roi_editing(True)
        self.log_panel.append_message(
            "Modalità editing ROI avviata. Clicca sulla preview per aggiungere punti. "
            "Click destro per terminare."
        )

    def _on_save_roi(self) -> None:
        from core.paths import get_tmp_path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salva ROI", os.path.join(get_tmp_path(), "roi.json"), "JSON Files (*.json)"
        )
        if not file_path:
            return
        if self.controller.roi_manager.save_roi(file_path):
            self.settings.set("last_roi_path", file_path)
            self.log_panel.append_message(f"ROI salvata: {file_path}")
        else:
            self.log_panel.append_message(f"Errore durante il salvataggio della ROI: {file_path}")

    def _on_roi_point_added(self, norm_x: float, norm_y: float) -> None:
        self.controller.roi_manager.add_point(norm_x, norm_y)

    def _on_roi_finalized(self) -> None:
        self.controller.roi_manager.finalize_roi()
        self.preview_panel.set_roi_editing(False)
        self.log_panel.append_message("ROI finalizzata.")

    def _on_generate_roi(self) -> None:
        self.log_panel.append_message("Generazione ROI automatica in corso (raccolta frame per 2 minuti)...")
        self.controller.generate_roi()

    def _on_frame_received(self, frame: np.ndarray) -> None:
        if frame is None:
            return

        if self.controller.roi_manager.editing_mode:
            self.preview_panel.draw_roi_overlay(frame, self.controller.roi_manager.roi_points)

        self.preview_panel.update_frame(frame)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.settings.save()
        self.controller.close_all()
        super().closeEvent(event)


def main_window_run(controller: IController, settings: SettingsManager) -> None:
    """Crea la QApplication, la finestra MainWindow e avvia il loop Qt."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    # Applica l'icona anche all'intera app (su macOS cambia l'icona nel Dock se forzata)
    # Evitiamo di farlo su macOS per non sovrascrivere l'icona HQ .icns del bundle .app
    from core.paths import get_assets_path
    icon_path = os.path.join(get_assets_path(), "logo.png")
    if sys.platform != "darwin" and os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        
    window = MainWindow(controller, settings)
    window.show()
    sys.exit(app.exec())
