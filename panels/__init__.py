"""Pannelli UI separati per l'interfaccia grafica."""

from panels.control_panel import ControlPanel, control_panel_run
from panels.preview_panel import PreviewPanel, preview_panel_run
from panels.log_panel import LogPanel, log_panel_run

__all__ = [
    "ControlPanel", "PreviewPanel", "LogPanel",
    "control_panel_run", "preview_panel_run", "log_panel_run"
]
