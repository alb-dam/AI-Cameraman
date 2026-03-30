"""Pannello log testuale."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit


class LogPanel(QWidget):
    """Pannello con log testuale read-only e autoscroll."""

    # ── Inizializzazione ────────────────────────────────────────────────

    def __init__(self, parent: QWidget = None) -> None:
        """Crea il widget log con QTextEdit read-only."""
        super().__init__(parent)
        self._setup_ui()

    # ── Setup UI ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        """Inizializza il layout con il QTextEdit."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        layout.addWidget(self.log_text)

    # ── API pubblica ────────────────────────────────────────────────────

    def append_message(self, message: str) -> None:
        """Aggiunge un messaggio al log e scorre fino in fondo.

        Args:
            message: testo da aggiungere al log.
        """
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def log_panel_run(parent: QWidget = None) -> LogPanel:
    """Crea e ritorna un'istanza di LogPanel."""
    return LogPanel(parent)
