"""Pannello di preview video con supporto editing ROI."""

from typing import List, Tuple

import cv2
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, QObject, QEvent, QPointF
from PySide6.QtGui import QImage, QPixmap


class PreviewPanel(QWidget):
    """Preview video con aspect ratio fisso e intercettazione click per ROI.

    Emette segnali puri per l'editing ROI, non conosce Backend.
    """

    # ── Segnali ─────────────────────────────────────────────────────────

    roi_point_added = Signal(float, float)   # (norm_x, norm_y)
    roi_finalized = Signal()

    # ── Inizializzazione ────────────────────────────────────────────────

    def __init__(self, parent: QWidget = None) -> None:
        """Crea la label preview con aspect ratio fisso."""
        super().__init__(parent)

        self._last_frame_size: Tuple[int, int] = (1920, 1080)
        self._roi_editing: bool = False

        self._setup_ui()

    # ── Setup UI ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        """Inizializza la label preview con sfondo nero."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel("Nessun segnale video in corso")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: black; color: white;")
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.installEventFilter(self)
        layout.addWidget(self.preview_label)

    # ── API pubblica ────────────────────────────────────────────────────

    def update_frame(self, frame: np.ndarray) -> None:
        """Converte un frame BGR in QPixmap e aggiorna la preview.

        Args:
            frame: frame BGR da OpenCV (numpy array HxWx3).
        """
        if frame is None:
            return

        h, w = frame.shape[:2]
        self._last_frame_size = (w, h)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    def draw_roi_overlay(self, frame: np.ndarray,
                         roi_points: List[Tuple[float, float]]) -> None:
        """Disegna punti e linee dell'editing ROI direttamente sul frame.

        Deve essere chiamato PRIMA di update_frame.

        Args:
            frame: frame BGR su cui disegnare (modificato in-place).
            roi_points: lista di punti normalizzati (0-1).
        """
        h, w = frame.shape[:2]
        for i, p in enumerate(roi_points):
            cx, cy = int(p[0] * w), int(p[1] * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            if i > 0:
                px, py = int(roi_points[i - 1][0] * w), int(roi_points[i - 1][1] * h)
                cv2.line(frame, (px, py), (cx, cy), (0, 255, 0), 2)

    def set_roi_editing(self, editing: bool) -> None:
        """Abilita/disabilita l'intercettazione dei click per editing ROI.

        Args:
            editing: True per abilitare, False per disabilitare.
        """
        self._roi_editing = editing

    @property
    def last_frame_size(self) -> Tuple[int, int]:
        """Restituisce le dimensioni (w, h) dell'ultimo frame ricevuto."""
        return self._last_frame_size

    # ── Event filter per ROI editing ────────────────────────────────────

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        """Intercetta click sulla preview per editing ROI."""
        if source != self.preview_label or not self._roi_editing:
            return super().eventFilter(source, event)

        if event.type() != QEvent.MouseButtonPress:
            return super().eventFilter(source, event)

        if event.button() == Qt.LeftButton:
            self._handle_roi_click(event.position())
        elif event.button() == Qt.RightButton:
            self.roi_finalized.emit()
        return True

    def _handle_roi_click(self, pos: QPointF) -> None:
        """Converte il click in coordinate normalizzate ed emette il segnale."""
        w, h = self._last_frame_size
        lw, lh = self.preview_label.width(), self.preview_label.height()

        scale = min(lw / w, lh / h)
        pw, ph = w * scale, h * scale
        x_offset = (lw - pw) / 2
        y_offset = (lh - ph) / 2

        click_x, click_y = pos.x(), pos.y()

        if x_offset <= click_x <= x_offset + pw and y_offset <= click_y <= y_offset + ph:
            norm_x = (click_x - x_offset) / pw
            norm_y = (click_y - y_offset) / ph
            self.roi_point_added.emit(norm_x, norm_y)


def preview_panel_run(parent: QWidget = None) -> PreviewPanel:
    """Crea e ritorna un'istanza di PreviewPanel."""
    return PreviewPanel(parent)
