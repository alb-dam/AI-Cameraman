"""Test unitari per ROIManager.filter_detections_by_feet()."""

import pytest
import numpy as np

from core.roi import ROIManager
from core.models import Detection


def make_detection(x1: int, y1: int, x2: int, y2: int, conf: float = 0.9) -> Detection:
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return Detection(box=(x1, y1, x2, y2), center=(cx, cy), conf=conf)


class TestFilterByFeet:
    """Test per il filtraggio delle detection basato sulla posizione dei piedi."""

    @pytest.fixture
    def roi_manager_with_rect(self) -> ROIManager:
        """Crea un ROIManager con un rettangolo ROI al centro (0.2-0.8 x 0.2-0.8)."""
        rm = ROIManager()
        points = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.2, 0.2)]
        rm.create_from_points(points)
        return rm

    def test_feet_inside_roi_included(self, roi_manager_with_rect: ROIManager):
        """Giocatore con piedi dentro la ROI → incluso."""
        # Frame 1000x1000, ROI da (200,200) a (800,800)
        # Bbox con piedi (bottom-center) a (500, 600) → dentro la ROI
        det = make_detection(450, 400, 550, 600)
        result = roi_manager_with_rect.filter_detections_by_feet([det], 1000, 1000)
        assert len(result) == 1

    def test_feet_outside_roi_excluded(self, roi_manager_with_rect: ROIManager):
        """Giocatore con piedi fuori dalla ROI → escluso."""
        # Bbox con piedi (bottom-center) a (500, 900) → fuori dalla ROI
        det = make_detection(450, 700, 550, 900)
        result = roi_manager_with_rect.filter_detections_by_feet([det], 1000, 1000)
        assert len(result) == 0

    def test_feet_inside_torso_outside_included(self, roi_manager_with_rect: ROIManager):
        """Giocatore con piedi dentro ma busto fuori dalla ROI → incluso."""
        # Piedi a (500, 250) dentro ROI, testa a (500, 100) fuori ROI
        det = make_detection(450, 100, 550, 250)
        result = roi_manager_with_rect.filter_detections_by_feet([det], 1000, 1000)
        assert len(result) == 1

    def test_feet_outside_torso_inside_excluded(self, roi_manager_with_rect: ROIManager):
        """Giocatore con piedi fuori ma busto dentro la ROI → escluso."""
        # Busto a (500, 600) dentro ROI, piedi a (500, 850) fuori ROI
        det = make_detection(450, 600, 550, 850)
        result = roi_manager_with_rect.filter_detections_by_feet([det], 1000, 1000)
        assert len(result) == 0

    def test_invalid_roi_passthrough(self):
        """ROI non valida → tutte le detection passano."""
        rm = ROIManager()  # Nessun poligono
        det = make_detection(100, 100, 200, 200)
        result = rm.filter_detections_by_feet([det], 1000, 1000)
        assert len(result) == 1

    def test_editing_mode_passthrough(self, roi_manager_with_rect: ROIManager):
        """In modalità editing → tutte le detection passano."""
        roi_manager_with_rect.editing_mode = True
        det = make_detection(450, 700, 550, 900)  # Piedi fuori ROI
        result = roi_manager_with_rect.filter_detections_by_feet([det], 1000, 1000)
        assert len(result) == 1

    def test_mixed_detections(self, roi_manager_with_rect: ROIManager):
        """Mix di detection dentro e fuori → filtra correttamente."""
        det_inside = make_detection(450, 400, 550, 600)   # piedi (500,600) → dentro
        det_outside = make_detection(450, 700, 550, 900)  # piedi (500,900) → fuori
        result = roi_manager_with_rect.filter_detections_by_feet(
            [det_inside, det_outside], 1000, 1000
        )
        assert len(result) == 1

    def test_empty_list(self, roi_manager_with_rect: ROIManager):
        """Lista vuota → lista vuota."""
        result = roi_manager_with_rect.filter_detections_by_feet([], 1000, 1000)
        assert len(result) == 0
