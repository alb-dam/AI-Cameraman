import pytest
import numpy as np
from core.geometry import GeometryService

class TestGeometryService:
    
    def test_calculate_crop_region_center(self):
        # 1920x1080 frame, zoom 2x (so we expect 960x540 crop)
        # Center at (960, 540)
        crop = GeometryService.calculate_crop_region(
            center=(960, 540), zoom=2.0, frame_w=1920, frame_h=1080
        )
        x1, y1, x2, y2 = crop
        assert (x2 - x1) == 960
        assert (y2 - y1) == 540
        assert x1 == 960 - 480
        assert y1 == 540 - 270

    def test_calculate_crop_region_edges(self):
        # Target center is top-left corner
        crop = GeometryService.calculate_crop_region(
            center=(0, 0), zoom=2.0, frame_w=1920, frame_h=1080
        )
        x1, y1, x2, y2 = crop
        assert x1 == 0
        assert y1 == 0
        assert x2 == 960
        assert y2 == 540
        
        # Target center is bottom-right corner
        crop_br = GeometryService.calculate_crop_region(
            center=(1920, 1080), zoom=2.0, frame_w=1920, frame_h=1080
        )
        x1, y1, x2, y2 = crop_br
        assert x1 == 1920 - 960
        assert y1 == 1080 - 540
        assert x2 == 1920
        assert y2 == 1080

    def test_apply_polygon_mask_none(self, mock_frame_rgb):
        frame = mock_frame_rgb()
        # Should return same frame unharmed
        masked = GeometryService.apply_polygon_mask(frame, None)
        assert np.array_equal(frame, masked)

    def test_apply_polygon_mask_valid(self, mock_frame_rgb):
        frame = mock_frame_rgb(100, 100)
        # make it fully white
        frame.fill(255)
        
        # A triangle polygon spanning left half (norm coords)
        polygon = np.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 1.0]
        ], dtype=np.float32)

        masked = GeometryService.apply_polygon_mask(frame, polygon)
        
        # Top-left corner should be white (inside triangle)
        assert np.array_equal(masked[0, 0], [255, 255, 255])
        # Top-right corner should be black (outside)
        assert np.array_equal(masked[0, 99], [0, 0, 0])

    def test_draw_polygon(self, mock_frame_rgb):
        frame = mock_frame_rgb(100, 100)
        polygon = np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]])
        
        GeometryService.draw_polygon(frame, polygon, color=(0, 255, 0), thickness=2)
        
        # Assert pixels were drawn (not all zero anymore)
        assert np.any(frame[:, :, 1] == 255) # Green channel has some 255s
