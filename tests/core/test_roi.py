import numpy as np
from core.roi import ROIManager

class TestROIManager:
    def test_is_valid_initial(self):
        m = ROIManager()
        assert not m.is_valid
        assert m.roi.polygon is None

    def test_create_from_points_valid(self):
        m = ROIManager()
        points = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
        m.create_from_points(points)
        assert m.is_valid
        assert np.array_equal(m.roi.polygon, np.array(points, dtype=np.float32))

    def test_create_from_points_invalid(self):
        m = ROIManager()
        points = [(0.1, 0.1), (0.9, 0.1)] # Only 2 points
        m.create_from_points(points)
        assert not m.is_valid
        assert m.roi.polygon is None

    def test_apply_mask_invalid(self, mock_frame_rgb):
        m = ROIManager()
        frame = mock_frame_rgb()
        masked = m.apply_roi(frame)
        assert np.array_equal(masked, frame)

    def test_apply_mask_valid(self, mock_frame_rgb):
        m = ROIManager()
        frame = mock_frame_rgb(100, 100)
        frame.fill(255)
        
        m.create_from_points([(0,0), (0.5, 0), (0, 1)])
        masked = m.apply_roi(frame)
        assert m.is_valid
        assert np.array_equal(masked[0, 0], [255, 255, 255])
        assert np.array_equal(masked[0, 99], [0, 0, 0])

    def test_draw_roi(self, mock_frame_rgb):
        m = ROIManager()
        frame = mock_frame_rgb(100, 100)
        
        m.draw_roi(frame)
        assert np.all(frame == 0) # No drawing if invalid
        
        m.create_from_points([(0,0), (0.5, 0), (0, 1)])
        m.draw_roi(frame)
        assert np.any(frame != 0) # It drew something

    def test_editing_mode(self):
        m = ROIManager()
        m.start_roi_selection()
        assert m.editing_mode
        assert len(m.roi_points) == 0
        
        m.add_point(0.1, 0.1)
        m.add_point(0.9, 0.1)
        m.add_point(0.5, 0.9)
        assert len(m.roi_points) == 3
        
        m.finalize_roi()
        assert not m.editing_mode
        assert m.is_valid
