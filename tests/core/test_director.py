import pytest
from core.director import Director, director_run, ValueSmoother, PointSmoother, CameraStrategy

class TestValueSmoother:
    def test_smooth(self):
        s = ValueSmoother(smoothing_factor=0.5, initial_value=1.0)
        v1 = s.smooth(2.0)
        assert v1 == 1.5
        v2 = s.smooth(2.0)
        assert v2 == 1.75

class TestPointSmoother:
    def test_smooth_initial(self):
        s = PointSmoother(smoothing_factor=0.5)
        pt = s.smooth((100.0, 100.0))
        assert pt == (100.0, 100.0)

    def test_smooth_subsequent(self):
        s = PointSmoother(smoothing_factor=0.5)
        s.smooth((100.0, 100.0))
        pt2 = s.smooth((200.0, 200.0))
        assert pt2 == (150.0, 150.0)

class TestCameraStrategy:
    def test_set_config(self):
        cs = CameraStrategy()
        # 50% fixed zoom -> 2.0x
        # 100% dynamic -> 1.0 intensity
        cs.set_config(50.0, 100.0)
        assert cs.fixed_zoom == 2.0
        assert cs.dynamic_intensity == 1.0

    def test_compute_target_zoom_no_dynamic(self):
        cs = CameraStrategy()
        cs.set_config(25.0, 0.0) # 1.5x zoom, 0 dynamic
        
        z = cs.compute_target_zoom(500.0)
        assert z == 1.5

    def test_compute_target_zoom_dynamic(self):
        cs = CameraStrategy()
        cs.set_config(0.0, 100.0) # 1.0x fixed, full dynamic
        
        # Max spread 1000.0, dynamic scale 1.0
        # spread 0 -> full bonus: 1.0 + (1.0 * 1.0 * 1.0) = 2.0
        z_tight = cs.compute_target_zoom(0.0)
        assert z_tight == 2.0

        # spread 1000 -> no bonus = 1.0
        z_wide = cs.compute_target_zoom(1000.0)
        assert z_wide == 1.0

class TestDirector:
    def test_director_run(self):
        d = director_run()
        assert isinstance(d, Director)

    def test_set_config(self):
        d = Director()
        d.set_config(25.0, 50.0)
        assert d.camera_strategy.fixed_zoom == 1.5

    def test_process(self, mock_frame_rgb):
        d = Director()
        # Center in a 1920x1080 frame
        frame = mock_frame_rgb(1920, 1080)
        
        action_center = (960, 540)
        player_spread = 800.0
        
        instr = d.process(frame, action_center, player_spread)
        
        # Initial zoom smoother is 1.0, and smoothing factor is 0.1
        # Target zoom for default config (fixed=1.0, dyn=0.0) is 1.0
        # Smoothed zoom is 1.0
        assert instr.zoom_level == 1.0
        assert instr.smoothed_center == (960, 540)
        
        # Crop box for 1.0 zoom should be full frame
        assert instr.crop_box == (0, 0, 1920, 1080)
        assert instr.cropped_frame.shape == (1080, 1920, 3)
