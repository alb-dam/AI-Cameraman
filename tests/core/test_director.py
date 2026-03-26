import pytest
import numpy as np
from core.director import Director, director_run, VirtualPTZModel


class TestVirtualPTZModel:
    def test_set_config(self):
        ptz = VirtualPTZModel()
        ptz.set_config(50.0, 100.0)
        assert ptz.fixed_zoom == 2.0
        assert ptz.dynamic_intensity == 1.0

    def test_step_no_dynamic(self):
        ptz = VirtualPTZModel()
        ptz.set_config(25.0, 0.0)  # 1.5x zoom, 0 dynamic
        
        # Primo step: inizializzazione
        cx, cy, z = ptz.step(500.0, 300.0, 0.0, 1000.0)
        assert z == 1.5

    def test_step_with_dynamic(self):
        ptz = VirtualPTZModel()
        ptz.set_config(0.0, 100.0)  # 1.0x fixed, full dynamic
        
        # spread=0 → full bonus
        cx, cy, z = ptz.step(500.0, 300.0, 0.0, 1000.0)
        # fixed=1.0 + dynamic bonus > 1.0
        assert z > 1.0


class TestDirector:
    def test_director_run(self):
        d = director_run()
        assert isinstance(d, Director)

    def test_set_config(self):
        d = Director()
        d.set_config(25.0, 50.0)
        assert d.ptz_model.fixed_zoom == 1.5

    def test_process(self, mock_frame_rgb):
        d = Director()
        frame = mock_frame_rgb(1920, 1080)
        
        action_center = (960, 540)
        player_spread = 0.0
        
        instr = d.process(frame, action_center, player_spread)
        
        assert instr.zoom_level == 1.0
        assert instr.smoothed_center == (960, 540)
        assert instr.crop_box == (0, 0, 1920, 1080)
        assert instr.cropped_frame.shape == (1080, 1920, 3)

