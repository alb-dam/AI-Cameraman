import pytest
import numpy as np
from typing import Tuple

from core.models import Detection, TrackedObject

@pytest.fixture
def mock_detection():
    def _create(box: Tuple[int, int, int, int] = (10, 10, 50, 50), 
                center: Tuple[int, int] = (30, 30), 
                conf: float = 0.9):
        return Detection(box=box, center=center, conf=conf)
    return _create

@pytest.fixture
def mock_tracked_object():
    def _create(object_id: int = 0, 
                center: Tuple[int, int] = (30, 30), 
                raw_box: Tuple[int, int, int, int] = (10, 10, 50, 50)):
        return TrackedObject(object_id=object_id, center=center, raw_box=raw_box)
    return _create

@pytest.fixture
def mock_frame_rgb():
    def _create(width: int = 1920, height: int = 1080):
        # Return a black RGB frame
        return np.zeros((height, width, 3), dtype=np.uint8)
    return _create
