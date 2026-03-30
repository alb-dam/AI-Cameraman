"""Test unitari per ActionCenterCalculator (core/tracking.py)."""

from core.tracking import ActionCenterCalculator
from core.models import TrackedObject


def make_tracked(object_id: int, cx: int, cy: int) -> TrackedObject:
    return TrackedObject(
        object_id=object_id,
        center=(cx, cy),
        raw_box=(cx - 10, cy - 20, cx + 10, cy + 20),
    )


class TestActionCenterCalculator:
    def test_single_player_center(self):
        p = make_tracked(0, 100, 200)
        result = ActionCenterCalculator.compute_center([p], None)
        assert result == (100, 200)

    def test_two_players_equal_weight(self):
        p1 = make_tracked(0, 0, 0)
        p2 = make_tracked(1, 100, 100)
        result = ActionCenterCalculator.compute_center([p1, p2], None)
        assert result == (50, 50)

    def test_no_players_returns_none(self):
        result = ActionCenterCalculator.compute_center([], None)
        assert result is None

    def test_ball_does_not_affect_center(self):
        """Il pallone non contribuisce al centro (BALL_WEIGHT=0.0 disabilitato)."""
        p = make_tracked(0, 100, 100)
        ball = make_tracked(-1, 500, 500)  # lontanissimo
        result_with_ball = ActionCenterCalculator.compute_center([p], ball)
        result_no_ball = ActionCenterCalculator.compute_center([p], None)
        # Il pallone non deve spostare il centro
        assert result_with_ball == result_no_ball

    def test_three_players_centroid(self):
        players = [
            make_tracked(0, 0, 0),
            make_tracked(1, 300, 0),
            make_tracked(2, 150, 300),
        ]
        result = ActionCenterCalculator.compute_center(players, None)
        # Baricentro: x=150, y=100
        assert result == (150, 100)
