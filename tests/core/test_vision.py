import pytest
import numpy as np
from core.vision import SimpleKalman, KalmanTracker

class TestSimpleKalman:
    def test_initialization(self):
        k = SimpleKalman(10.0, 20.0)
        assert k.x[0, 0] == 10.0
        assert k.x[1, 0] == 20.0
        assert k.time_since_update == 0

    def test_predict(self):
        k = SimpleKalman(100.0, 100.0)
        # Provide velocity via state matrix directly to test predict moves it
        k.x[2, 0] = 30.0 # vx
        k.x[3, 0] = 0.0  # vy
        
        px, py = k.predict()
        assert px > 100.0  # Should have moved in +x direction
        assert py == 100.0
        assert k.time_since_update == 1

    def test_update(self):
        k = SimpleKalman(100.0, 100.0)
        k.predict()
        ux, uy = k.update(110.0, 105.0)
        
        assert k.time_since_update == 0
        assert 100.0 < ux < 110.0  # Filter averages between model and measurement
        assert 100.0 < uy < 105.0

    def test_set_config(self):
        k = SimpleKalman(0, 0)
        k.set_config(5.0, 3.0)
        assert np.all(k.Q == np.eye(4) * 5.0)
        assert np.all(k.R == np.eye(2) * 3.0)

class TestKalmanTracker:
    def test_tracker_initialization(self):
        t = KalmanTracker(q_std=10.0, r_std=15.0)
        assert t.q_std == 10.0
        assert t.r_std == 15.0
        assert t.ball_kalman is None
        assert len(t.players_kalman) == 0

    def test_set_config(self):
        t = KalmanTracker()
        t.set_config(5.0, 5.0)
        assert t.q_std == 5.0

    def test_update_ball_new(self, mock_detection):
        t = KalmanTracker()
        raw_ball = mock_detection(center=(500, 500))
        players, ball = t.update([], raw_ball)
        
        assert ball is not None
        assert ball.object_id == -1
        assert ball.center == (500, 500)
        assert t.ball_kalman is not None

    def test_update_ball_lost(self, mock_detection):
        t = KalmanTracker()
        raw_ball = mock_detection(center=(500, 500))
        t.update([], raw_ball) # init
        
        # Set velocity to see if prediction carries it
        t.ball_kalman.x[2, 0] = 30 # vx
        
        players, ball = t.update([], None) # predict only for ball
        assert ball is not None
        assert ball.center[0] > 500 # Should have moved based on prediction

    def test_update_players_new_and_match(self, mock_detection):
        t = KalmanTracker()
        
        # 1. New player
        p1 = mock_detection(center=(100, 100))
        players, _ = t.update([p1], None)
        assert len(players) == 1
        assert players[0].object_id == 0
        
        # 2. Match player (moves slightly)
        p1_moved = mock_detection(center=(105, 105))
        players2, _ = t.update([p1_moved], None)
        assert len(players2) == 1
        assert players2[0].object_id == 0 # Should match same ID
        
        # 3. Add second player
        p2 = mock_detection(center=(500, 500))
        players3, _ = t.update([p1_moved, p2], None)
        assert len(players3) == 2
        assert {p.object_id for p in players3} == {0, 1}

    def test_update_predict_only(self, mock_detection):
        t = KalmanTracker()
        t.update([mock_detection(center=(100, 100))], mock_detection(center=(500, 500)))
        
        pred_players, pred_ball = t.update([], None, predict_only=True)
        assert len(pred_players) == 1
        assert pred_ball is not None
