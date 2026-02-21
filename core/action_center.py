"""Calcolo centro d'azione basato su rilevamenti."""

from typing import List, Dict, Any, Tuple, Optional


class ActionCenterCalculator:
    """Calcola il centro d'azione come media ponderata giocatori + pallone."""

    PLAYER_WEIGHT = 1.0
    BALL_WEIGHT = 3.0

    @classmethod
    def compute_center(
        cls,
        players: List[Dict[str, Any]],
        ball: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[int, int]]:
        """Ritorna il baricentro ponderato dell'azione sul campo. Ritorna None se vuoto."""
        if not players and not ball:
            return None

        total_w = 0.0
        wx, wy = 0.0, 0.0

        for p in players:
            cx, cy = p["center"]
            wx += cx * cls.PLAYER_WEIGHT
            wy += cy * cls.PLAYER_WEIGHT
            total_w += cls.PLAYER_WEIGHT

        if ball:
            bx, by = ball["center"]
            wx += bx * cls.BALL_WEIGHT
            wy += by * cls.BALL_WEIGHT
            total_w += cls.BALL_WEIGHT

        if total_w > 0:
            return (int(wx / total_w), int(wy / total_w))
        return None
