"""Modulo di tracking: calcolo centro d'azione.

Contiene ActionCenterCalculator, estratto da core/director.py per eliminare
la dipendenza circolare tra vision.py e director.py.
"""

from typing import List, Optional, Tuple

from core.models import TrackedObject


class ActionCenterCalculator:
    """Calcola il centro d'azione come media ponderata sui giocatori.

    Il pallone è disabilitato per design (BALL_WEIGHT=0.0): non contribuisce
    al baricentro per evitare salti bruschi di inquadratura causati da
    rilevamenti di palla instabili (allucinazioni YOLO o occlusioni).
    Potrà essere riabilitato in futuro aumentando BALL_WEIGHT.
    """

    PLAYER_WEIGHT = 1.0

    @classmethod
    def compute_center(
        cls,
        players: List[TrackedObject],
        ball: Optional[TrackedObject],
    ) -> Optional[Tuple[int, int]]:
        """Ritorna il baricentro ponderato dei giocatori. Ritorna None se vuoto."""
        if not players:
            return None

        total_w = 0.0
        wx, wy = 0.0, 0.0

        for p in players:
            cx, cy = p.center
            wx += cx * cls.PLAYER_WEIGHT
            wy += cy * cls.PLAYER_WEIGHT
            total_w += cls.PLAYER_WEIGHT

        if total_w > 0:
            return (int(wx / total_w), int(wy / total_w))
        return None
