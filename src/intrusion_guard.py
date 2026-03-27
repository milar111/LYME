import time
from enum import Enum, auto
from src import config


class GuardState(Enum):
    CLEAR     = auto()
    DWELLING  = auto()
    INTRUDING = auto()


class IntrusionGuard:
    def __init__(self):
        self._state: GuardState = GuardState.CLEAR
        self._dwell_start: float = 0.0
        self.dwell_seconds: float = getattr(config, 'INTRUSION_DWELL_SECONDS', 2.0)


    def update(self, body_in_zone: bool) -> tuple[GuardState, float]:
        """
        Call once per frame with whether the body centre is currently in zone.

        Returns:
            (GuardState, dwell_progress_0_to_1)
            dwell_progress is 0.0 when CLEAR, rises to 1.0 when INTRUDING.
        """
        now = time.monotonic()

        if not body_in_zone:
            self._state = GuardState.CLEAR
            self._dwell_start = 0.0
            return self._state, 0.0

        if self._state == GuardState.CLEAR:
            self._state = GuardState.DWELLING
            self._dwell_start = now

        if self._state == GuardState.DWELLING:
            elapsed = now - self._dwell_start
            progress = min(elapsed / self.dwell_seconds, 1.0)
            if elapsed >= self.dwell_seconds:
                self._state = GuardState.INTRUDING
            return self._state, progress

        return GuardState.INTRUDING, 1.0

    @property
    def state(self) -> GuardState:
        return self._state

    def is_intruding(self) -> bool:
        return self._state == GuardState.INTRUDING

    def just_triggered(self, previous_state: GuardState) -> bool:
        """True only on the exact frame the state first becomes INTRUDING."""
        return (previous_state != GuardState.INTRUDING
                and self._state == GuardState.INTRUDING)