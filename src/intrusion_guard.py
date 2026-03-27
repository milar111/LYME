"""
intrusion_guard.py
──────────────────
2-second dwell-timer state machine.

Rules:
  • A person must have their BODY CENTRE inside the forbidden zone
    continuously for >= INTRUSION_DWELL_SECONDS before an intrusion
    is declared.
  • The moment the body centre leaves the zone the timer resets.
  • This cleanly filters out a hand or arm briefly crossing the line,
    because body_centre_in_zone() uses the torso midpoint (hips + shoulders).
  • Once an intrusion is declared the guard stays in INTRUDING state
    (so the caller can keep the visual alert on) until the person leaves.
"""

import time
from enum import Enum, auto
from src import config


class GuardState(Enum):
    CLEAR     = auto()   # Nobody in zone
    DWELLING  = auto()   # Body in zone but not long enough yet
    INTRUDING = auto()   # Confirmed intrusion


class IntrusionGuard:
    def __init__(self):
        self._state: GuardState = GuardState.CLEAR
        self._dwell_start: float = 0.0
        self.dwell_seconds: float = getattr(config, 'INTRUSION_DWELL_SECONDS', 2.0)

    # ── Public API ────────────────────────────────────────────────────────────

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

        # Body is in zone
        if self._state == GuardState.CLEAR:
            self._state = GuardState.DWELLING
            self._dwell_start = now

        if self._state == GuardState.DWELLING:
            elapsed = now - self._dwell_start
            progress = min(elapsed / self.dwell_seconds, 1.0)
            if elapsed >= self.dwell_seconds:
                self._state = GuardState.INTRUDING
            return self._state, progress

        # INTRUDING — stay intruding until body leaves
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