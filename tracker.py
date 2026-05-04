"""
Vehicle exit tracker — state machine.

IDLE  →  min_frames detections  →  TRACKING  →  exit_timeout silence  →  log  →  COOLDOWN  →  IDLE

On commit:
  - Always logs to "All Detections" + "Should Install" (if sticker_count < 3)
  - If unknown_sticker was seen, also logs to "Unknown Stickers" with screenshot
"""

import time
from copy import copy
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from detector import DetectionResult
    from logger import ExcelLogger


@dataclass
class VehicleTracker:
    exit_timeout: float = 3.0   # seconds of silence → vehicle has left
    min_frames:   int   = 2     # frames needed to confirm presence (set 1 for fast cars)
    cooldown:     float = 4.0   # post-log lockout to avoid double-logging same vehicle

    _state:          str   = field(default="IDLE", init=False)
    _hit_count:      int   = field(default=0,      init=False)
    _last_hit:       float = field(default=0.0,    init=False)
    _first_seen:     float = field(default=0.0,    init=False)
    _cooldown_until: float = field(default=0.0,    init=False)
    _best:           Optional[object] = field(default=None, init=False)
    _best_frame:     Optional[np.ndarray] = field(default=None, repr=False, init=False)

    def update(
        self,
        result: "DetectionResult",
        logger: "ExcelLogger",
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Call once per processed frame.
        Returns True when a vehicle exit is committed to Excel.
        """
        now = time.monotonic()

        if now < self._cooldown_until:
            return False

        if result.any_detected:
            if self._hit_count == 0:
                self._first_seen = now
            self._last_hit  = now
            self._hit_count += 1
            self._accumulate(result, frame)
            if self._hit_count >= self.min_frames:
                self._state = "TRACKING"
            return False

        # No detection this frame
        if self._state == "TRACKING":
            if (now - self._last_hit) >= self.exit_timeout:
                self._commit(logger)
                self._cooldown_until = now + self.cooldown
                self._reset()
                return True

        elif self._state == "IDLE" and self._hit_count > 0:
            # Flickered but never confirmed — discard
            if (now - self._last_hit) > self.exit_timeout:
                self._reset()

        return False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _accumulate(self, result: "DetectionResult", frame: Optional[np.ndarray]):
        """Keep the most positive detection seen across all frames."""
        if self._best is None:
            self._best = copy(result)
            if frame is not None:
                self._best_frame = frame.copy()
        else:
            self._best.logo1          = self._best.logo1 or result.logo1
            self._best.logo2          = self._best.logo2 or result.logo2
            self._best.logo3          = self._best.logo3 or result.logo3
            self._best.unknown_sticker = self._best.unknown_sticker or result.unknown_sticker
            self._best.logo1_conf     = max(self._best.logo1_conf, result.logo1_conf)
            if result.vehicle_number and not self._best.vehicle_number:
                self._best.vehicle_number = result.vehicle_number
            # Keep the clearest frame (highest logo1 confidence)
            if frame is not None and result.logo1_conf >= self._best.logo1_conf:
                self._best_frame = frame.copy()

    def _commit(self, logger: "ExcelLogger"):
        if self._best is None:
            return
            
        # Pass the captured best frame to the log function
        logger.log(
            vehicle_number=self._best.vehicle_number,
            logo1=self._best.logo1,
            logo2=self._best.logo2,
            logo3=self._best.logo3,
            frame=self._best_frame  # Ensure this is passed
        )
        
        # Existing unknown sticker logic
        if self._best.unknown_sticker:
            logger.log_unknown(
                vehicle_number=self._best.vehicle_number,
                frame=self._best_frame,
                notes="Non-AM-Motors sticker detected."
            )

    def _reset(self):
        self._state      = "IDLE"
        self._hit_count  = 0
        self._last_hit   = 0.0
        self._first_seen = 0.0
        self._best       = None
        self._best_frame = None

    # ── Read-only UI properties ───────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_tracking(self) -> bool:
        return self._state == "TRACKING"

    @property
    def tracking_duration(self) -> float:
        if self._state != "TRACKING" or self._first_seen == 0:
            return 0.0
        return time.monotonic() - self._first_seen

    @property
    def frames_seen(self) -> int:
        return self._hit_count
