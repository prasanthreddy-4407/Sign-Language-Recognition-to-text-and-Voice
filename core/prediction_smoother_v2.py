"""
core/prediction_smoother_v2.py  —  NEW
Improved PredictionSmoother with:
  • FPS-aware adaptive window & cooldown
  • Confidence dead-zone rejection (oscillation guard)
  • Exponential moving average (EMA) confidence smoothing
"""

import time
from collections import deque
from typing import Optional, Tuple


class PredictionSmootherV2:
    """
    Adaptive temporal smoother for sign language letter predictions.

    Key improvements over V1
    -------------------------
    * Window size and cooldown adapt to live FPS so behaviour is
      consistent at 10 FPS or 60 FPS.
    * Confidence dead-zone: if confidence oscillates more than
      `osc_threshold` over the last few frames the prediction is
      suppressed (prevents flickering between two similar classes).
    * EMA on confidence scores for a smoother confidence readout.
    """

    def __init__(
        self,
        target_fps:      float = 30.0,
        stability_frames: int  = 5,
        min_confidence:  float = 0.60,
        cooldown_sec:    float = 0.55,
        osc_threshold:   float = 0.18,
        ema_alpha:       float = 0.30,
    ):
        """
        Parameters
        ----------
        target_fps       : expected frame rate (updated live via set_fps)
        stability_frames : consecutive same-label frames needed
        min_confidence   : per-frame confidence floor
        cooldown_sec     : minimum time between accepted letters
        osc_threshold    : max confidence std-dev allowed (oscillation guard)
        ema_alpha        : EMA smoothing factor for confidence display
        """
        self.target_fps       = target_fps
        self.stability_frames = stability_frames
        self.min_confidence   = min_confidence
        self.cooldown_sec     = cooldown_sec
        self.osc_threshold    = osc_threshold
        self.ema_alpha        = ema_alpha

        # Derived from FPS
        self._window_size = max(6, int(target_fps * 0.4))

        # Rolling history : (prediction, confidence, timestamp)
        self._history: deque = deque(maxlen=self._window_size)

        self._cur_pred       = None
        self._consec_count   = 0
        self._last_stable    = None
        self._last_stable_ts = 0.0

        # EMA state
        self._ema_conf: float = 0.0

    # ── Public API ────────────────────────────────────────────────

    def set_fps(self, fps: float):
        """Call this each frame so the smoother adapts to actual FPS."""
        if fps < 5:
            fps = 5
        self.target_fps   = fps
        new_window        = max(6, int(fps * 0.35))
        if new_window != self._window_size:
            self._window_size = new_window
            self._history     = deque(self._history, maxlen=new_window)

    def add_prediction(
        self,
        prediction: Optional[str],
        confidence:  float,
    ) -> Optional[str]:
        """
        Submit one frame's prediction.

        Returns a stable letter string if all conditions are met,
        otherwise returns None.
        """
        now = time.time()

        # ── Low quality → reset streak ────────────────────────────
        if prediction is None or confidence < self.min_confidence:
            self._history.append((None, 0.0, now))
            self._cur_pred     = None
            self._consec_count = 0
            self._ema_conf     = max(0.0, self._ema_conf - 0.05)
            return None

        # ── Update EMA confidence ─────────────────────────────────
        self._ema_conf = (
            self.ema_alpha * confidence +
            (1 - self.ema_alpha) * self._ema_conf
        )

        # ── Store history ─────────────────────────────────────────
        self._history.append((prediction, confidence, now))

        # ── Consecutive count ─────────────────────────────────────
        if prediction == self._cur_pred:
            self._consec_count += 1
        else:
            self._cur_pred     = prediction
            self._consec_count = 1

        if self._consec_count < self.stability_frames:
            return None

        # ── Oscillation / dead-zone guard ─────────────────────────
        recent_confs = [c for p, c, _ in self._history if p == prediction]
        if len(recent_confs) >= 3:
            import statistics
            if statistics.stdev(recent_confs) > self.osc_threshold:
                return None   # confidence is too noisy → suppress

        # ── Cooldown ──────────────────────────────────────────────
        if now - self._last_stable_ts < self.cooldown_sec:
            return None

        # ── Same-letter duplicate guard ───────────────────────────
        if prediction == self._last_stable:
            return None

        # ── Accept ────────────────────────────────────────────────
        self._last_stable    = prediction
        self._last_stable_ts = now
        return prediction

    @property
    def smooth_confidence(self) -> float:
        """EMA-smoothed confidence value for UI display."""
        return round(self._ema_conf, 4)

    def get_majority(self) -> Optional[Tuple[str, float]]:
        """Majority vote over current window."""
        counts: dict = {}
        conf_sums: dict = {}
        for p, c, _ in self._history:
            if p is not None:
                counts[p]    = counts.get(p, 0) + 1
                conf_sums[p] = conf_sums.get(p, 0.0) + c
        if not counts:
            return None
        best = max(counts, key=counts.get)
        return best, conf_sums[best] / counts[best]

    def reset(self):
        """Reset internal state (keep cooldown timer)."""
        self._history.clear()
        self._cur_pred     = None
        self._consec_count = 0
        self._ema_conf     = 0.0
