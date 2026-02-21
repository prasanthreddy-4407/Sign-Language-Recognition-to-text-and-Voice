"""
core/hand_processor.py  —  NEW
Robust hand landmark extraction & normalization.

Improvements over the original inline code:
  • Wrist-relative coordinates  (subtract landmark-0 so position-invariant)
  • Scale normalization          (divide by wrist→middle-finger-tip distance)
  • Bounding box with padding
  • Returns None cleanly on bad/empty landmarks
"""

import numpy as np
import torch
from typing import Optional, Tuple


class HandProcessor:
    """
    Encapsulates MediaPipe landmark → feature tensor pipeline.

    Usage
    -----
    hp = HandProcessor()
    tensor, bbox = hp.extract(hand_landmarks, frame_w, frame_h)
    """

    # MediaPipe landmark index for the middle-finger tip (used for scale)
    _MIDDLE_TIP_IDX = 12

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    def extract(
        self,
        hand_landmarks,
        frame_w: int,
        frame_h: int,
        padding: int = 12,
    ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int, int, int]]]:
        """
        Extract and normalise landmarks into a model-ready tensor.

        Parameters
        ----------
        hand_landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        frame_w, frame_h : pixel dimensions of the source frame
        padding : bounding-box padding in pixels

        Returns
        -------
        tensor : (1, 63, 1) float32 on self.device, or None on failure
        bbox   : (x1, y1, x2, y2) pixel coords, or None on failure
        """
        try:
            lms = hand_landmarks.landmark
            n   = len(lms)
            if n == 0:
                return None, None

            xs = np.array([lm.x for lm in lms], dtype=np.float64)
            ys = np.array([lm.y for lm in lms], dtype=np.float64)
            zs = np.array([lm.z for lm in lms], dtype=np.float64)

            # ── 1. Wrist-relative (landmark 0 = wrist) ──────────────
            xs -= xs[0]
            ys -= ys[0]
            zs -= zs[0]

            # ── 2. Scale normalisation ───────────────────────────────
            # Distance from wrist (0) to middle-finger tip (12)
            scale = np.sqrt(
                xs[self._MIDDLE_TIP_IDX] ** 2 +
                ys[self._MIDDLE_TIP_IDX] ** 2 +
                zs[self._MIDDLE_TIP_IDX] ** 2
            )
            if scale < 1e-6:
                # Fall back to min-max range normalisation
                r = max(xs.ptp(), ys.ptp(), 1e-6)
                xs /= r;  ys /= r;  zs /= r
            else:
                xs /= scale;  ys /= scale;  zs /= scale

            # ── 3. Build feature vector (63 values: x0..x20, y0..y20, z0..z20) ─
            features = np.concatenate([xs, ys, zs]).astype(np.float32)  # (63,)
            tensor   = torch.from_numpy(
                features.reshape(1, 63, 1)
            ).to(self.device)

            # ── 4. Bounding box (pixel space, un-normalised) ─────────
            raw_lms = hand_landmarks.landmark   # original (not shifted)
            rx = [lm.x for lm in raw_lms]
            ry = [lm.y for lm in raw_lms]
            x1 = max(0,       int(min(rx) * frame_w) - padding)
            y1 = max(0,       int(min(ry) * frame_h) - padding)
            x2 = min(frame_w, int(max(rx) * frame_w) + padding)
            y2 = min(frame_h, int(max(ry) * frame_h) + padding)

            return tensor, (x1, y1, x2, y2)

        except Exception as exc:
            print(f"⚠️  HandProcessor.extract error: {exc}")
            return None, None
