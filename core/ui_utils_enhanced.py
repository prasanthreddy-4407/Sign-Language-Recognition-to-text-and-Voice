"""
Professional UI for Real-Time Sign Language Recognition
Dark dashboard design with glow ring, letter history, gradient bars, and animated indicators.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time
from typing import Optional, List, Tuple
from collections import deque


class UIOverlayEnhanced:
    """
    Professional dark-dashboard UI renderer.
    Features: glow ring, letter history strip, gradient confidence bar,
    pill suggestions, animated speaking waveform, FPS + model badge.
    """

    def __init__(self, video_width=640, video_height=480):
        self.video_width = video_width
        self.video_height = video_height
        self.panel_width = 420
        self.total_width = video_width + self.panel_width
        self.total_height = video_height

        # ── Color palette ──────────────────────────────────────────
        self.bg_dark        = (10,  12,  24)   # deep navy
        self.panel_bg       = (18,  22,  40)   # slightly lighter navy
        self.section_bg     = (25,  30,  52)   # card background
        self.border_dim     = (40,  50,  80)   # subtle border
        self.accent_cyan    = (0,  220, 200)   # cyan accent
        self.accent_green   = (50, 230, 120)   # green
        self.accent_yellow  = (255, 210,  50)  # yellow
        self.accent_orange  = (255, 140,  40)  # orange
        self.accent_red     = (255,  70,  70)  # red
        self.text_bright    = (230, 235, 255)  # bright white-blue
        self.text_dim       = (100, 115, 150)  # muted
        self.speaking_col   = (255,  80,  80)  # waveform red

        # ── Fonts ──────────────────────────────────────────────────
        self.font_xl     = self._load_font(80)
        self.font_large  = self._load_font(42)
        self.font_medium = self._load_font(28)
        self.font_normal = self._load_font(20)
        self.font_small  = self._load_font(16)
        self.font_tiny   = self._load_font(13)

        # ── Animation state ─────────────────────────────────────────
        self._frame_count = 0

    # ──────────────────────────────────────────────────────────────
    # Font loading
    # ──────────────────────────────────────────────────────────────
    def _load_font(self, size):
        paths = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for p in paths:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    # ──────────────────────────────────────────────────────────────
    # Low-level drawing helpers
    # ──────────────────────────────────────────────────────────────
    def _pil_text(self, img, text, pos, font, color):
        """Render anti-aliased text via PIL."""
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        r, g, b = color[2], color[1], color[0]  # BGR→RGB
        draw.text(pos, text, font=font, fill=(r, g, b))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def _card(self, canvas, x, y, w, h, bg=None, border=None, radius=8):
        """Draw a rounded-rectangle card."""
        bg     = bg     or self.section_bg
        border = border or self.border_dim
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1)

    def _grad_bar(self, canvas, x, y, w, h, ratio):
        """Gradient bar: red → yellow → green based on ratio (0-1)."""
        # background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 35, 60), -1)
        filled = max(1, int(w * ratio))
        for i in range(filled):
            t = i / max(w - 1, 1)
            if t < 0.5:
                r = 255
                g = int(t * 2 * 200)
                b = 0
            else:
                r = int((1 - (t - 0.5) * 2) * 255)
                g = 200 + int((t - 0.5) * 2 * 55)
                b = 0
            cv2.line(canvas, (x + i, y), (x + i, y + h), (b, g, r), 1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.border_dim, 1)

    def _glow_circle(self, canvas, cx, cy, radius, color, layers=4):
        """Draw a multi-layer glow ring."""
        for i in range(layers, 0, -1):
            alpha = int(60 * i / layers)
            r_off = layers - i + 1
            overlay = canvas.copy()
            cv2.circle(overlay, (cx, cy), radius + r_off * 4, color, r_off * 2)
            cv2.addWeighted(overlay, alpha / 255.0, canvas, 1 - alpha / 255.0, 0, canvas)
        cv2.circle(canvas, (cx, cy), radius, color, 2)

    def _separator(self, canvas, x1, x2, y, color=None, thickness=1):
        color = color or self.border_dim
        cv2.line(canvas, (x1, y), (x2, y), color, thickness)

    # ──────────────────────────────────────────────────────────────
    # Hand bounding box
    # ──────────────────────────────────────────────────────────────
    def draw_styled_box(self, frame, x1, y1, x2, y2, color=None):
        """Corner-bracket hand highlight."""
        color = color or self.accent_cyan
        L, T = 25, 3
        pts = [
            # TL
            ((x1, y1), (x1 + L, y1)), ((x1, y1), (x1, y1 + L)),
            # TR
            ((x2, y1), (x2 - L, y1)), ((x2, y1), (x2, y1 + L)),
            # BL
            ((x1, y2), (x1 + L, y2)), ((x1, y2), (x1, y2 - L)),
            # BR
            ((x2, y2), (x2 - L, y2)), ((x2, y2), (x2, y2 - L)),
        ]
        for p1, p2 in pts:
            cv2.line(frame, p1, p2, color, T)

    # ──────────────────────────────────────────────────────────────
    # Panel sections
    # ──────────────────────────────────────────────────────────────
    def _draw_header(self, canvas, px, model_type, fps):
        """Header bar: app name  |  model badge  |  fps."""
        h = 44
        cv2.rectangle(canvas, (px, 0), (px + self.panel_width, h), (14, 17, 35), -1)
        self._separator(canvas, px, px + self.panel_width, h, self.accent_cyan, 2)

        canvas = self._pil_text(canvas, "SIGN LANG AI",
                                 (px + 12, 10), self.font_small, self.accent_cyan)

        badge = model_type or "ALPHABET"
        canvas = self._pil_text(canvas, badge,
                                 (px + 155, 13), self.font_tiny, self.accent_green)

        fps_str = f"{fps:.0f} FPS" if fps else "-- FPS"
        canvas = self._pil_text(canvas, fps_str,
                                 (px + self.panel_width - 68, 13),
                                 self.font_tiny, self.text_dim)
        return canvas, h

    def _draw_prediction(self, canvas, px, y, prediction, confidence):
        """Large letter with glow ring + gradient confidence bar."""
        section_h = 130
        self._card(canvas, px + 10, y, self.panel_width - 20, section_h)

        cx = px + 75
        cy = y + 65

        if prediction and confidence is not None:
            # Glow ring – color matches confidence
            if confidence >= 0.75:
                ring_col = self.accent_green
            elif confidence >= 0.5:
                ring_col = self.accent_yellow
            else:
                ring_col = self.accent_orange
            self._glow_circle(canvas, cx, cy, 52, ring_col)

            # Big letter
            canvas = self._pil_text(canvas, prediction,
                                     (cx - 25, cy - 42), self.font_xl, ring_col)

            # Right side: label + bar + %
            bx = px + 145
            canvas = self._pil_text(canvas, "DETECTION",
                                     (bx, y + 18), self.font_tiny, self.text_dim)
            canvas = self._pil_text(canvas, "Confidence",
                                     (bx, y + 52), self.font_tiny, self.text_dim)
            self._grad_bar(canvas, bx, y + 70, 220, 12, confidence)
            canvas = self._pil_text(canvas, f"{confidence * 100:.0f}%",
                                     (bx, y + 86), self.font_tiny, self.text_bright)
        else:
            cv2.circle(canvas, (cx, cy), 50, self.border_dim, 2)
            # dashed waiting indicator
            for angle in range(0, 360, 30):
                rad = np.radians(angle + self._frame_count * 4)
                x0 = int(cx + 45 * np.cos(rad))
                y0 = int(cy + 45 * np.sin(rad))
                cv2.circle(canvas, (x0, y0), 2, self.text_dim, -1)
            canvas = self._pil_text(canvas, "Waiting...",
                                     (px + 145, y + 54), self.font_small, self.text_dim)

        return canvas, y + section_h + 8

    def _draw_letter_history(self, canvas, px, y, history: List[str]):
        """Row of last N detected letters, fading older ones."""
        section_h = 48
        self._card(canvas, px + 10, y, self.panel_width - 20, section_h)
        canvas = self._pil_text(canvas, "RECENT",
                                 (px + 18, y + 6), self.font_tiny, self.text_dim)

        items = list(history)[-6:]  # last 6
        slot_w = 40
        start_x = px + 80

        for i, ch in enumerate(items):
            alpha = 80 + int(175 * (i + 1) / len(items)) if items else 255
            col = tuple(int(c * alpha / 255) for c in self.accent_yellow)
            bx = start_x + i * slot_w
            cv2.rectangle(canvas, (bx, y + 10), (bx + 32, y + 38), (30, 35, 62), -1)
            canvas = self._pil_text(canvas, ch, (bx + 6, y + 12),
                                     self.font_normal, col)

        if not items:
            canvas = self._pil_text(canvas, "none yet",
                                     (start_x, y + 16), self.font_tiny, self.text_dim)

        return canvas, y + section_h + 8

    def _draw_word_panel(self, canvas, px, y, current_word, suggestions):
        """Current word with blinking cursor + pill suggestions."""
        section_h = 90
        self._card(canvas, px + 10, y, self.panel_width - 20, section_h,
                   border=self.accent_yellow)

        canvas = self._pil_text(canvas, "CURRENT WORD",
                                 (px + 18, y + 8), self.font_tiny, self.text_dim)

        # Word + cursor
        cursor = "_" if (self._frame_count // 15) % 2 == 0 else " "
        display = (current_word + cursor) if current_word else cursor
        canvas = self._pil_text(canvas, display,
                                 (px + 18, y + 26), self.font_large, self.accent_yellow)

        # Pill suggestions
        if suggestions:
            sx = px + 18
            for s in suggestions[:3]:
                sw = len(s) * 10 + 16
                cv2.rectangle(canvas, (sx, y + 70), (sx + sw, y + 86),
                               (40, 50, 80), -1)
                cv2.rectangle(canvas, (sx, y + 70), (sx + sw, y + 86),
                               self.accent_cyan, 1)
                canvas = self._pil_text(canvas, s, (sx + 6, y + 71),
                                         self.font_tiny, self.accent_cyan)
                sx += sw + 8

        return canvas, y + section_h + 8

    def _draw_sentence_panel(self, canvas, px, y, sentence):
        """Sentence display with word-wrap inside a card."""
        section_h = 90
        self._card(canvas, px + 10, y, self.panel_width - 20, section_h,
                   border=self.accent_green)

        canvas = self._pil_text(canvas, "SENTENCE",
                                 (px + 18, y + 8), self.font_tiny, self.text_dim)

        if sentence:
            # simple word-wrap at ~28 chars
            words = sentence.split()
            lines, line = [], ""
            for w in words:
                if len(line) + len(w) + 1 <= 28:
                    line += ("" if not line else " ") + w
                else:
                    lines.append(line)
                    line = w
            if line:
                lines.append(line)

            for i, l in enumerate(lines[:3]):
                canvas = self._pil_text(canvas, l,
                                         (px + 18, y + 28 + i * 20),
                                         self.font_small, self.text_bright)
        else:
            canvas = self._pil_text(canvas, "Your sentence will appear here...",
                                     (px + 18, y + 36), self.font_tiny, self.text_dim)

        return canvas, y + section_h + 8

    def _draw_voice_controls(self, canvas, px, y, is_speaking,
                              auto_speak_enabled, has_word, has_sentence):
        """Compact voice controls row + animated waveform."""
        section_h = 70
        self._card(canvas, px + 10, y, self.panel_width - 20, section_h)

        canvas = self._pil_text(canvas, "VOICE CONTROLS",
                                 (px + 18, y + 6), self.font_tiny, self.text_dim)

        # Three status pills
        def _pill(c, bx, by, label, active, enabled):
            col  = self.accent_green   if (active and enabled) \
                   else (self.accent_cyan if enabled else self.text_dim)
            bcol = col
            cv2.rectangle(c, (bx, by), (bx + 120, by + 24), (28, 35, 58), -1)
            cv2.rectangle(c, (bx, by), (bx + 120, by + 24), bcol, 1)
            c = self._pil_text(c, label, (bx + 8, by + 5), self.font_tiny, col)
            return c

        canvas = _pill(canvas, px + 14, y + 26,
                       "🗣  Speak (T)", is_speaking and has_word, has_word)
        canvas = _pill(canvas, px + 144, y + 26,
                       "📢  All (Shift+T)", is_speaking and has_sentence, has_sentence)
        auto_label = "🔄  Auto: ON" if auto_speak_enabled else "🔄  Auto: OFF"
        canvas = _pill(canvas, px + 274, y + 26,
                       auto_label, auto_speak_enabled, True)

        # Animated waveform bars when speaking
        if is_speaking:
            bx = px + 14
            by = y + 54
            for bi in range(5):
                phase = (self._frame_count * 0.4 + bi * 0.8)
                bar_h = max(4, int(10 * abs(np.sin(phase))))
                bby = by + (12 - bar_h)
                cv2.rectangle(canvas, (bx + bi * 10, bby),
                               (bx + bi * 10 + 6, by + 12),
                               self.speaking_col, -1)
            canvas = self._pil_text(canvas, "SPEAKING",
                                     (px + 70, y + 54), self.font_tiny, self.speaking_col)

        return canvas, y + section_h + 8

    def _draw_hotkeys(self, canvas, px, y):
        """Compact color-coded hotkey reference."""
        section_h = 68
        self._card(canvas, px + 10, y, self.panel_width - 20, section_h)

        keys = [
            ("Space", "Finalize word",    self.accent_cyan),
            ("Bksp",  "Delete letter",    self.text_dim),
            ("D",     "Delete word",      self.text_dim),
            ("C",     "Clear sentence",   self.accent_orange),
            ("S",     "Save",             self.accent_green),
            ("Q",     "Quit",             self.accent_red),
        ]

        col1_x = px + 14
        col2_x = px + 14 + (self.panel_width // 2) - 20
        for i, (k, desc, col) in enumerate(keys):
            row = i % 3
            cx_ = col1_x if i < 3 else col2_x
            ry  = y + 8 + row * 19
            canvas = self._pil_text(canvas, f"[{k}]", (cx_, ry),
                                     self.font_tiny, col)
            canvas = self._pil_text(canvas, desc, (cx_ + 58, ry),
                                     self.font_tiny, self.text_dim)

        return canvas, y + section_h

    # ──────────────────────────────────────────────────────────────
    # Main compose
    # ──────────────────────────────────────────────────────────────
    def compose_frame(
        self,
        video_frame,
        prediction: Optional[str] = None,
        hand_box: Optional[Tuple[int, int, int, int]] = None,
        confidence: Optional[float] = None,
        current_letters: str = "",
        current_word: str = "",
        sentence: str = "",
        suggestions: Optional[List[str]] = None,
        is_speaking: bool = False,
        auto_speak_enabled: bool = False,
        stats: dict = None,
        fps: float = 0.0,
        model_type: str = "ALPHABET",
        letter_history: Optional[List[str]] = None,
    ):
        self._frame_count += 1

        # ── Canvas ────────────────────────────────────────────────
        canvas = np.zeros((self.total_height, self.total_width, 3), dtype=np.uint8)
        canvas[:] = self.bg_dark

        # slight gradient tint for right panel
        canvas[:, self.video_width:] = self.panel_bg

        # vertical separator
        cv2.line(canvas,
                 (self.video_width, 0),
                 (self.video_width, self.total_height),
                 self.accent_cyan, 2)

        # ── Video feed ───────────────────────────────────────────
        h, w = video_frame.shape[:2]
        canvas[:h, :w] = video_frame

        # hand box on video
        if hand_box:
            x1, y1, x2, y2 = hand_box
            self.draw_styled_box(canvas, x1, y1, x2, y2)

        # FPS overlay on video
        if fps:
            canvas = self._pil_text(canvas, f"{fps:.0f} fps",
                                     (10, 10), self.font_tiny, self.accent_cyan)

        # ── Right panel sections ──────────────────────────────────
        px = self.video_width
        ph = self.panel_width

        canvas, y = self._draw_header(canvas, px, model_type, fps)

        canvas, y = self._draw_prediction(canvas, px, y, prediction, confidence)

        canvas, y = self._draw_letter_history(
            canvas, px, y, letter_history or [])

        canvas, y = self._draw_word_panel(
            canvas, px, y, current_word, suggestions or [])

        canvas, y = self._draw_sentence_panel(canvas, px, y, sentence)

        canvas, y = self._draw_voice_controls(
            canvas, px, y, is_speaking, auto_speak_enabled,
            bool(current_word), bool(sentence))

        # push hotkeys to bottom
        hk_y = self.total_height - 78
        if y < hk_y:
            self._draw_hotkeys(canvas, px, hk_y)
        else:
            self._draw_hotkeys(canvas, px, y)

        return canvas
