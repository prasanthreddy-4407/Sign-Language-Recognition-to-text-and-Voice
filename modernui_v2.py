"""
modernui_v2.py  —  NEW main entry point
Sign Language Recognition with all backend improvements:
  • GPU / CUDA / MPS auto-detection
  • HandProcessor  (wrist-relative + scale-normalised landmarks)
  • PredictionSmootherV2  (FPS-adaptive, oscillation guard, EMA)
  • SessionLogger  (timestamped JSON log saved on exit)
  • Frame-skip inference  (CNN runs every other frame at 25+ FPS)
  • Professional UI  (ui_utils_enhanced.py)
"""

import os
import time
import warnings
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import mediapipe as mp
import torch
import numpy as np

from torch.nn import (
    Linear, ReLU, Sequential, Conv1d, MaxPool1d,
    Module, BatchNorm1d, Dropout,
)

# ── New backend modules ──────────────────────────────────────────
from core.hand_processor       import HandProcessor
from core.session_logger        import SessionLogger
from core.prediction_smoother_v2 import PredictionSmootherV2
from core.word_builder          import WordBuilder
from core.tts_manager           import TTSManager
from core.ui_utils_enhanced     import UIOverlayEnhanced
from models.CNNModel            import CNNModel


# ════════════════════════════════════════════════════════════════
# 1. DEVICE SELECTION  (GPU → MPS → CPU)
# ════════════════════════════════════════════════════════════════
def select_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}  →  using CUDA")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("🚀 Apple Silicon MPS detected  →  using MPS")
    else:
        dev = torch.device("cpu")
        print("💻 No GPU found  →  using CPU")
    return dev


DEVICE = select_device()


# ════════════════════════════════════════════════════════════════
# 2. MODEL  (combined 36-class → fallback alphabet 26-class)
# ════════════════════════════════════════════════════════════════
class CombinedCNNModel(Module):
    """36-class CNN  (A-Z + 0-9)"""
    def __init__(self):
        super().__init__()
        self.cnnLayers = Sequential(
            Conv1d(63, 32, 3, 1, 2), BatchNorm1d(32), ReLU(),
            Conv1d(32, 64, 3, 1, 2), BatchNorm1d(64), ReLU(),
            MaxPool1d(2, 2),
            Conv1d(64, 128, 3, 1, 2), BatchNorm1d(128), ReLU(), Dropout(0.3),
            Conv1d(128, 256, 3, 1, 2), BatchNorm1d(256), ReLU(),
            MaxPool1d(2, 2),
            Conv1d(256, 512, 5, 1, 2), BatchNorm1d(512), ReLU(),
            MaxPool1d(2, 2),
            Conv1d(512, 512, 5, 1, 2), BatchNorm1d(512), ReLU(), Dropout(0.4),
        )
        self.linearLayers = Sequential(
            Linear(512, 36), BatchNorm1d(36), ReLU()
        )
    def forward(self, x):
        x = self.cnnLayers(x)
        x = x.view(x.size(0), -1)
        return self.linearLayers(x)


class AlphabetCNNModel(Module):
    """26-class CNN  (A-Z only)"""
    def __init__(self):
        super().__init__()
        self.cnnLayers = Sequential(
            Conv1d(63, 32, 3, 1, 2), BatchNorm1d(32), ReLU(),
            Conv1d(32, 64, 3, 1, 2), BatchNorm1d(64), ReLU(),
            MaxPool1d(2, 2),
            Conv1d(64, 128, 3, 1, 2), BatchNorm1d(128), ReLU(), Dropout(0.3),
            Conv1d(128, 256, 3, 1, 2), BatchNorm1d(256), ReLU(),
            MaxPool1d(2, 2),
            Conv1d(256, 512, 5, 1, 2), BatchNorm1d(512), ReLU(),
            MaxPool1d(2, 2),
            Conv1d(512, 512, 5, 1, 2), BatchNorm1d(512), ReLU(), Dropout(0.4),
        )
        self.linearLayers = Sequential(
            Linear(512, 26), BatchNorm1d(26), ReLU()
        )
    def forward(self, x):
        x = self.cnnLayers(x)
        x = x.view(x.size(0), -1)
        return self.linearLayers(x)


print("🔄 Loading model …")
try:
    model      = CombinedCNNModel()
    weights    = torch.load("models/CNN_model_combined_SIBI.pth",
                            map_location=DEVICE)
    model.load_state_dict(weights)
    model_type  = "combined"
    model_label = "COMBINED A-Z + 0-9"
    CLASSES = {
        **{ch: i for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")},
        **{str(d): 26 + d for d in range(10)},
    }
    print("✅ Combined model loaded  (36 classes: A-Z + 0-9)")
except FileNotFoundError:
    model      = AlphabetCNNModel()
    weights    = torch.load("models/CNN_model_alphabet_SIBI.pth",
                            map_location=DEVICE)
    model.load_state_dict(weights)
    model_type  = "alphabet"
    model_label = "ALPHABET A-Z"
    CLASSES = {ch: i for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    print("✅ Alphabet model loaded  (26 classes: A-Z)")

model.to(DEVICE)
model.eval()
IDX_TO_CLASS = {v: k for k, v in CLASSES.items()}
print(f"🖥️  Inference device : {DEVICE}")


# ════════════════════════════════════════════════════════════════
# 3. MEDIAPIPE
# ════════════════════════════════════════════════════════════════
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles
hand_det    = mp_hands.Hands(
    static_image_mode      = True,
    min_detection_confidence = 0.25,
    max_num_hands          = 1,
)


# ════════════════════════════════════════════════════════════════
# 4. BACKEND COMPONENTS
# ════════════════════════════════════════════════════════════════
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Cannot open webcam — check connection / permissions.")
    raise SystemExit(1)

frame_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📷 Camera  {frame_w}×{frame_h}")

hand_proc   = HandProcessor(device=DEVICE)

smoother    = PredictionSmootherV2(
    target_fps       = 30.0,
    stability_frames = 5,
    min_confidence   = 0.60,
    cooldown_sec     = 0.55,
    osc_threshold    = 0.18,
)

word_builder = WordBuilder(
    pause_threshold = 2.0,
    use_dictionary  = True,
    max_suggestions = 3,
)

tts = TTSManager(rate=150, volume=0.9, auto_speak=True)

ui  = UIOverlayEnhanced(video_width=frame_w, video_height=frame_h)

logger = SessionLogger(log_dir="logs")

# Letter history for UI strip
letter_history: deque = deque(maxlen=6)


# ════════════════════════════════════════════════════════════════
# 5. MAIN LOOP
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("🎯  SIGN LANGUAGE RECOGNITION  v2  —  ENHANCED BACKEND")
print("=" * 68)
print("\n⌨️  Controls:")
print("  Space      → Finalize word        T       → Speak word")
print("  Backspace  → Delete letter        Shift+T → Speak sentence")
print("  D          → Delete word          A       → Toggle auto-speak")
print("  C          → Clear sentence       S       → Save sentence")
print("  Q          → Quit\n")

fps_prev   = time.time()
fps        = 30.0
frame_idx  = 0
save_count = 0

# Cache last inference result for frame-skip
last_pred  = None
last_conf  = 0.0
last_box   = None

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Frame capture failed — stopping.")
            break

        frame_idx += 1
        now        = time.time()
        fps        = 0.9 * fps + 0.1 * (1.0 / max(now - fps_prev, 1e-4))
        fps_prev   = now
        smoother.set_fps(fps)

        height, width = frame.shape[:2]

        # ── Hand detection (every frame — lightweight) ─────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand_det.process(rgb)
        hand_box = None

        # ── CNN inference (frame-skip at high FPS) ─────────────────
        run_inference = (fps < 25) or (frame_idx % 2 == 0)

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame, hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            # Extract features (new normalisation)
            tensor, hand_box = hand_proc.extract(hand_lm, width, height)

            if tensor is not None and run_inference:
                with torch.no_grad():
                    logits      = model(tensor)
                    probs       = torch.nn.functional.softmax(logits, dim=1)
                    conf, idx   = torch.max(probs, 1)
                    last_conf   = float(conf.cpu())
                    last_pred   = IDX_TO_CLASS.get(int(idx.cpu()), "?")
                    last_box    = hand_box
            elif tensor is None:
                last_pred, last_conf, last_box = None, 0.0, None

        else:
            last_pred, last_conf, last_box = None, 0.0, None
            smoother.add_prediction(None, 0.0)

        # ── Smoother ───────────────────────────────────────────────
        stable = smoother.add_prediction(last_pred, last_conf)
        display_conf = smoother.smooth_confidence

        # ── Word builder + logger ─────────────────────────────────
        if stable:
            letter_history.append(stable)
            logger.log_letter(stable, last_conf)
            word_done = word_builder.add_letter(stable)
            if word_done:
                completed = word_builder.words[-1]
                logger.log_word(completed)
                print(f"💬 Word: {completed}")
                if tts.auto_speak:
                    tts.speak_word(completed)

        current_word = word_builder.get_current_word()
        sentence     = word_builder.get_sentence()
        suggestions  = word_builder.get_suggestions()

        # ── Compose UI frame ───────────────────────────────────────
        final = ui.compose_frame(
            video_frame      = frame,
            prediction       = last_pred,
            hand_box         = last_box,
            confidence       = display_conf if last_pred else None,
            current_word     = current_word,
            sentence         = sentence,
            suggestions      = suggestions,
            is_speaking      = tts.is_speaking,
            auto_speak_enabled = tts.auto_speak,
            fps              = fps,
            model_type       = model_label,
            letter_history   = list(letter_history),
        )

        cv2.imshow("Sign Language Recognition v2", final)

        # ── Keyboard handling ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            print("\n👋 Quitting …")
            break

        elif key == ord(" "):
            done = word_builder.finalize_word()
            if done:
                logger.log_word(done)
                print(f"💬 Word finalised: {done}")

        elif key == 8:   # Backspace
            word_builder.delete_last_letter()

        elif key in (127, ord("d"), ord("D")):
            word_builder.delete_current_word()

        elif key == ord("t"):
            w = word_builder.get_current_word()
            if w:
                tts.speak_word(w)

        elif key == ord("T"):
            s = word_builder.get_sentence()
            if s:
                tts.speak_sentence(s)
                logger.log_sentence(s)

        elif key in (ord("a"), ord("A")):
            tts.toggle_auto_speak()
            print(f"🔊 Auto-speak: {'ON' if tts.auto_speak else 'OFF'}")

        elif key in (ord("c"), ord("C")):
            s = word_builder.get_sentence()
            if s:
                logger.log_sentence(s)
            word_builder.clear_sentence()
            print("🗑️  Sentence cleared")

        elif key in (ord("s"), ord("S")):
            s = word_builder.get_sentence()
            if s:
                save_count += 1
                fname = f"sentence_{save_count}.txt"
                with open(fname, "w") as f:
                    f.write(s)
                logger.log_sentence(s)
                print(f"💾 Saved: {fname}")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted")
except Exception as exc:
    import traceback
    print(f"\n❌ ERROR: {exc}")
    traceback.print_exc()
finally:
    tts.stop()
    cam.release()
    cv2.destroyAllWindows()
    logger.save()
    print("✅ Closed cleanly.")
