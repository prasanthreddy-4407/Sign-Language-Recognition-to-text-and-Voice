# 🤟 Real-Time Sign Language Recognition

A real-time sign language recognition system that detects **hand gestures via webcam** and converts them into **letters, words, and spoken sentences** — powered by a CNN model, MediaPipe, Text-to-Speech, and an enhanced backend pipeline.

---

## 📂 Project Structure

```
main project/
├── modernui_v2.py         ← Main app (v2 — enhanced backend)
├── run.bat                ← Windows launcher (double-click to start)
├── requirements.txt
├── README.md
│
├── core/                  ← Runtime modules
│   ├── hand_processor.py          NEW — robust landmark normalization
│   ├── session_logger.py          NEW — JSON session logs
│   ├── prediction_smoother_v2.py  NEW — FPS-adaptive smoother
│   ├── ui_utils_enhanced.py       — Professional dark dashboard UI
│   ├── word_builder.py            — Word/sentence formation
│   └── tts_manager.py             — Text-to-speech engine
│
├── models/                ← CNN architecture & trained weights
│   ├── CNNModel.py
│   ├── CNN_model_alphabet_SIBI.pth   (A-Z, 26 classes)
│   ├── CNN_model_number_SIBI.pth     (0-9, 10 classes)
│   ├── handLandMarks.py
│   └── mediapipeHandDetection.py
│
├── training/              ← Model training & evaluation
│   ├── training.py
│   ├── training_combined.py
│   └── testCNN.py
│
├── data/                  ← Training & testing datasets (xlsx)
└── logs/                  ← Auto-saved session logs (JSON)
```

---

## 🧠 How It Works

```
Webcam
   │
   ▼
MediaPipe  →  21 hand landmarks (x, y, z)
   │
   ▼
HandProcessor  →  wrist-relative + scale-normalized (63 features)
   │
   ▼
CNN Model  →  classifies gesture → letter (A-Z or 0-9)
   │
   ▼
PredictionSmootherV2  →  FPS-adaptive window, EMA, oscillation guard
   │
   ▼
WordBuilder  →  letters → words → sentences  (pause detection + suggestions)
   │
   ▼
TTSManager  →  speaks completed words/sentences  (non-blocking thread)
   │
   ▼
Professional UI  →  dark dashboard, glow ring, letter history, waveform
   │
   ▼
SessionLogger  →  saves timestamped JSON log to logs/
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch

```bash
# Double-click:
run.bat

# Or command line:
python modernui_v2.py
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `Space` | Finalize current word |
| `Backspace` | Delete last letter |
| `D` | Delete current word |
| `T` | Speak current word |
| `Shift + T` | Speak full sentence |
| `A` | Toggle auto-speak |
| `C` | Clear sentence |
| `S` | Save sentence to `.txt` |
| `Q` | Quit |

---

## 🔧 Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥ 4.5.0 | Webcam & UI window |
| `mediapipe` | ≥ 0.10.0 | Hand landmark detection |
| `torch` | ≥ 2.0.0 | CNN inference (GPU if available) |
| `numpy` | ≥ 1.24.0 | Math ops |
| `pandas` | ≥ 2.0.0 | Dataset loading |
| `Pillow` | ≥ 10.0.0 | UI text rendering |
| `pyttsx3` | ≥ 2.90 | Text-to-speech |
| `pyspellchecker` | ≥ 0.7.0 | Word suggestions |

---

## 🚀 Backend Features (v2)

| Feature | Detail |
|---------|--------|
| **GPU support** | Auto-detects CUDA → MPS → CPU at startup |
| **Robust normalization** | Wrist-relative + scale-normalized landmarks |
| **Adaptive smoother** | Window size scales with live FPS |
| **Oscillation guard** | Suppresses noisy confidence predictions |
| **EMA confidence** | Smooth confidence readout via exponential moving average |
| **Frame-skip** | CNN runs every other frame at 25+ FPS |
| **Session logger** | Saves letters/words/sentences + stats to `logs/*.json` |

---

## 🏋️ Training Your Own Model (Optional)

```bash
# Step 1 — Collect landmark data
python models/handLandMarks.py

# Step 2 — Train
python training/training.py          # alphabet only
python training/training_combined.py # A-Z + 0-9 combined

# Step 3 — Evaluate
python training/testCNN.py
```

---

## 📝 Notes

- App auto-loads `CNN_model_combined_SIBI.pth` (36 classes) if present, otherwise falls back to `CNN_model_alphabet_SIBI.pth` (A-Z only).
- Session logs are saved to `logs/session_YYYYMMDD_HHMMSS.json` on every clean exit.
- The SIBI (Sistem Isyarat Bahasa Indonesia) dataset was used for training.

---

## 🤝 Contributing

Contributions are welcome! Open an issue or submit a pull request.
