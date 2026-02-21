"""
core/session_logger.py  —  NEW
Timestamped JSON session logger for sign language recognition.

Logs every detected letter, completed word, and finalized sentence.
Saves automatically to  logs/session_YYYYMMDD_HHMMSS.json  on close.
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any


class SessionLogger:
    """
    Records all recognition events and saves a structured JSON log.

    Usage
    -----
    logger = SessionLogger(log_dir="logs")
    logger.log_letter("A", confidence=0.92)
    logger.log_word("HELLO")
    logger.log_sentence("HELLO WORLD")
    logger.save()          # call on app exit
    stats = logger.get_stats()
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir   = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._start_ts  = time.time()
        os.makedirs(log_dir, exist_ok=True)
        self._filepath  = os.path.join(
            log_dir, f"session_{self.session_id}.json"
        )

        self._data: Dict[str, Any] = {
            "session_id":   self.session_id,
            "start_time":   datetime.now().isoformat(),
            "end_time":     None,
            "duration_sec": 0,
            "letters":      [],   # {ts, char, confidence}
            "words":        [],   # {ts, word}
            "sentences":    [],   # {ts, sentence}
        }
        print(f"📝 Session logger started  →  {self._filepath}")

    # ── Public API ────────────────────────────────────────────────

    def log_letter(self, char: str, confidence: float = 0.0):
        """Record a stably detected letter."""
        self._data["letters"].append({
            "ts":          self._elapsed(),
            "char":        char.upper(),
            "confidence":  round(float(confidence), 4),
        })

    def log_word(self, word: str):
        """Record a completed word."""
        self._data["words"].append({
            "ts":   self._elapsed(),
            "word": word.upper(),
        })
        print(f"📝  Word logged: {word}")

    def log_sentence(self, sentence: str):
        """Record a finalized sentence."""
        self._data["sentences"].append({
            "ts":       self._elapsed(),
            "sentence": sentence,
        })
        print(f"📝  Sentence logged: {sentence}")

    def get_stats(self) -> Dict[str, Any]:
        """Return live statistics."""
        elapsed = max(self._elapsed(), 1)
        n_letters   = len(self._data["letters"])
        n_words     = len(self._data["words"])
        avg_conf    = (
            sum(l["confidence"] for l in self._data["letters"]) / n_letters
            if n_letters else 0.0
        )
        return {
            "elapsed_sec":    round(elapsed, 1),
            "letters_total":  n_letters,
            "words_total":    n_words,
            "sentences_total": len(self._data["sentences"]),
            "letters_per_min": round(n_letters / elapsed * 60, 1),
            "words_per_min":   round(n_words   / elapsed * 60, 1),
            "avg_confidence":  round(avg_conf, 3),
        }

    def save(self):
        """Flush everything to disk.  Call this on app exit."""
        self._data["end_time"]     = datetime.now().isoformat()
        self._data["duration_sec"] = round(self._elapsed(), 1)
        self._data["stats"]        = self.get_stats()

        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

        s = self.get_stats()
        print(
            f"\n📊 Session saved  →  {self._filepath}\n"
            f"   Letters : {s['letters_total']}  "
            f"({s['letters_per_min']} /min)\n"
            f"   Words   : {s['words_total']}  "
            f"({s['words_per_min']} /min)\n"
            f"   Avg conf: {s['avg_confidence'] * 100:.1f}%"
        )

    # ── Private ───────────────────────────────────────────────────

    def _elapsed(self) -> float:
        return time.time() - self._start_ts
