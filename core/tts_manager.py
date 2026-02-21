"""
Text-to-Speech Manager for Sign Language Recognition
Provides voice output with configurable settings and non-blocking operation
"""

import threading
from queue import Queue, Empty
from typing import Optional
import time


class TTSManager:
    """
    Manages text-to-speech functionality with non-blocking operation.
    Supports auto-speak mode and configurable voice settings.
    """
    
    def __init__(
        self,
        rate: int = 150,
        volume: float = 0.9,
        auto_speak: bool = False
    ):
        """
        Initialize the TTS manager.
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
            auto_speak: Enable auto-speak on word completion
        """
        self.rate = rate
        self.volume = volume
        self.auto_speak = auto_speak
        
        # TTS engine
        self.engine = None
        self.engine_available = False
        
        # Speech queue for non-blocking operation
        self.speech_queue = Queue()
        self.is_speaking = False
        
        # Worker thread
        self.worker_thread = None
        self.should_stop = False
        
        # Initialize engine
        self._init_engine()
        
        # Start worker thread
        if self.engine_available:
            self._start_worker()
    
    def _init_engine(self):
        """Initialize the pyttsx3 engine."""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            self.engine_available = True
            print("✅ Text-to-Speech initialized successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize TTS: {e}")
            print("   Voice output will be disabled.")
            self.engine_available = False
    
    def _start_worker(self):
        """Start the worker thread for processing speech queue."""
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()
    
    def _speech_worker(self):
        """Worker thread that processes speech requests from the queue."""
        while not self.should_stop:
            try:
                # Get text from queue with timeout
                text = self.speech_queue.get(timeout=0.1)
                
                if text and self.engine_available:
                    self.is_speaking = True
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as e:
                        print(f"⚠️  TTS error: {e}")
                    finally:
                        self.is_speaking = False
                
                self.speech_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"⚠️  Speech worker error: {e}")
    
    def speak(self, text: str, async_mode: bool = True):
        """
        Speak the given text.
        
        Args:
            text: Text to speak
            async_mode: If True, speak asynchronously; if False, block until done
        """
        if not text or not self.engine_available:
            return
        
        if async_mode:
            # Add to queue for async processing
            self.speech_queue.put(text)
        else:
            # Speak synchronously (blocking)
            try:
                self.is_speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"⚠️  TTS error: {e}")
            finally:
                self.is_speaking = False
    
    def speak_word(self, word: str):
        """Speak a single word."""
        if word:
            self.speak(word, async_mode=True)
    
    def speak_sentence(self, sentence: str):
        """Speak a full sentence."""
        if sentence:
            self.speak(sentence, async_mode=True)
    
    def toggle_auto_speak(self) -> bool:
        """
        Toggle auto-speak mode.
        
        Returns:
            New auto-speak state
        """
        self.auto_speak = not self.auto_speak
        return self.auto_speak
    
    def set_rate(self, rate: int):
        """
        Set speech rate.
        
        Args:
            rate: Words per minute (typically 100-200)
        """
        if self.engine_available:
            self.rate = rate
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        """
        Set speech volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if self.engine_available:
            self.volume = max(0.0, min(1.0, volume))
            self.engine.setProperty('volume', self.volume)
    
    def get_available_voices(self) -> list:
        """
        Get list of available voices.
        
        Returns:
            List of voice objects
        """
        if self.engine_available:
            try:
                return self.engine.getProperty('voices')
            except:
                return []
        return []
    
    def set_voice(self, voice_id: int = 0):
        """
        Set voice by index.
        
        Args:
            voice_id: Index of voice in available voices list
        """
        if self.engine_available:
            try:
                voices = self.get_available_voices()
                if 0 <= voice_id < len(voices):
                    self.engine.setProperty('voice', voices[voice_id].id)
            except Exception as e:
                print(f"⚠️  Could not set voice: {e}")
    
    def stop(self):
        """Stop the TTS manager and cleanup."""
        self.should_stop = True
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        
        if self.engine_available and self.engine:
            try:
                self.engine.stop()
            except:
                pass
    
    def get_stats(self) -> dict:
        """
        Get current TTS statistics.
        
        Returns:
            Dictionary with current stats
        """
        return {
            'available': self.engine_available,
            'auto_speak': self.auto_speak,
            'is_speaking': self.is_speaking,
            'rate': self.rate,
            'volume': self.volume,
            'queue_size': self.speech_queue.qsize()
        }
