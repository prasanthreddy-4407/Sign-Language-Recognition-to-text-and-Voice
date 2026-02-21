"""
Word Builder for Sign Language Recognition
Intelligently forms words from detected letters with pause detection and suggestions
"""

import time
from typing import List, Optional
from spellchecker import SpellChecker


class WordBuilder:
    """
    Manages word and sentence formation from individual letter predictions.
    Supports pause detection, manual finalization, and word suggestions.
    """
    
    def __init__(
        self,
        pause_threshold: float = 2.0,
        use_dictionary: bool = True,
        max_suggestions: int = 3
    ):
        """
        Initialize the word builder.
        
        Args:
            pause_threshold: Seconds of pause to auto-finalize word
            use_dictionary: Enable dictionary-based suggestions
            max_suggestions: Maximum number of word suggestions to show
        """
        self.pause_threshold = pause_threshold
        self.use_dictionary = use_dictionary
        self.max_suggestions = max_suggestions
        
        # Current letter buffer
        self.letter_buffer = []
        self.letter_timestamps = []
        
        # Completed words in the sentence
        self.words = []
        
        # Last letter addition time
        self.last_letter_time = 0
        
        # Initialize spell checker
        if use_dictionary:
            try:
                self.spell = SpellChecker()
            except:
                print("⚠️  Warning: Could not load spell checker. Word suggestions disabled.")
                self.use_dictionary = False
                self.spell = None
        else:
            self.spell = None
    
    def add_letter(self, letter: str) -> bool:
        """
        Add a letter to the current word buffer.
        
        Args:
            letter: The detected letter
            
        Returns:
            True if a word was auto-finalized due to pause, False otherwise
        """
        current_time = time.time()
        
        # Check for pause (auto-finalize)
        if self.letter_buffer and (current_time - self.last_letter_time) > self.pause_threshold:
            self.finalize_word()
            # Now add the new letter to start a new word
            self.letter_buffer = [letter.upper()]
            self.letter_timestamps = [current_time]
            self.last_letter_time = current_time
            return True
        
        # Add letter to buffer
        self.letter_buffer.append(letter.upper())
        self.letter_timestamps.append(current_time)
        self.last_letter_time = current_time
        
        return False
    
    def finalize_word(self) -> Optional[str]:
        """
        Finalize the current word and add it to the sentence.
        
        Returns:
            The completed word, or None if buffer is empty
        """
        if not self.letter_buffer:
            return None
        
        word = ''.join(self.letter_buffer)
        self.words.append(word)
        
        # Clear the buffer
        completed_word = word
        self.letter_buffer = []
        self.letter_timestamps = []
        
        return completed_word
    
    def delete_last_letter(self) -> bool:
        """
        Delete the last letter from the current word buffer.
        
        Returns:
            True if a letter was deleted, False if buffer is empty
        """
        if self.letter_buffer:
            self.letter_buffer.pop()
            if self.letter_timestamps:
                self.letter_timestamps.pop()
            self.last_letter_time = time.time()
            return True
        return False
    
    def delete_current_word(self) -> bool:
        """
        Delete the entire current word buffer.
        
        Returns:
            True if word was deleted, False if buffer was already empty
        """
        if self.letter_buffer:
            self.letter_buffer = []
            self.letter_timestamps = []
            self.last_letter_time = time.time()
            return True
        return False
    
    def delete_last_word_from_sentence(self) -> bool:
        """
        Delete the last completed word from the sentence.
        
        Returns:
            True if a word was deleted, False if no words in sentence
        """
        if self.words:
            self.words.pop()
            return True
        return False
    
    def get_current_word(self) -> str:
        """Get the current word being formed."""
        return ''.join(self.letter_buffer)
    
    def get_sentence(self) -> str:
        """Get the complete sentence with all finalized words."""
        return ' '.join(self.words)
    
    def get_full_display(self) -> str:
        """Get sentence + current word for display."""
        sentence = self.get_sentence()
        current = self.get_current_word()
        
        if sentence and current:
            return f"{sentence} {current}"
        elif sentence:
            return sentence
        elif current:
            return current
        else:
            return ""
    
    def get_suggestions(self) -> List[str]:
        """
        Get word suggestions based on current letter buffer.
        
        Returns:
            List of suggested words
        """
        if not self.use_dictionary or not self.spell or not self.letter_buffer:
            return []
        
        current_word = self.get_current_word().lower()
        
        # If word is too short, don't suggest
        if len(current_word) < 2:
            return []
        
        try:
            # Get candidates from spell checker
            candidates = self.spell.candidates(current_word)
            
            if not candidates:
                return []
            
            # Filter to words that start with the current letters
            suggestions = [
                word.upper() for word in candidates 
                if word.lower().startswith(current_word.lower())
            ]
            
            # Return top N suggestions
            return suggestions[:self.max_suggestions]
        
        except Exception as e:
            # Silently fail if spell checker has issues
            return []
    
    def clear_sentence(self):
        """Clear the entire sentence and current word."""
        self.letter_buffer = []
        self.letter_timestamps = []
        self.words = []
        self.last_letter_time = 0
    
    def get_stats(self) -> dict:
        """
        Get statistics about the current state.
        
        Returns:
            Dictionary with current stats
        """
        current_time = time.time()
        time_since_last = current_time - self.last_letter_time if self.last_letter_time > 0 else 0
        
        return {
            'current_word': self.get_current_word(),
            'letter_count': len(self.letter_buffer),
            'word_count': len(self.words),
            'sentence': self.get_sentence(),
            'time_since_last_letter': time_since_last,
            'will_auto_finalize': time_since_last >= self.pause_threshold and len(self.letter_buffer) > 0,
            'suggestions': self.get_suggestions()
        }
