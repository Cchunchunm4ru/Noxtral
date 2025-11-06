from .wrapper_stt import BaseSTT
from faster_whisper import WhisperModel
import tempfile
import wave
import os
import nltk 
from events.EventManager import EventManager

# Create a global event manager instance
event_manager = EventManager()

class WhisperSTT(BaseSTT):
    def __init__(self, model_name="base", sample_rate=16000):
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self.sample_rate = sample_rate
        self.transcribed_words = []  # Keep track of all transcribed words
        self.word_count = 0  # Track word count for recursive printing
    
    def save_wav(self, filename, audio_bytes, sample_rate=16000, channels=1):
        """Save audio bytes to a WAV file"""
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
            
    def chunk_to_wav(self, audio_chunk: bytes) -> str:
        """Convert audio chunk to a temporary WAV file and return its path"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            self.save_wav(temp_filename, audio_chunk, self.sample_rate)
        return temp_filename
    
    def print_words_recursively(self, words_list, index=0):
        """Recursively print words starting from the given index"""
        if index >= len(words_list):
            return
        
        print(f"Word {index + 1}: {words_list[index]}")
        self.print_words_recursively(words_list, index + 1)
    
    def chunk_transcription_logger(self, audio_chunk: bytes) -> str:
        """Log transcription of a single audio chunk (for debugging)"""
        result = self.transcribe_audio(audio_chunk)
        print(f"Chunk transcription: {result}")
        return result
    
    def transcribe_chunk(self, audio_chunk: bytes) -> str:
        """Transcribe a single audio chunk - required by BaseSTT"""
        return self.transcribe_audio(audio_chunk)
    
    def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe raw audio data bytes"""
        wav_path = self.chunk_to_wav(audio_data)
        
        try:
            segments, info = self.model.transcribe(wav_path, beam_size=5)
            transcribed_text = ""
            
            for segment in segments:
                transcribed_text += segment.text + " "
                words = segment.text.strip().split()
                for word in words:
                    if word:  
                        self.transcribed_words.append(word)
                        self.word_count += 1
                        
                        # Trigger event instantly for each word detected
                        event_manager.publish_stt_activates(
                            transcription=word,
                        )
                        
                        if self.word_count % 2 == 0:
                            self.print_words_recursively(self.transcribed_words[-2:])
            
            return transcribed_text.strip()
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def finalize(self) -> str:
        """Finalize transcription and print all words recursively"""
        return " ".join(self.transcribed_words)

