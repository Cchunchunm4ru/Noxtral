from .wrapper_stt import BaseSTT
from .frame import TranscriptionFrame  # Import your frame class
from faster_whisper import WhisperModel
import tempfile
import wave
import os
from events.EventManager import EventManager
import logging 

# Configure logger
logger = logging.getLogger(__name__)
event_manager = EventManager()

class SentencePrinter:
    """Handles printing complete sentences and transcription summaries"""
    
    def __init__(self):
        self.full_transcription = ""
    
    def print_full_sentence(self, sentence: str):
        """Print the complete transcribed sentence"""
        logger.info("--- Full Transcription ---")
        logger.info(f"Complete sentence: {sentence}")
        logger.info(f"Word count: {len(sentence.split())}")
        logger.info("--- End Transcription ---")
    
    def update_transcription(self, text: str):
        """Update the full transcription text"""
        self.full_transcription = text
    
    def get_sentence_summary(self):
        """Get a summary of the transcribed sentence"""
        words = self.full_transcription.split()
        return {
            "sentence": self.full_transcription,
            "word_count": len(words),
            "character_count": len(self.full_transcription),
            "first_word": words[0] if words else "",
            "last_word": words[-1] if words else ""
        }

class WhisperSTT(BaseSTT):
    def __init__(self, model_name="large-v3-turbo", sample_rate=16000):
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self.sample_rate = sample_rate
        self.transcription_frames = []  # Store transcription frames
        self.word_count = 0  # Track word count for recursive printing
        self.sentence_printer = SentencePrinter()  # Initialize sentence printer
    
    def save_wav(self, filename, audio_bytes, sample_rate=16000, channels=1):
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
    
    def print_frames_recursively(self, frames_list, index=0):
        if index >= len(frames_list):
            return
        
        frame = frames_list[index]
        # Fix: Use .text attribute instead of .get_text() method
        text = frame.text if hasattr(frame, 'text') else (frame.get_text() if hasattr(frame, 'get_text') else str(frame))
        confidence = frame.confidence if hasattr(frame, 'confidence') else (frame.get_confidence() if hasattr(frame, 'get_confidence') else 0.0)
        logger.debug(f"Frame {index + 1}: {text} (confidence: {confidence:.2f})")
        self.print_frames_recursively(frames_list, index + 1)
    
    def chunk_transcription_logger(self, audio_chunk: bytes) -> str:
        result = self.transcribe_audio(audio_chunk)
        logger.info(f"Chunk transcription: {result}")
        return result
    
    def transcribe_chunk(self, audio_chunk: bytes) -> str:
        return self.transcribe_audio(audio_chunk)
    
    def transcribe_audio(self, audio_data: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav_path = temp_file.name
            self.save_wav(wav_path, audio_data, self.sample_rate)
        
        try:
            segments, info = self.model.transcribe(wav_path, beam_size=5)
            transcribed_text = ""
            
            for segment in segments:
                transcribed_text += segment.text + " "
                words = segment.text.strip().split()
                
                for word in words:
                    if word:  
                        # Create transcription frame for each word
                        frame = TranscriptionFrame(
                            text=word,
                            confidence=getattr(segment, 'avg_logprob', 0.0),
                            start_time=getattr(segment, 'start', 0.0),
                            end_time=getattr(segment, 'end', 0.0)
                        )
                        
                        self.transcription_frames.append(frame)
                        self.word_count += 1
                        
                        # Trigger event instantly for each word detected
                        event_manager.publish_stt_activates(
                            transcription=word,
                        )
                        
                        if self.word_count % 2 == 0:
                            self.print_frames_recursively(self.transcription_frames[-2:])
            
            return transcribed_text.strip()
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def finalize(self) -> str:
        try:
            full_sentence = " ".join([frame.text for frame in self.transcription_frames])
        except AttributeError:
            try:
                full_sentence = " ".join([frame.get_text() for frame in self.transcription_frames])
            except AttributeError:
                full_sentence = " ".join([str(frame) for frame in self.transcription_frames])
        
        self.sentence_printer.update_transcription(full_sentence)
        self.sentence_printer.print_full_sentence(full_sentence)
        return full_sentence
    
    def get_transcription_frames(self):
        """Return all transcription frames"""
        return self.transcription_frames
    
    def get_frame_by_index(self, index):
        """Get a specific transcription frame by index"""
        if 0 <= index < len(self.transcription_frames):
            return self.transcription_frames[index]
        return None

