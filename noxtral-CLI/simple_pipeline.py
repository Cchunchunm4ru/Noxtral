# =====================================================================================
# SIMPLE SPEECH ISOLATION PIPELINE - MAIN ENTRY POINT
# =====================================================================================
# This script implements a real-time speaker verification and transcription system.
# It captures audio, verifies if it matches a target speaker, and transcribes speech.
#
# SETUP REQUIREMENTS:
# 1. Create a "captured_audio" directory in the project root
# 2. Add reference .wav files of your target speaker to that directory
# 3. Ensure all dependencies are installed (see requirements.txt)
# 4. Run this script to start the 10-minute analysis session
#
# CUSTOMIZATION POINTS:
# - Change session duration by modifying duration_seconds in main()
# - Adjust audio buffer settings in __init__() for different responsiveness
# - Modify verification thresholds in _show_final_analysis() for sensitivity
# - Change STT model in __init__() for different accuracy/speed trade-offs
# =====================================================================================

import asyncio
import logging
import os
import sys
import time
import numpy as np
import io
import wave

# Import our custom components - modify these paths if you reorganize the project structure
from F_def.capture import FrameCapture                    # Real-time audio capture system
from speaker_isolation.speaker_system import initialize_speaker_system  # Speaker verification
from STT.whisper_stt import WhisperSTT                   # Speech-to-text transcription

# =====================================================================================
# LOGGING CONFIGURATION - Customize output format and level here
# =====================================================================================
# Change level to logging.DEBUG for more detailed output, or logging.WARNING for less
# Modify format string to include/exclude timestamp, level names, etc.
logging.basicConfig(
    level=logging.INFO,                                  # Change to DEBUG for verbose output
    format='%(asctime)s - %(levelname)s - %(message)s'  # Customize log message format
)
logger = logging.getLogger(__name__)

class SimplePipeline:
    """
    Main pipeline class that orchestrates speaker verification and transcription.
    
    This class handles:
    - Real-time audio capture and processing
    - Speaker verification against reference samples
    - Speech transcription of verified audio
    - Session statistics and final analysis
    """
    
    def __init__(self, captured_audio_dir="captured_audio"):
        # =====================================================================================
        # CORE CONFIGURATION - Modify these settings to customize behavior
        # =====================================================================================
        
        # Directory containing reference audio files for target speaker
        # TO CUSTOMIZE: Change path if your reference files are stored elsewhere
        self.captured_audio_dir = captured_audio_dir
        
        # Pipeline state tracking - DO NOT MODIFY unless you understand the flow
        self.running = False                    # Controls main processing loop
        self.processed_frames = 0               # Total audio segments analyzed
        self.verified_frames = 0                # Segments matching target speaker
        self.transcribed_segments = 0           # Segments successfully transcribed
        self.similarity_scores = []             # Historical similarity data
        self.target_speaker_name = None         # Name of the speaker we're looking for
        self.transcriptions = []                # Store all transcribed speech
        
        # =====================================================================================
        # AUDIO PROCESSING SETTINGS - Tune these for performance vs accuracy
        # =====================================================================================
        
        # Audio buffering for improved recognition accuracy
        self.audio_buffer = []                  # Temporary storage for audio data
        self.buffer_duration = 0.0              # Current buffer length in seconds
        
        # CUSTOMIZE: Minimum audio length required for processing
        # Smaller values = more responsive, larger values = more accurate
        self.min_segment_duration = 1.5         # Require 1.5 seconds of audio minimum
        
        # CUSTOMIZE: Audio sample rate - match your input device capabilities
        # Higher rates = better quality but more processing power needed
        self.sample_rate = 16000                # 16kHz is good balance of quality/performance
        
        # =====================================================================================
        # STT (SPEECH-TO-TEXT) CONFIGURATION - Choose model based on your needs
        # =====================================================================================
        # Available Whisper models: tiny, base, small, medium, large
        # tiny    = fastest, least accurate, ~39MB
        # base    = good balance, ~74MB  <- DEFAULT
        # small   = better accuracy, ~244MB
        # medium  = high accuracy, ~769MB
        # large   = best accuracy, ~1550MB
        
        self.stt = WhisperSTT(
            model_name="base",                  # CUSTOMIZE: Change model size here
            sample_rate=self.sample_rate
        )
        
        # Initialization complete - log the configuration for user reference
        logger.info("Audio Pipeline initialized")
        logger.info(f"Reference audio directory: {captured_audio_dir}")
        logger.info("🎤 Speech-to-Text system initialized")
        
    def initialize(self):
        """
        Initialize the pipeline components and verify setup.
        
        This method:
        1. Checks for reference audio directory
        2. Finds and selects target speaker reference files
        3. Initializes the speaker verification system
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing pipeline components...")
            
            # =====================================================================================
            # STEP 1: Verify reference audio directory exists
            # =====================================================================================
            # TO TROUBLESHOOT: If this fails, create the directory and add .wav reference files
            if not os.path.exists(self.captured_audio_dir):
                logger.error(f"Directory not found: {self.captured_audio_dir}")
                logger.error("SOLUTION: Create the directory and add reference .wav files of your target speaker")
                return False
                
            # =====================================================================================
            # STEP 2: Find reference audio files for speaker verification
            # =====================================================================================
            # TO CUSTOMIZE: Change file extension filter if using different audio formats
            reference_files = [f for f in os.listdir(self.captured_audio_dir) 
                             if f.endswith('.wav')]  # MODIFY: Add other extensions like '.mp3', '.flac'
            
            if not reference_files:
                logger.error(f"No .wav files found in {self.captured_audio_dir}")
                logger.error("SOLUTION: Record and save reference audio samples of your target speaker as .wav files")
                return False
                
            # =====================================================================================
            # STEP 3: Select target speaker (uses most recent file by default)
            # =====================================================================================
            # TO CUSTOMIZE: Modify this logic to select specific files or combine multiple references
            reference_files.sort(reverse=True)  # Sort by filename (newest first if using timestamps)
            target_speaker = os.path.splitext(reference_files[0])[0]  # Use filename without extension
            self.target_speaker_name = target_speaker
            
            logger.info(f"Target speaker: {target_speaker}")
            logger.info(f"Available reference files: {reference_files}")
            
            # =====================================================================================
            # STEP 4: Initialize the speaker verification system
            # =====================================================================================
            # This loads the pre-trained models and prepares the verification engine
            if not initialize_speaker_system(target_speaker):
                logger.error("Failed to initialize speaker system")
                logger.error("SOLUTION: Check that pretrained models are available and reference audio is valid")
                return False
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    def process_audio_frame(self, audio_bytes):
        """
        Process incoming audio frames and accumulate them for speaker verification.
        
        This method implements a buffering strategy where audio is collected until
        we have enough data for reliable speaker verification. Short audio clips
        are less reliable for speaker identification.
        
        Args:
            audio_bytes: Raw audio data from the microphone
            
        Returns:
            bytes: Original audio if speaker verified, silence if not verified
        """
        try:
            # =====================================================================================
            # STEP 1: Convert audio bytes to normalized waveform
            # =====================================================================================
            # Convert 16-bit integer samples to float32 in range [-1.0, 1.0]
            # TO CUSTOMIZE: Adjust normalization if using different audio bit depths
            waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # =====================================================================================
            # STEP 2: Add to accumulation buffer
            # =====================================================================================
            # Accumulate audio samples to build longer segments for better verification accuracy
            self.audio_buffer.extend(waveform)
            self.buffer_duration = len(self.audio_buffer) / self.sample_rate
            
            # =====================================================================================
            # STEP 3: Process when we have sufficient audio
            # =====================================================================================
            # Only attempt verification when we have enough audio data
            # TO CUSTOMIZE: Modify min_segment_duration in __init__ to change responsiveness
            if self.buffer_duration >= self.min_segment_duration:
                return self._process_accumulated_audio(audio_bytes)
            else:
                # Not enough audio yet - return original frame and continue accumulating
                return audio_bytes
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return b'\x00' * len(audio_bytes)
    
    def _process_accumulated_audio(self, current_frame_bytes):
        """
        Process accumulated audio buffer through speaker verification and transcription.
        
        This is the core processing method that:
        1. Performs speaker verification on the accumulated audio
        2. Transcribes verified audio to text
        3. Stores results and clears the buffer
        4. Returns original audio if verified, silence if not
        
        Args:
            current_frame_bytes: The most recent audio frame (for return value)
            
        Returns:
            bytes: Original audio if speaker verified, silence if rejected
        """
        self.processed_frames += 1  # Increment counter for statistics
        
        try:
            # =====================================================================================
            # STEP 1: Prepare accumulated audio for verification
            # =====================================================================================
            accumulated_waveform = np.array(self.audio_buffer, dtype=np.float32)
            
            # Import the global verifier instance (initialized in speaker_system)
            from speaker_isolation.speaker_system import _global_verifier
            
            if _global_verifier is not None:
                # =====================================================================================
                # STEP 2: Perform speaker verification
                # =====================================================================================
                # This checks if the accumulated audio matches our target speaker's voice
                # TO CUSTOMIZE: Modify verification thresholds in the speaker_system module
                recognized_segments = _global_verifier.verify_and_diarize(accumulated_waveform)
                
                if recognized_segments:
                    # =====================================================================================
                    # SPEAKER VERIFIED - Process the audio
                    # =====================================================================================
                    self.verified_frames += 1  # Increment success counter
                    
                    # Convert waveform back to audio bytes for transcription
                    audio_bytes = self._waveform_to_bytes(accumulated_waveform)
                    
                    # =====================================================================================
                    # STEP 3: Transcribe verified audio to text
                    # =====================================================================================
                    transcription = self._transcribe_audio_segment(audio_bytes)
                    
                    # Store transcription if we got meaningful text
                    if transcription.strip():
                        self.transcribed_segments += 1
                        # Create detailed transcription record
                        self.transcriptions.append({
                            'timestamp': time.time(),                                    # When it was captured
                            'duration': len(accumulated_waveform)/self.sample_rate,     # Length in seconds
                            'transcription': transcription,                             # The actual text
                            'speaker': self.target_speaker_name                         # Who said it
                        })
                        logger.info(f"🎯 TRANSCRIBED: '{transcription}' (Speaker: {self.target_speaker_name})")
                    
                    # Clear buffer and allow audio to pass through
                    self.audio_buffer = []
                    self.buffer_duration = 0.0
                    logger.info(f"✅ Speaker verified in {len(accumulated_waveform)/self.sample_rate:.2f}s segment")
                    return current_frame_bytes  # Return original audio (speaker verified)
                else:
                    # =====================================================================================
                    # SPEAKER NOT VERIFIED - Block the audio
                    # =====================================================================================
                    self.audio_buffer = []  # Clear buffer for next attempt
                    self.buffer_duration = 0.0
                    logger.info(f"❌ Speaker NOT verified in {len(accumulated_waveform)/self.sample_rate:.2f}s segment")
                    return b'\x00' * len(current_frame_bytes)  # Return silence (block unverified audio)
            else:
                # =====================================================================================
                # NO VERIFIER AVAILABLE - Default to blocking
                # =====================================================================================
                self.audio_buffer = []
                self.buffer_duration = 0.0
                return b'\x00' * len(current_frame_bytes)  # Return silence
                
        except Exception as e:
            logger.error(f"Error in accumulated audio processing: {e}")
            self.audio_buffer = []
            self.buffer_duration = 0.0
            return b'\x00' * len(current_frame_bytes)
    
    def _waveform_to_bytes(self, waveform):
        """
        Convert numpy waveform array to WAV audio bytes for STT processing.
        
        This utility function converts the normalized float32 waveform back into
        a proper WAV format that the speech-to-text system can process.
        
        Args:
            waveform: numpy array of float32 audio samples in range [-1.0, 1.0]
            
        Returns:
            bytes: WAV-formatted audio data
        """
        try:
            # =====================================================================================
            # STEP 1: Convert float32 samples back to 16-bit integers
            # =====================================================================================
            # Scale from [-1.0, 1.0] back to [-32767, 32767] range
            # TO CUSTOMIZE: Adjust scaling if using different bit depths
            waveform_int16 = (waveform * 32767).astype(np.int16)
            
            # =====================================================================================
            # STEP 2: Create WAV file in memory
            # =====================================================================================
            buffer = io.BytesIO()  # In-memory buffer for WAV data
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)                    # Mono audio
                wav_file.setsampwidth(2)                    # 16-bit samples (2 bytes each)
                wav_file.setframerate(self.sample_rate)     # Match our configured sample rate
                wav_file.writeframes(waveform_int16.tobytes())  # Write the actual audio data
            
            # Return the complete WAV file as bytes
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error converting waveform to bytes: {e}")
            return b''
    
    def _transcribe_audio_segment(self, audio_bytes):
        """
        Transcribe an audio segment to text using the Whisper STT system.
        
        Args:
            audio_bytes: WAV-formatted audio data
            
        Returns:
            str: Transcribed text, or empty string if transcription fails
        """
        try:
            # Skip processing if no audio data
            if len(audio_bytes) == 0:
                return ""
            
            # =====================================================================================
            # TRANSCRIPTION PROCESSING
            # =====================================================================================
            # Use the configured Whisper model to convert speech to text
            # TO CUSTOMIZE: The STT model was configured in __init__() - change it there
            transcription = self.stt.transcribe_audio(audio_bytes)
            return transcription
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""  # Return empty string on failure
    
    def _get_latest_similarity_score(self):
        """
        Extract the latest similarity score from the verification process.
        
        NOTE: This is a placeholder method for future enhancement.
        You could modify the speaker verification system to return similarity
        scores and store/analyze them here.
        
        Returns:
            None: Currently not implemented
        """
        # TO IMPLEMENT: Modify speaker_system to return similarity scores
        # Then store and analyze them here for detailed statistics
        return None
    
    def _show_final_analysis(self, elapsed_time):
        """
        Display comprehensive analysis results at the end of the session.
        
        This method provides detailed statistics about:
        - Speaker verification success rates
        - Transcription results and captured speech
        - Overall assessment of speaker likelihood
        
        Args:
            elapsed_time: Total session duration in seconds
        """
        logger.info("=" * 70)
        logger.info("SPEAKER VERIFICATION & TRANSCRIPTION ANALYSIS COMPLETE")
        logger.info("=" * 70)
        
        # Handle case where no audio was processed
        if self.processed_frames == 0:
            logger.info("No audio frames were processed during the session.")
            logger.info("TROUBLESHOOTING: Check microphone connection and audio input levels")
            return
        
        # =====================================================================================
        # CALCULATE VERIFICATION SUCCESS RATE
        # =====================================================================================
        verification_rate = (self.verified_frames / self.processed_frames * 100) if self.processed_frames > 0 else 0
        
        # =====================================================================================
        # DETERMINE SPEAKER LIKELIHOOD CATEGORY
        # =====================================================================================
        # TO CUSTOMIZE: Adjust these thresholds based on your accuracy requirements
        # Higher thresholds = more strict verification, lower thresholds = more permissive
        if verification_rate >= 30:              # CUSTOMIZE: Change threshold for HIGH confidence
            likelihood = "HIGH"
            confidence = "Very likely the target speaker"
            emoji = "✅"
        elif verification_rate >= 15:            # CUSTOMIZE: Change threshold for MODERATE confidence
            likelihood = "MODERATE" 
            confidence = "Possibly the target speaker"
            emoji = "⚠️"
        elif verification_rate >= 5:             # CUSTOMIZE: Change threshold for LOW confidence
            likelihood = "LOW"
            confidence = "Unlikely to be the target speaker"
            emoji = "❌"
        else:                                    # Below 5% = VERY LOW confidence
            likelihood = "VERY LOW"
            confidence = "Very unlikely to be the target speaker"
            emoji = "🚫"
        
        # =====================================================================================
        # DISPLAY SESSION SUMMARY
        # =====================================================================================
        logger.info(f"Session Duration: {elapsed_time/60:.1f} minutes")
        logger.info(f"Target Speaker: {self.target_speaker_name}")
        logger.info(f"Audio Frames Analyzed: {self.processed_frames:,}")
        logger.info(f"Frames Matching Speaker: {self.verified_frames:,}")
        logger.info("")
        logger.info(f"{emoji} SPEAKER LIKELIHOOD: {likelihood}")
        logger.info(f"   Verification Rate: {verification_rate:.1f}%")
        logger.info(f"   Assessment: {confidence}")
        logger.info("")
        
        # =====================================================================================
        # TRANSCRIPTION RESULTS SUMMARY
        # =====================================================================================
        logger.info("🎤 TRANSCRIPTION RESULTS:")
        logger.info(f"   Segments Transcribed: {self.transcribed_segments}")
        logger.info(f"   Total Transcriptions: {len(self.transcriptions)}")
        
        if self.transcriptions:
            # =====================================================================================
            # DETAILED TRANSCRIPTION BREAKDOWN
            # =====================================================================================
            logger.info("")
            logger.info("📝 CAPTURED SPEECH:")
            total_words = 0
            for i, trans in enumerate(self.transcriptions, 1):
                words = len(trans['transcription'].split())
                total_words += words
                duration = trans['duration']
                logger.info(f"   {i}. [{duration:.1f}s] \"{trans['transcription']}\" ({words} words)")
            
            # =====================================================================================
            # SPEECH ANALYSIS STATISTICS
            # =====================================================================================
            logger.info("")
            logger.info(f"📊 SPEECH STATISTICS:")
            logger.info(f"   Total Words Captured: {total_words}")
            logger.info(f"   Average Words per Segment: {total_words/len(self.transcriptions):.1f}")
            
            # =====================================================================================
            # COMPLETE CONVERSATION RECONSTRUCTION
            # =====================================================================================
            # Combine all verified speech segments into a single conversation
            full_conversation = " ".join([t['transcription'] for t in self.transcriptions])
            if full_conversation.strip():
                logger.info("")
                logger.info("💬 FULL CONVERSATION:")
                logger.info(f'   "{full_conversation}"')
        else:
            # =====================================================================================
            # NO TRANSCRIPTIONS AVAILABLE
            # =====================================================================================
            logger.info("   No speech was transcribed during verified segments")
            logger.info("   POSSIBLE CAUSES:")
            logger.info("     - Speaker verification threshold too strict")
            logger.info("     - Background noise interfering with recognition")
            logger.info("     - Reference audio quality issues")
            logger.info("     - Microphone input levels too low")
        
        # =====================================================================================
        # FINAL ASSESSMENT AND RECOMMENDATIONS
        # =====================================================================================
        logger.info("")
        if verification_rate > 0:
            logger.info("✓ Voice patterns detected matching reference samples")
            if verification_rate < 15:
                logger.info("💡 SUGGESTIONS TO IMPROVE ACCURACY:")
                logger.info("   - Record higher quality reference samples")
                logger.info("   - Reduce background noise during testing")
                logger.info("   - Speak more clearly and at consistent volume")
        else:
            logger.info("✗ No matching voice patterns detected")
            logger.info("💡 TROUBLESHOOTING SUGGESTIONS:")
            logger.info("   - Verify reference audio contains the correct speaker")
            logger.info("   - Check microphone is working and input levels are adequate")
            logger.info("   - Ensure reference and test audio have similar quality/conditions")
            logger.info("   - Consider lowering verification thresholds in the code")
            
        logger.info("=" * 70)
    
    async def run(self, duration_seconds=600):
        """
        Run the main audio processing pipeline for the specified duration.
        
        This method orchestrates the entire process:
        1. Initializes all components
        2. Starts audio capture
        3. Processes audio frames in real-time
        4. Provides progress updates
        5. Shows final analysis results
        
        Args:
            duration_seconds: How long to run the session (default: 600 = 10 minutes)
                             TO CUSTOMIZE: Change this value to run for different durations
        """
        # =====================================================================================
        # INITIALIZATION PHASE
        # =====================================================================================
        if not self.initialize():
            logger.error("Cannot start - initialization failed")
            logger.error("Please check the troubleshooting suggestions above and try again")
            return
        
        logger.info(f"Starting {duration_seconds/60:.0f}-minute speaker verification session...")
        logger.info("Speak into your microphone - analysis in progress...")
        logger.info("Press Ctrl+C to stop early if needed")
        self.running = True
        
        # =====================================================================================
        # AUDIO CAPTURE SETUP
        # =====================================================================================
        audio_capture = FrameCapture()  # Initialize real-time audio capture
        audio_queue = audio_capture.start()  # Start capturing audio frames
        
        start_time = time.time()
        last_progress_time = start_time
        
        try:
            # =====================================================================================
            # MAIN PROCESSING LOOP
            # =====================================================================================
            while self.running and (time.time() - start_time) < duration_seconds:
                # Process any available audio frames
                if not audio_queue.empty():
                    audio_bytes = audio_queue.get()  # Get next audio frame
                    
                    # Process through speaker verification and transcription pipeline
                    filtered_audio = self.process_audio_frame(audio_bytes)
                
                # =====================================================================================
                # PROGRESS REPORTING (every 2 minutes)
                # =====================================================================================
                # TO CUSTOMIZE: Change 120 to different interval (in seconds) for progress updates
                current_time = time.time()
                if current_time - last_progress_time >= 120:  # Every 2 minutes
                    elapsed_minutes = int((current_time - start_time) / 60)
                    total_minutes = int(duration_seconds / 60)
                    verification_rate = (self.verified_frames / self.processed_frames * 100) if self.processed_frames > 0 else 0
                    logger.info(f"Progress: {elapsed_minutes}/{total_minutes} minutes | "
                              f"Verified: {self.verified_frames}/{self.processed_frames} ({verification_rate:.1f}%) | "
                              f"Transcribed: {self.transcribed_segments} segments")
                    last_progress_time = current_time
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.001)
                
        except KeyboardInterrupt:
            logger.info("Session stopped by user (Ctrl+C pressed)")
        finally:
            # =====================================================================================
            # CLEANUP AND FINAL ANALYSIS
            # =====================================================================================
            self.running = False          # Stop the processing loop
            audio_capture.stop()          # Stop audio capture
            
            # Display comprehensive results
            self._show_final_analysis(time.time() - start_time)

# =====================================================================================
# MAIN PROGRAM ENTRY POINT
# =====================================================================================

async def main():
    """
    Main entry point for the Speech Isolation Pipeline.
    
    This function:
    1. Checks for required setup (reference audio directory)
    2. Creates and configures the pipeline
    3. Runs the analysis session
    
    TO CUSTOMIZE: Modify the session duration or directory path here
    """
    logger.info("Simple Speech Isolation Pipeline Starting...")
    
    # =====================================================================================
    # SETUP VERIFICATION
    # =====================================================================================
    # TO CUSTOMIZE: Change this path if you want to store reference files elsewhere
    captured_audio_dir = "captured_audio"
    
    if not os.path.exists(captured_audio_dir):
        logger.error(f"Directory not found: {captured_audio_dir}")
        logger.error("SETUP REQUIRED:")
        logger.error("1. Create a 'captured_audio' directory in the project root")
        logger.error("2. Add .wav files containing reference samples of your target speaker")
        logger.error("3. Run this script again")
        return
    
    # =====================================================================================
    # PIPELINE EXECUTION
    # =====================================================================================
    pipeline = SimplePipeline(captured_audio_dir)
    
    # TO CUSTOMIZE: Change 600 to run for different durations
    # 300 = 5 minutes, 600 = 10 minutes, 1200 = 20 minutes, etc.
    await pipeline.run(600)  # Run for 10 minutes (600 seconds)

# =====================================================================================
# SCRIPT EXECUTION - This runs when the file is executed directly
# =====================================================================================

if __name__ == "__main__":
    """
    Script execution entry point.
    
    This section handles:
    - Running the main pipeline
    - Graceful handling of user interruption (Ctrl+C)
    - Error reporting and cleanup
    
    TO RUN: Execute this script with: python simple_pipeline.py
    """
    try:
        # Start the async pipeline
        asyncio.run(main())
        
    except KeyboardInterrupt:
        # User pressed Ctrl+C - this is normal and expected
        logger.info("Pipeline terminated by user (Ctrl+C)")
        logger.info("Session ended early - partial results may be available above")
        
    except Exception as e:
        # Unexpected error occurred - log details and exit
        logger.error(f"Pipeline failed with error: {e}")
        logger.error("Please check the error message above and verify your setup")
        sys.exit(1)  # Exit with error code