from capture import Frames, FrameCapture, audio_q
from STT import WhisperSTT
from events.EventManager import EventManager

def handle_stt_activation(transcription=None, confidence=None, audio_duration=None):
    """Handle STT activation events instantly"""
    print(f"🎤 STT Activated! Transcription: {transcription}")
    if confidence:
        print(f"📊 Confidence: {confidence:.2f}")
    if audio_duration:
        print(f"⏱️  Duration: {audio_duration:.2f}s")

def main():
    fc = FrameCapture()
    stt = WhisperSTT()
    
    # Get the event manager from the STT module and subscribe to events
    from STT.whisper_stt import event_manager
    event_manager.subscribe_to_stt_activates(handle_stt_activation)
    
    # Start audio capture
    import time
    start_time = time.time()
    fc.start()
    
    # Process audio chunks in real-time as they come in
    chunk_buffer = b""
    chunk_size = 32000  # Process every ~1 second of audio (16000 samples/sec * 2 bytes)
    frame_count = 0
    capture_duration = 5  # seconds
    
    print("🎙️ Real-time audio processing started...")
    
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= capture_duration:
            break
            
        # Check if there are new audio frames
        if not audio_q.empty():
            frame = audio_q.get()
            chunk_buffer += frame
            frame_count += 1
            
            # Process chunk when we have enough audio data
            if len(chunk_buffer) >= chunk_size:
                print(f"\nProcessed chunk in real-time...")
                result = stt.transcribe_chunk(chunk_buffer)
                chunk_buffer = b""  # Reset buffer for next chunk
        
        time.sleep(0.01)  # Small delay to prevent CPU overload
    
    fc.stop()
    
    # Process any remaining audio data
    if len(chunk_buffer) > 0:
        print(f"\n🔄 Processing final chunk...")
        result = stt.transcribe_chunk(chunk_buffer)
    
    print(f"\n📊 Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()