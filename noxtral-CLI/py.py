from capture import Frames, FrameCapture, audio_q
from STT import WhisperSTT
def main():
    fc = FrameCapture()
    stt = WhisperSTT()
    
    # Capture audio frames
    fc.Run(time_span=5, frame_size=0.20)
    
    # Collect all captured audio data
    audio_data = b""
    frame_count = 0
    while not audio_q.empty():
        frame = audio_q.get()
        audio_data += frame
        frame_count += 1
    
    print(f"Total frames collected: {frame_count}")
    print(f"Total audio data: {len(audio_data)} bytes")
    
    # Calculate duration: bytes / (sample_rate * channels * bytes_per_sample)
    duration = len(audio_data) / (16000 * 1 * 2) 
    print(stt.transcribe_audio)# 16kHz, 1 channel, 2 bytes per sample
    
    if len(audio_data) > 0:
        result = stt.transcribe_chunk(audio_data)
        print("Transcription result:", result)
        
    else:
        print("No audio data captured!")


if __name__ == "__main__":
    main()