import sounddevice as sd
import numpy as np
import wave
import os
from datetime import datetime

def capture_audio(duration=5, sample_rate=16000, channels=1, output_dir=r"C:\Users\Admin\Desktop\noxtral\Noxtral\captured_audio"):
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_audio_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    
    print(f"Recording for {duration} seconds...")
    
    audio_data = sd.rec(int(duration * sample_rate),samplerate=sample_rate,channels=channels,dtype=np.int16)
    sd.wait() 
    
    print("Recording finished!")
    
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2) 
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    print(f"Audio saved to: {filepath}")
    return filepath

if __name__ == "__main__":
    capture_audio(duration=5)
