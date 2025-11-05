import sounddevice as sd
from queue import Queue
import time
import numpy as np

audio_q = Queue()

class Frames:
    def __init__(self, samplerate, channels, blocksize, device=None):
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.stream = None

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert float32 audio data to int16 properly
        # Clip values to [-1.0, 1.0] range and scale to int16
        audio_int16 = np.clip(indata, -1.0, 1.0) * 32767
        audio_int16 = audio_int16.astype(np.int16)
        
        # Convert to bytes
        audio_bytes = audio_int16.tobytes()
        audio_q.put(audio_bytes)

    def start(self):
        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
            callback=self.audio_callback,
            dtype=np.float32  # Explicitly specify dtype
        )
        self.stream.start()
        print("Frame capture started.")
        return audio_q

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("Frame capture stopped.")
        return audio_q
    
    def get_current_frame(self):
        """Get the current frame without removing it from queue"""
        if not audio_q.empty():
            temp_frames = []
            current_frame = None
            
            while not audio_q.empty():
                current_frame = audio_q.get()
                temp_frames.append(current_frame)
            for frame in temp_frames:
                audio_q.put(frame)
                
            return current_frame
        return None

class FrameCapture(Frames):
    def __init__(self):
        super().__init__(samplerate=16000, channels=1, blocksize=1024)
        self.time_span = 0
        self.frame_size = 0
        
    def Run(self, time_span, frame_size):
        self.time_span = time_span
        self.frame_size = frame_size
        
        start_time = time.time()
        q = self.start()
        
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.time_span:
                break
            time.sleep(self.frame_size)
            current_queue_size = q.qsize()
            frame = self.get_current_frame()
            # if frame:
            #     print(f"[{elapsed_time:.1f}s] Frames in queue: {current_queue_size}, Current frame: {len(frame)} bytes")
            #     print()
            # else:
            #     print(f"[{elapsed_time:.1f}s] Frames in queue: {current_queue_size}, No current frame available")
        
        self.stop()
        return q.qsize(),frame,current_queue_size