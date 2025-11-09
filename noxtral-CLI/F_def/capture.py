import sounddevice as sd
from queue import Queue
import time
import numpy as np
import logging

# Set up logger
logger = logging.getLogger(__name__)

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
            logger.warning(f"Audio callback status: {status}")
            
        audio_int16 = np.clip(indata, -1.0, 1.0) * 32767
        audio_int16 = audio_int16.astype(np.int16)

        audio_bytes = audio_int16.tobytes()
        audio_q.put(audio_bytes)

    def start(self):
        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
            callback=self.audio_callback,
            dtype=np.float32  
        )
        self.stream.start()
        logger.info("Frame capture started.")
        return audio_q

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logger.info("Frame capture stopped.")
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
        # Use larger blocks for better speaker recognition (4096 samples = ~256ms)
        super().__init__(samplerate=16000, channels=1, blocksize=4096)
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
    
    
class AudioFrame:
    """
    Represents a small chunk of audio data in a streaming pipeline.
    """

    def __init__(self, data: np.ndarray, sample_rate: int, channels: int = 1, timestamp: float = None):
        assert data.ndim in (1, 2), "AudioFrame data must be mono (1D) or multi-channel (2D)"
        self.data = data.astype(np.float32)  
        self.sample_rate = sample_rate
        self.channels = channels
        self.timestamp = timestamp or time.time()

    @property
    def duration(self) -> float:
        """Duration of this frame in seconds."""
        return len(self.data) / self.sample_rate

    def to_bytes(self) -> bytes:
        """Convert float32 data to bytes for streaming or socket transmission."""
        return self.data.tobytes()

    def __repr__(self):
        return (f"AudioFrame(samples={len(self.data)}, "
                f"rate={self.sample_rate}, "
                f"ch={self.channels}, "
                f"ts={self.timestamp:.3f})")


class AudioRawFrame:
    """
    Low-level raw audio frame from capture source.
    Usually bytes (e.g., PCM16), before any normalization or decoding.
    """
    def __init__(self, data: bytes, sample_rate: int, channels: int = 1, timestamp: float = None):
        self.data = data
        self.sample_rate = sample_rate
        self.channels = channels
        self.timestamp = timestamp or time.time()

    def to_float32(self) -> "AudioFrame":
        """Convert raw PCM16 bytes → normalized float32 array"""
        arr = np.frombuffer(self.data, dtype=np.int16).astype(np.float32) / 32768.0
        return AudioFrame(arr, self.sample_rate, self.channels, self.timestamp)

    def __repr__(self):
        return f"AudioRawFrame(len={len(self.data)}, rate={self.sample_rate}, ch={self.channels})"
