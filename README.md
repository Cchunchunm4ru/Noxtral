# Noxtral 🎙️

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/Open%20Source-❤️-red)](https://github.com/Cchunchunm4ru/Noxtral)

**Noxtral** is a real-time speech-to-text (STT) processing system with advanced speaker isolation and verification capabilities. It captures audio in real-time, transcribes it using OpenAI's Whisper model, and provides speaker identification features with an event-driven architecture.

## ✨ Features

- 🎯 **Real-time Audio Capture**: Continuous audio recording with configurable sample rates
- 🚀 **Fast Speech Recognition**: Uses Faster-Whisper for efficient transcription
- � **Speaker Isolation & Verification**: Advanced speaker identification and voice isolation capabilities
- 🔊 **Robust Audio Processing**: Enhanced audio processing with noise reduction and voice activity detection
- �📡 **Event-Driven Architecture**: Modular design with publish-subscribe pattern
- 🔄 **Streaming Processing**: Process audio chunks as they arrive
- 🎛️ **Configurable Settings**: Adjustable capture duration, chunk sizes, and audio parameters
- 🖥️ **Cross-Platform**: Works on Windows, macOS, and Linux

## 🏗️ Architecture

```
noxtral/
├── noxtral-CLI/
│   ├── simple_pipeline.py   # Main application pipeline
│   ├── F_def/               # Frame and capture definitions
│   │   ├── capture.py       # Audio capture implementation
│   │   └── frame.py         # Frame management utilities
│   ├── STT/                 # Speech-to-Text module
│   │   ├── __init__.py
│   │   ├── whisper_stt.py   # Whisper STT implementation
│   │   ├── wrapper_stt.py   # STT base interface
│   │   └── frame.py         # STT frame processing
│   ├── speaker_isolation/   # Speaker identification and isolation
│   │   ├── robust_realtime_verifier.py  # Real-time speaker verification
│   │   └── speaker_system.py # Speaker isolation system
│   └── events/
│       └── EventManager.py  # Event management system
├── pretrained_models/       # Pre-trained model storage
│   └── spkrec-ecapa-voxceleb/  # ECAPA-TDNN speaker recognition models
├── captured_audio/          # Stored audio recordings
├── enrollments/             # Speaker enrollment data
└── requirements.txt         # Project dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- A microphone for audio input

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Cchunchunm4ru/Noxtral.git
   cd Noxtral
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install sounddevice numpy faster-whisper nltk speechbrain torch librosa scipy scikit-learn webrtcvad soundfile
   ```

3.**create an audio dir**
   in this format 
   ```dir
       |___audio1.wav
       |___audio2.wav
   ```
4. **change the Path after recording speech instances**
   in speaker_isolation/speaker_system 
   change the path ```RECORDINGS_REF_PATH= your audio dir path```

5. **Run the application**
   ```bash
   cd noxtral-CLI
   python simple_pipeline.py
   ```

## 📋 Dependencies

- **sounddevice**: Real-time audio I/O
- **numpy**: Numerical operations for audio processing  
- **faster-whisper**: Optimized Whisper implementation
- **nltk**: Natural language processing utilities
- **speechbrain**: Advanced speech and audio processing
- **torch**: PyTorch for deep learning models
- **librosa**: Audio analysis and feature extraction
- **scipy**: Scientific computing for signal processing
- **scikit-learn**: Machine learning utilities
- **webrtcvad**: Voice activity detection
- **soundfile**: Audio file I/O operations

## 🔧 Configuration

### Audio Settings

You can modify audio capture settings in `F_def/capture.py`:

```python
class FrameCapture(Frames):
    def __init__(self):
        super().__init__(
            samplerate=16000,    # Sample rate (Hz)
            channels=1,          # Mono audio
            blocksize=1024       # Buffer size
        )
```

### Transcription Settings

Modify Whisper model settings in `STT/whisper_stt.py`:

```python
class WhisperSTT(BaseSTT):
    def __init__(self, model_name="base", sample_rate=16000):
        # Available models: tiny, base, small, medium, large
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
```

### Speaker Isolation Settings

Configure speaker identification in `speaker_isolation/speaker_system.py`:

```python
class SpeakerSystem:
    def __init__(self):
        # Speaker recognition using SpeechBrain ECAPA-TDNN
        # Configurable thresholds and verification parameters
        self.verification_threshold = 0.25
        self.embedding_model = "speechbrain/spkrec-ecapa-voxceleb"
```

## 🎯 Usage Examples

### Basic Usage

```python
from F_def.capture import FrameCapture
from STT import WhisperSTT
from speaker_isolation.speaker_system import SpeakerSystem

# Initialize components
fc = FrameCapture()
stt = WhisperSTT()
speaker_system = SpeakerSystem()

# Start audio capture
fc.start()

# Process audio in real-time with speaker identification
# (See simple_pipeline.py for complete implementation)
```

### Event Handling

```python
def handle_transcription(transcription=None, confidence=None, audio_duration=None):
    print(f"Transcribed: {transcription}")

# Subscribe to transcription events
from STT.whisper_stt import event_manager
event_manager.subscribe_to_stt_activates(handle_transcription)
```

### Speaker Enrollment & Verification

```python
from speaker_isolation.robust_realtime_verifier import RobustRealtimeVerifier

# Initialize speaker verification system
verifier = RobustRealtimeVerifier()

# Enroll a new speaker
speaker_id = verifier.enroll_speaker(audio_samples, speaker_name="John")

# Verify speaker during real-time processing
is_verified, confidence = verifier.verify_speaker(audio_chunk, speaker_id)
```

## 🔄 How It Works

1. **Audio Capture**: The `FrameCapture` class continuously records audio from the microphone
2. **Speaker Processing**: Advanced speaker isolation and verification using SpeechBrain models
3. **Voice Activity Detection**: WebRTC VAD ensures only speech segments are processed
4. **Chunking**: Audio is processed in configurable chunks for real-time performance
5. **Transcription**: Each audio chunk is transcribed using the Faster-Whisper model
6. **Event Publishing**: Transcribed words and speaker information trigger events through the EventManager
7. **Real-time Output**: Transcriptions with speaker identification are displayed instantly

## 📊 Performance

- **STT Latency**: ~1-2 seconds from speech to transcription (depending on chunk size)
- **Speaker Verification**: Real-time speaker identification with <500ms latency
- **Accuracy**: Depends on Whisper model size (base model recommended for real-time use)
- **Speaker Recognition**: High accuracy using ECAPA-TDNN embeddings with cosine similarity
- **Resource Usage**: Optimized for CPU processing with int8 quantization and efficient neural networks

## 🛠️ Development

### Project Structure

- **`simple_pipeline.py`**: Main application orchestrating all components
- **`F_def/capture.py`**: Audio input handling and frame management
- **`F_def/frame.py`**: Frame processing utilities
- **`STT/whisper_stt.py`**: Whisper integration and transcription logic
- **`speaker_isolation/speaker_system.py`**: Speaker identification and isolation
- **`speaker_isolation/robust_realtime_verifier.py`**: Real-time speaker verification
- **`events/EventManager.py`**: Event system for loose coupling between components

### Adding New STT Engines

Extend the `BaseSTT` class in `STT/wrapper_stt.py`:

```python
class CustomSTT(BaseSTT):
    def transcribe_chunk(self, audio_chunk: bytes) -> str:
        # Your implementation here
        pass
    
    def finalize(self) -> str:
        # Finalization logic
        pass
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the incredible speech recognition model
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for the optimized implementation
- The open-source community for inspiration and support

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Cchunchunm4ru/Noxtral/issues) page
2. Create a new issue if your problem isn't already reported
3. Provide detailed information about your environment and the issue

---


**Made with ❤️ by [Cchunchunm4ru](https://github.com/Cchunchunm4ru)**

