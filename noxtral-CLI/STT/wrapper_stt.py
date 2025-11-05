from abc import ABC, abstractmethod

class BaseSTT(ABC):
    @abstractmethod
    def transcribe_chunk(self, audio_chunk: bytes) -> str:
        pass

    @abstractmethod
    def finalize(self) -> str:
        pass
