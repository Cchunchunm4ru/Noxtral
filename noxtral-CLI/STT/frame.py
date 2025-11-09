from F_def.frame import Frame

class TranscriptionFrame(Frame):
    def __init__(
        self,
        text: str,
        is_final: bool = True,
        confidence: float = 1.0,
        start_time: float = None,
        end_time: float = None,
        language: str = "en"
    ):
        super().__init__(type="transcription", data=text)
        self.text = text
        self.is_final = is_final
        self.confidence = confidence
        self.start_time = start_time
        self.end_time = end_time
        self.language = language

    def __repr__(self):
        return f"<TranscriptionFrame text='{self.text}' final={self.is_final}>"

class InterruptionFrame(Frame):
    def __init__(
        self,
        reason: str,
        source: str = "user",
        interrupted_text: str = "",
        timestamp: float = None,
        priority: int = 1
    ):
        super().__init__(type="interruption", data=reason)
        self.reason = reason
        self.source = source
        self.interrupted_text = interrupted_text
        self.timestamp = timestamp
        self.priority = priority
        self.is_handled = False

    def mark_handled(self):
        """Mark this interruption as handled"""
        self.is_handled = True

    def get_interruption_type(self) -> str:
        """Return a categorized interruption type"""
        if "silence" in self.reason.lower():
            return "SILENCE"
        elif "noise" in self.reason.lower():
            return "NOISE"
        elif "manual" in self.reason.lower() or "user" in self.source.lower():
            return "MANUAL"
        elif "timeout" in self.reason.lower():
            return "TIMEOUT"
        else:
            return "OTHER"

    def __repr__(self):
        return f"<InterruptionFrame reason='{self.reason}' source='{self.source}' handled={self.is_handled}>"


