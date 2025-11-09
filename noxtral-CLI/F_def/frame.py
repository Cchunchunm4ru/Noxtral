class FrameProcessor:
    def __init__(self):
        self.next = None  # next processor in pipeline

    def process(self, frame):
        """Process a frame and return transformed frames."""
        raise NotImplementedError

    def send(self, frame):
        """Send processed frame to next stage."""
        if self.next:
            self.next.process(frame)

class Frame:
    def __init__(self, type: str, data=None, timestamp=None):
        self.type = type
        self.data = data
        self.timestamp = timestamp