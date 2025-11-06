class EventManager:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_name, callback_function):
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        self.subscribers[event_name].append(callback_function)
        print(f"✅ STT Subscribed: {callback_function.__name__} is now listening for '{event_name}'")

    def publish(self, event_name, **data):
        if event_name in self.subscribers:
            for callback in self.subscribers[event_name]:
                callback(**data)
        else:
            print(f"⚠️ Warning: No STT subscribers for event: {event_name}")

    def publish_stt_activates(self, transcription=None, confidence=None, audio_duration=None):
        self.publish("stt_activates", 
                    transcription=transcription)

    def subscribe_to_stt_activates(self, callback_function):
        self.subscribe("stt_activates", callback_function)