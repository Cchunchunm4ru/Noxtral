class EventManager:
    """The central hub for managing event subscriptions, tailored for STT events."""
    def __init__(self):
        # Dictionary: {event_name: [list_of_callback_functions]}
        self.subscribers = {}

    def subscribe(self, event_name, callback_function):
        """Register a function (subscriber) for a specific event."""
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        self.subscribers[event_name].append(callback_function)
        print(f"✅ STT Subscribed: {callback_function.__name__} is now listening for '{event_name}'")

    def publish(self, event_name, **data):
        """Announce an event, triggering all registered subscribers."""
        if event_name in self.subscribers:
            for callback in self.subscribers[event_name]:
                # Call the subscriber function, passing the event data
                callback(**data)
        else:
            print(f"⚠️ Warning: No STT subscribers for event: {event_name}")

    def publish_stt_activates(self, transcription=None, confidence=None, audio_duration=None):
        """Convenience method to publish STT activation event."""
        self.publish("stt_activates", 
                    transcription=transcription)

    def subscribe_to_stt_activates(self, callback_function):
        """Convenience method to subscribe to STT activation events."""
        self.subscribe("stt_activates", callback_function)