"""Local, opt-in speaker verification for JARVIS."""

from jarvis.voice.siamese import OwnerVoiceVerifier, VoiceEnrollmentError

__all__ = ["OwnerVoiceVerifier", "VoiceEnrollmentError"]
