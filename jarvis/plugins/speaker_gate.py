"""STT wrapper that drops utterances not matching the enrolled local owner voice."""
from __future__ import annotations

import logging
import re

import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from jarvis.voice.siamese import OwnerVoiceVerifier
from jarvis.activation import VOICE_NOT_RECOGNIZED

logger = logging.getLogger("jarvis.voice_gate")
_WAKE_PHRASE = re.compile(r"^\s*hey[\s,.:;!-]+jarvis\b", re.I)


class SpeakerVerifiedSTT(stt.STT):
    """Turn any batch-capable STT into a fail-closed owner-voice gate.

    It deliberately exposes non-streaming capabilities so LiveKit's VAD buffers a
    complete utterance and checks its speaker locally. Approved speech is sent to
    the underlying recognizer unchanged. For a rejected speaker, JARVIS remains
    silent unless their *transcript starts with* "Hey Jarvis"; in that case the
    wrapper emits a private control signal that receives the fixed reply "Voice not
    recognized." No untrusted transcript ever reaches wake-word or tool routing.
    """

    def __init__(self, inner: stt.STT, verifier: OwnerVoiceVerifier):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._inner = inner
        self._verifier = verifier

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def provider(self) -> str:
        return self._inner.provider

    async def _recognize_impl(
        self, buffer: utils.AudioBuffer, *, language=None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            data = data.reshape(-1, frame.num_channels).mean(axis=1)
        accepted, score = self._verifier.verify(data, frame.sample_rate)
        if accepted:
            return await self._inner.recognize(buffer, language=language, conn_options=conn_options)

        # We need the transcript only to distinguish an explicit attempted wake-up
        # from ordinary roommate/background speech. The result is discarded either
        # way; it is never forwarded as a user command.
        event = await self._inner.recognize(buffer, language=language, conn_options=conn_options)
        text = event.alternatives[0].text if event.alternatives else ""
        logger.info("Ignoring unrecognized speaker (score %.3f; threshold %.3f)", score, self._verifier.threshold)
        if _WAKE_PHRASE.match(text):
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language=language or "en", text=VOICE_NOT_RECOGNIZED)],
            )
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[])

    async def aclose(self) -> None:
        await self._inner.aclose()
