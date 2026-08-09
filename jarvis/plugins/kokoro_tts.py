"""Local TTS plugin for LiveKit Agents, backed by Kokoro (ONNX).

Kokoro-82M is Apache-2.0, runs on CPU, and produces natural 24 kHz speech.
Synthesis is non-streaming (one utterance -> one buffer), pushed to LiveKit's
AudioEmitter.

Model files (download once, see README):
  models/kokoro-v1.0.onnx
  models/voices-v1.0.bin
"""
from __future__ import annotations

import numpy as np
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

_KOKORO_SR = 24000


class KokoroTTS(tts.TTS):
    def __init__(
        self,
        *,
        model_path: str,
        voices_path: str,
        voice: str = "bm_george",
        speed: float = 1.0,
        lang: str = "en-gb",
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=_KOKORO_SR,
            num_channels=1,
        )
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(model_path, voices_path)
        self._voice = voice
        self._speed = speed
        self._lang = lang

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "KokoroChunkedStream":
        return KokoroChunkedStream(
            tts=self, input_text=text, conn_options=conn_options
        )


class KokoroChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: KokoroTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: KokoroTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        samples, sample_rate = self._tts._kokoro.create(
            self._input_text,
            voice=self._tts._voice,
            speed=self._tts._speed,
            lang=self._tts._lang,
        )
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

        output_emitter.initialize(
            request_id=utils_shortuuid(),
            sample_rate=sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm16)
        output_emitter.flush()


def utils_shortuuid() -> str:
    from livekit.agents import utils

    return utils.shortuuid()
