"""Local STT plugin for LiveKit Agents, backed by faster-whisper.

faster-whisper is non-streaming, so this STT advertises `streaming=False`.
Combined with a VAD on the AgentSession, LiveKit buffers each utterance and
calls `_recognize_impl` once per turn (the StreamAdapter pattern).

On Apple Silicon this runs on CPU (int8). For a snappier turn use base.en /
small.en; upgrade path is whisper.cpp (Metal) or mlx-whisper.
"""
from __future__ import annotations

import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

_WHISPER_SR = 16000


class WhisperSTT(stt.STT):
    def __init__(self, *, model: str = "base.en", language: str = "en"):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        from faster_whisper import WhisperModel

        # int8 on CPU is the fast, low-memory option on Apple Silicon.
        self._model = WhisperModel(model, device="cpu", compute_type="int8")
        self._language = language

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = language if isinstance(language, str) else self._language
        frame = rtc.combine_audio_frames(buffer)
        samples = self._to_mono_16k_float32(frame)

        segments, _ = self._model.transcribe(
            samples,
            language=lang,
            beam_size=1,          # greedy = lowest latency
            vad_filter=False,     # VAD already handled upstream
        )
        text = " ".join(seg.text for seg in segments).strip()

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=lang, text=text)],
        )

    @staticmethod
    def _to_mono_16k_float32(frame: rtc.AudioFrame) -> np.ndarray:
        data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            data = data.reshape(-1, frame.num_channels).mean(axis=1)
        if frame.sample_rate != _WHISPER_SR:
            n_out = int(round(len(data) * _WHISPER_SR / frame.sample_rate))
            if n_out > 0:
                data = np.interp(
                    np.linspace(0.0, len(data), n_out, endpoint=False),
                    np.arange(len(data)),
                    data,
                ).astype(np.float32)
        return data
