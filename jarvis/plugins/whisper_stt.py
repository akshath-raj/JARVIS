"""Local STT plugin for LiveKit Agents.

Two backends, selected by `backend`:
  * "mlx"    -> mlx-whisper, Metal-accelerated on Apple Silicon (lowest latency).
  * "faster" -> faster-whisper (CTranslate2), CPU int8, portable fallback.

Both are non-streaming, so this advertises `streaming=False`; combined with a VAD
on the AgentSession, LiveKit buffers each utterance and calls `_recognize_impl`
once per turn.
"""
from __future__ import annotations

import logging
import re
import zlib

import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger("jarvis.stt")

_WHISPER_SR = 16000

# Bias the decoder toward the assistant's name so it stops mishearing "Jarvis" as
# "Jairus"/"Jarvus"/etc. Whisper uses the previous-text prompt to condition
# spelling; naming Jarvis here makes the correct token far likelier. Kept short and
# neutral so it doesn't hallucinate the word into silence.
_NAME_BIAS_PROMPT = "Hey Jarvis. The assistant's name is Jarvis."

# HF repos for the Metal (mlx) models, keyed by the faster-whisper-style name.
_MLX_REPOS = {
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _dehallucinate(text: str) -> str:
    """Drop a transcript that is a repetition loop (e.g. 'Inodorodorodoro…').
    Whisper occasionally loops on ambiguous audio; such text compresses far more
    than natural speech, so a high compression ratio is a reliable tell."""
    t = text.strip()
    if len(t) < 40:
        return t
    compact = re.sub(r"\s+", " ", t)
    ratio = len(compact) / max(1, len(zlib.compress(compact.encode("utf-8"))))
    if ratio > 3.2:
        logger.info("dropped hallucinated transcript (%d chars, ratio %.1f)", len(t), ratio)
        return ""
    return t


class WhisperSTT(stt.STT):
    def __init__(self, *, model: str = "base.en", language: str = "en", backend: str = "mlx"):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._language = language
        self._backend = backend
        self._faster = None
        self._mlx_repo = None

        if backend == "mlx":
            try:
                import mlx_whisper  # noqa: F401  (import proves availability)

                self._mlx_repo = _MLX_REPOS.get(model, model)
                logger.info("STT: mlx-whisper %s (Metal)", self._mlx_repo)
                return
            except Exception as e:
                logger.warning("mlx-whisper unavailable (%s); using faster-whisper.", e)
                self._backend = "faster"

        from faster_whisper import WhisperModel

        self._faster = WhisperModel(model, device="cpu", compute_type="int8")
        logger.info("STT: faster-whisper %s (CPU int8)", model)

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language=None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = language if isinstance(language, str) else self._language
        frame = rtc.combine_audio_frames(buffer)
        samples = self._to_mono_16k_float32(frame)

        if self._backend == "mlx":
            import mlx_whisper

            result = mlx_whisper.transcribe(
                samples,
                path_or_hf_repo=self._mlx_repo,
                language=lang,
                fp16=True,
                initial_prompt=_NAME_BIAS_PROMPT,
                # anti-hallucination: don't let the model loop on its own output
                # ("Inodorodoro…"), and drop low-confidence / silent / repetitive segments.
                condition_on_previous_text=False,
                temperature=0.0,
                compression_ratio_threshold=2.2,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                hallucination_silence_threshold=2.0,
            )
            text = (result.get("text") or "").strip()
        else:
            segments, _ = self._faster.transcribe(
                samples,
                language=lang,
                beam_size=1,
                vad_filter=True,  # drop non-speech that seeds hallucinations
                initial_prompt=_NAME_BIAS_PROMPT,
                hotwords="Jarvis",
                condition_on_previous_text=False,
                temperature=0.0,
                compression_ratio_threshold=2.2,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                no_repeat_ngram_size=3,  # hard-stops the repetition loop
            )
            text = " ".join(seg.text for seg in segments).strip()

        text = _dehallucinate(text)
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
