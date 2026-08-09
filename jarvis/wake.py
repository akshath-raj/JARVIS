"""Acoustic wake-word gate ("Hey Jarvis") using openWakeWord.

Sits in front of STT (via each agent's `stt_node`). While ASLEEP it feeds audio
to openWakeWord and drops it before STT/LLM run — so the expensive pipeline stays
idle until the wake word fires. Once triggered it opens a rolling ACTIVE window
(refreshed by continued speech) so follow-up turns don't each need the wake word.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from livekit import rtc

logger = logging.getLogger("jarvis.wake")

_SR = 16000
_CHUNK = 1280  # openWakeWord expects 80 ms @ 16 kHz
_SPEECH_RMS = 0.01  # frames above this refresh the active window


class WakeGate:
    def __init__(
        self,
        *,
        enabled: bool,
        model: str = "hey_jarvis",
        threshold: float = 0.5,
        active_seconds: float = 12.0,
    ):
        self.enabled = enabled
        self.threshold = threshold
        self.active_seconds = active_seconds
        self._model_name = model
        self._active_until = 0.0
        self._buf = np.zeros(0, dtype=np.float32)
        self._oww = None

        if not enabled:
            return
        try:
            import openwakeword
            from openwakeword.model import Model

            openwakeword.utils.download_models()
            self._oww = Model(wakeword_models=[model], inference_framework="onnx")
            logger.info("Wake word active: '%s' (threshold %.2f)", model, threshold)
        except Exception as e:
            logger.warning("openWakeWord unavailable (%s); wake gate disabled.", e)
            self.enabled = False

    @property
    def is_active(self) -> bool:
        return time.time() < self._active_until

    def accept(self, frame: rtc.AudioFrame) -> bool:
        """Return True if this frame should pass through to STT."""
        if not self.enabled or self._oww is None:
            return True

        mono16 = self._to_mono_16k(frame)
        now = time.time()

        if now < self._active_until:
            # Active: pass through, and keep the window open while speech continues.
            if _rms(mono16) > _SPEECH_RMS:
                self._active_until = now + self.active_seconds
            return True

        # Asleep: run wake detection on 80 ms chunks.
        self._buf = np.concatenate([self._buf, mono16])
        while len(self._buf) >= _CHUNK:
            chunk = (self._buf[:_CHUNK] * 32767.0).astype(np.int16)
            self._buf = self._buf[_CHUNK:]
            scores = self._oww.predict(chunk)
            if scores.get(self._model_name, 0.0) >= self.threshold:
                logger.info("Wake word detected.")
                self._active_until = now + self.active_seconds
                self._buf = np.zeros(0, dtype=np.float32)
                return True
        return False

    @staticmethod
    def _to_mono_16k(frame: rtc.AudioFrame) -> np.ndarray:
        data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            data = data.reshape(-1, frame.num_channels).mean(axis=1)
        if frame.sample_rate != _SR:
            n_out = int(round(len(data) * _SR / frame.sample_rate))
            if n_out > 0:
                data = np.interp(
                    np.linspace(0.0, len(data), n_out, endpoint=False),
                    np.arange(len(data)),
                    data,
                ).astype(np.float32)
        return data


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
