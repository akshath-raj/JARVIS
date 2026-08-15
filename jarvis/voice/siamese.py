"""Local one-owner speaker verification using a compact Siamese network.

The network maps log-mel speech segments into a normalised embedding space. During
enrolment it learns to keep varied recordings of the owner close together while
pushing non-speech and heavily transformed negatives away. At runtime an utterance
is accepted only when its embedding is sufficiently similar to the enrolled voice
prototype. No audio or biometric embedding is sent to a service.

This is a *speaker-verification* gate, not voice identification: it answers only
"does this sound like the enrolled owner?" and rejects uncertain or short audio.
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
N_MELS = 40
EMBEDDING_SIZE = 128
_EPS = 1e-6


class VoiceEnrollmentError(RuntimeError):
    """Raised when the local owner-voice profile is missing or unusable."""


def _torch():
    try:
        import torch
        import torch.nn.functional as F

        return torch, F
    except ImportError as e:  # clear setup error instead of silently disabling security
        raise VoiceEnrollmentError(
            "speaker verification needs PyTorch; install the project requirements first"
        ) from e


def _resample(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate == SAMPLE_RATE or not len(samples):
        return samples
    count = max(1, round(len(samples) * SAMPLE_RATE / sample_rate))
    return np.interp(
        np.linspace(0, len(samples) - 1, count, dtype=np.float32),
        np.arange(len(samples), dtype=np.float32), samples,
    ).astype(np.float32)


def _mel_filterbank():
    """Create a small Slaney-style mel bank without pulling in librosa."""
    torch, _ = _torch()
    n_fft, bins = 400, 201
    lo, hi = 0.0, 2595.0 * math.log10(1.0 + (SAMPLE_RATE / 2) / 700.0)
    mels = np.linspace(lo, hi, N_MELS + 2)
    hertz = 700.0 * (np.power(10.0, mels / 2595.0) - 1.0)
    points = np.floor((n_fft + 1) * hertz / SAMPLE_RATE).astype(int)
    points = np.clip(points, 0, bins - 1)
    bank = np.zeros((N_MELS, bins), dtype=np.float32)
    for i in range(N_MELS):
        left, center, right = points[i : i + 3]
        if center > left:
            bank[i, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
        if right > center:
            bank[i, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)
    return torch.from_numpy(bank)


def log_mel(samples: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Return a [n_mels, frames] tensor for a mono waveform."""
    torch, F = _torch()
    data = _resample(samples, sample_rate)
    # Very short clips cannot establish a voice identity; pad only so STFT itself
    # remains valid. `verify` separately rejects short recordings.
    data = np.pad(data, (0, max(0, 400 - len(data))))
    wave = torch.from_numpy(data).float()
    spectrum = torch.stft(
        wave, n_fft=400, hop_length=160, win_length=400,
        window=torch.hann_window(400), return_complex=True,
    ).abs().square()
    mel = _mel_filterbank().to(spectrum) @ spectrum
    mel = torch.log(mel + _EPS)
    return (mel - mel.mean(dim=1, keepdim=True)) / (mel.std(dim=1, keepdim=True) + _EPS)


def _augment(samples: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Create a positive-pair view resilient to volume and room noise changes."""
    x = np.asarray(samples, dtype=np.float32).copy()
    x *= float(rng.uniform(0.75, 1.20))
    x += rng.normal(0.0, rng.uniform(0.001, 0.008), len(x)).astype(np.float32)
    shift = int(rng.integers(-SAMPLE_RATE // 8, SAMPLE_RATE // 8 + 1))
    return np.roll(x, shift).clip(-1.0, 1.0)


def _negative(samples: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Non-owner proxy negatives for a one-owner enrolment without recording others."""
    n = len(samples)
    noise = rng.normal(0.0, 0.25, n).astype(np.float32)
    # Reversal removes natural speaker/prosody ordering while retaining comparable
    # spectrum, making it a harder negative than plain silence.
    return (0.6 * noise + 0.4 * samples[::-1]).clip(-1.0, 1.0)


def _encoder_class():
    torch, F = _torch()

    class VoiceEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv1d(N_MELS, 64, kernel_size=5, padding=2),
                torch.nn.BatchNorm1d(64), torch.nn.GELU(),
                torch.nn.Conv1d(64, 96, kernel_size=5, padding=2, stride=2),
                torch.nn.BatchNorm1d(96), torch.nn.GELU(),
                torch.nn.Conv1d(96, 128, kernel_size=3, padding=1, stride=2),
                torch.nn.BatchNorm1d(128), torch.nn.GELU(),
            )
            self.project = torch.nn.Linear(128, EMBEDDING_SIZE)

        def forward(self, features):
            x = self.features(features)
            x = x.mean(dim=-1)
            return F.normalize(self.project(x), dim=-1)

    return VoiceEncoder


@dataclass
class OwnerVoiceVerifier:
    """An enrolled local owner-voice profile backed by a Siamese encoder."""

    encoder: object
    prototype: object
    threshold: float

    @classmethod
    def enroll(
        cls, recordings: list[np.ndarray], *, epochs: int = 35, seed: int = 7
    ) -> "OwnerVoiceVerifier":
        torch, F = _torch()
        recordings = [_resample(x, SAMPLE_RATE) for x in recordings if len(x) >= SAMPLE_RATE]
        if len(recordings) < 6:
            raise VoiceEnrollmentError("record at least six clear one-second voice samples")
        if sum(np.sqrt(np.mean(x * x)) > 0.008 for x in recordings) < 6:
            raise VoiceEnrollmentError("the recordings are too quiet; speak closer to the microphone")

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        encoder = _encoder_class()()
        encoder.train()
        opt = torch.optim.AdamW(encoder.parameters(), lr=2e-3, weight_decay=1e-4)
        margin = 0.8
        for _ in range(epochs):
            order = rng.permutation(len(recordings))
            for index in order:
                owner = recordings[index]
                left = log_mel(owner).unsqueeze(0)
                positive = log_mel(_augment(owner, rng=rng)).unsqueeze(0)
                negative = log_mel(_negative(owner, rng=rng)).unsqueeze(0)
                e_left, e_positive, e_negative = encoder(left), encoder(positive), encoder(negative)
                pos_dist = (e_left - e_positive).square().sum(dim=1)
                neg_dist = (e_left - e_negative).square().sum(dim=1)
                loss = pos_dist.mean() + torch.relu(margin - neg_dist).square().mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                opt.step()

        encoder.eval()
        with torch.no_grad():
            embeds = torch.cat([encoder(log_mel(x).unsqueeze(0)) for x in recordings])
            prototype = F.normalize(embeds.mean(dim=0), dim=0)
            similarities = embeds @ prototype
        # Conservative calibration: permit normal changes in the owner's samples,
        # but never set a permissive threshold from a small enrollment set.
        threshold = max(0.72, min(0.94, float(similarities.min().item()) - 0.025))
        return cls(encoder=encoder, prototype=prototype, threshold=threshold)

    def score(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
        torch, _ = _torch()
        x = _resample(samples, sample_rate)
        with torch.no_grad():
            embedding = self.encoder(log_mel(x).unsqueeze(0))[0]
            return float(torch.dot(embedding, self.prototype).item())

    def verify(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[bool, float]:
        duration = len(samples) / max(1, sample_rate)
        if duration < 1.0:
            return False, -1.0
        score = self.score(samples, sample_rate)
        return score >= self.threshold, score

    def save(self, path: str) -> None:
        torch, _ = _torch()
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(target.parent, 0o700)
        payload = {
            "version": 1,
            "state_dict": self.encoder.state_dict(),
            "prototype": self.prototype.detach().cpu(),
            "threshold": self.threshold,
        }
        # Atomic write prevents a half-written profile from accidentally disabling
        # the assistant after an interrupted enrollment.
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
            temp_path = tmp.name
        try:
            torch.save(payload, temp_path)
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, target)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @classmethod
    def load(cls, path: str) -> "OwnerVoiceVerifier":
        torch, F = _torch()
        source = Path(path).expanduser()
        if not source.is_file():
            raise VoiceEnrollmentError(
                f"no enrolled voice profile at {source}; run python -m jarvis.voice.enroll first"
            )
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
            if payload.get("version") != 1:
                raise ValueError("unsupported voice-profile version")
            encoder = _encoder_class()()
            encoder.load_state_dict(payload["state_dict"])
            encoder.eval()
            prototype = F.normalize(payload["prototype"].float(), dim=0)
            threshold = float(payload["threshold"])
        except Exception as e:  # invalid/corrupt profile must not silently fail open
            raise VoiceEnrollmentError(f"could not load voice profile: {e}") from e
        return cls(encoder=encoder, prototype=prototype, threshold=threshold)
