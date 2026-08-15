"""Tests for the local owner-voice speaker-verification gate.

These exercise the security-critical behaviour of ``OwnerVoiceVerifier``:
enrolment calibration, accept/reject decisions, short-clip rejection, and the
save/load round-trip. They need torch + numpy, so the whole module is skipped
where those are unavailable (e.g. a minimal CI image).
"""
from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("torch")
import numpy as np  # noqa: E402

from jarvis.voice.siamese import (  # noqa: E402
    SAMPLE_RATE,
    OwnerVoiceVerifier,
    VoiceEnrollmentError,
)

_RNG = np.random.default_rng(0)


def _owner_clip(seconds: float = 2.0) -> np.ndarray:
    """A synthetic but internally consistent 'voice' timbre (fixed harmonics)."""
    t = np.linspace(0.0, seconds, int(seconds * SAMPLE_RATE), dtype=np.float32)
    sig = (
        0.30 * np.sin(2 * np.pi * 140 * t)
        + 0.20 * np.sin(2 * np.pi * 280 * t)
        + 0.10 * np.sin(2 * np.pi * 420 * t)
    )
    sig += _RNG.normal(0.0, 0.01, len(t)).astype(np.float32)
    return sig.astype(np.float32)


@pytest.fixture(scope="module")
def verifier() -> OwnerVoiceVerifier:
    # Few epochs keeps the test fast; correctness of the loss/threshold logic does
    # not depend on long training.
    return OwnerVoiceVerifier.enroll([_owner_clip() for _ in range(8)], epochs=8)


def test_enroll_calibrates_a_conservative_threshold(verifier):
    # Calibration is clamped to [0.72, 0.94]; it must never fall open.
    assert 0.72 <= verifier.threshold <= 0.94


def test_accepts_the_enrolled_owner(verifier):
    accepted, score = verifier.verify(_owner_clip())
    assert accepted is True
    assert score >= verifier.threshold


def test_rejects_dissimilar_audio(verifier):
    noise = _RNG.normal(0.0, 0.3, int(2.0 * SAMPLE_RATE)).astype(np.float32)
    accepted, score = verifier.verify(noise)
    assert accepted is False
    assert score < verifier.threshold


def test_rejects_too_short_audio(verifier):
    # Under one second cannot establish identity and must be refused outright,
    # without even scoring it.
    accepted, score = verifier.verify(_owner_clip(seconds=0.5))
    assert accepted is False
    assert score == -1.0


def test_enrollment_rejects_too_few_samples():
    with pytest.raises(VoiceEnrollmentError):
        OwnerVoiceVerifier.enroll([_owner_clip() for _ in range(3)])


def test_enrollment_rejects_too_quiet_samples():
    quiet = [np.zeros(int(2.0 * SAMPLE_RATE), dtype=np.float32) for _ in range(8)]
    with pytest.raises(VoiceEnrollmentError):
        OwnerVoiceVerifier.enroll(quiet)


def test_save_load_round_trip_preserves_decisions(verifier):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "profile", "owner_siamese.pt")
        verifier.save(path)

        # Profile and its directory are written with owner-only permissions.
        assert oct(os.stat(path).st_mode)[-3:] == "600"
        assert oct(os.stat(os.path.dirname(path)).st_mode)[-3:] == "700"

        reloaded = OwnerVoiceVerifier.load(path)
        assert reloaded.threshold == pytest.approx(verifier.threshold)
        assert reloaded.verify(_owner_clip())[0] is True


def test_load_missing_profile_fails_closed():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(VoiceEnrollmentError) as excinfo:
            OwnerVoiceVerifier.load(os.path.join(d, "nope.pt"))
    assert "jarvis.voice.enroll" in str(excinfo.value)


def test_load_rejects_corrupt_profile():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "corrupt.pt")
        with open(path, "wb") as fh:
            fh.write(b"not a torch checkpoint")
        with pytest.raises(VoiceEnrollmentError):
            OwnerVoiceVerifier.load(path)
