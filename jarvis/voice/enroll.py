"""Interactive local enrolment for JARVIS speaker verification."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jarvis.voice.siamese import OwnerVoiceVerifier, SAMPLE_RATE, VoiceEnrollmentError


def _record(samples: int, seconds: float) -> list[np.ndarray]:
    try:
        import sounddevice as sd
    except ImportError as e:
        raise VoiceEnrollmentError("enrolment needs sounddevice; install the project requirements") from e
    clips: list[np.ndarray] = []
    prompts = (
        "Say: Hey Jarvis, what is on my calendar today?",
        "Say: Hey Jarvis, play my focus playlist.",
        "Say: Hey Jarvis, open my study notes.",
    )
    for i in range(samples):
        input(f"\nSample {i + 1}/{samples}. {prompts[i % len(prompts)]}\nPress Enter, then speak: ")
        print("Recording…")
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        clips.append(np.asarray(audio[:, 0], dtype=np.float32))
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll the local JARVIS owner-voice gate")
    parser.add_argument("--samples", type=int, default=10, help="clear recordings to collect (minimum 6)")
    parser.add_argument("--seconds", type=float, default=3.5, help="seconds per recording")
    parser.add_argument("--output", default="~/.jarvis/voice/owner_siamese.pt", help="profile output path")
    args = parser.parse_args()
    if args.samples < 6 or args.seconds < 1:
        parser.error("use at least 6 samples of at least 1 second each")
    print("JARVIS will keep only a local encrypted-by-permissions voice profile; raw recordings are discarded.")
    verifier = OwnerVoiceVerifier.enroll(_record(args.samples, args.seconds))
    output = str(Path(args.output).expanduser())
    verifier.save(output)
    print(f"\nEnrolled. Profile saved locally to {output}")
    print("Set JARVIS_VOICE_VERIFY=1 in .env, then restart JARVIS.")


if __name__ == "__main__":
    main()
