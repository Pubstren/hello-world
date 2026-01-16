"""Audio capture utilities with optional voice activity detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional dependency
    sd = None

from .config import AudioConfig


@dataclass
class CaptureResult:
    audio: np.ndarray
    sample_rate: int


def _require_sounddevice() -> None:
    if sd is None:
        raise RuntimeError("sounddevice is required for audio capture")


def record_audio(duration_s: float, config: AudioConfig) -> CaptureResult:
    _require_sounddevice()
    audio = sd.rec(
        int(duration_s * config.sample_rate),
        samplerate=config.sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return CaptureResult(audio=audio.squeeze(axis=-1), sample_rate=config.sample_rate)


def simple_vad(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    if audio.size == 0:
        return audio
    mask = np.abs(audio) > threshold
    if not mask.any():
        return audio
    indices = np.where(mask)[0]
    return audio[indices[0] : indices[-1] + 1]


def capture_with_vad(
    duration_s: float, config: AudioConfig, vad_threshold: Optional[float] = 0.01
) -> CaptureResult:
    result = record_audio(duration_s, config)
    if vad_threshold is None:
        return result
    trimmed = simple_vad(result.audio, threshold=vad_threshold)
    return CaptureResult(audio=trimmed, sample_rate=result.sample_rate)
