"""MFCC feature extraction and quantization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import AudioConfig, QuantizationConfig

try:
    import librosa
except ImportError:  # pragma: no cover - optional dependency
    librosa = None


@dataclass
class FeatureResult:
    features: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def _require_librosa() -> None:
    if librosa is None:
        raise RuntimeError("librosa is required for MFCC extraction")


def compute_mfcc(audio: np.ndarray, config: AudioConfig) -> FeatureResult:
    _require_librosa()
    frame_length = int(config.sample_rate * config.frame_length_ms / 1000.0)
    hop_length = int(config.sample_rate * config.frame_hop_ms / 1000.0)
    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=frame_length,
        hop_length=hop_length,
    )
    if config.use_deltas:
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        stacked = np.vstack([mfcc, delta, delta2])
    else:
        stacked = mfcc
    stacked = stacked.T
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0) + 1e-8
    norm = (stacked - mean) / std
    return FeatureResult(features=norm.astype(np.float32), mean=mean, std=std)


def quantize_features(
    features: np.ndarray, config: QuantizationConfig
) -> Tuple[np.ndarray, float]:
    scale = float(config.scale)
    quantized = np.clip(np.round(features / scale), -128, 127).astype(np.int8)
    return quantized, scale


def dequantize_features(quantized: np.ndarray, scale: float) -> np.ndarray:
    return quantized.astype(np.float32) * float(scale)
