"""Configuration values for the voice temperature remote system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    frame_length_ms: float = 25.0
    frame_hop_ms: float = 10.0
    n_mfcc: int = 13
    use_deltas: bool = True


@dataclass(frozen=True)
class QuantizationConfig:
    scale: float = 0.1
    scale_q_multiplier: int = 1000


@dataclass(frozen=True)
class PacketConfig:
    magic: bytes = b"VT"
    version: int = 1


@dataclass(frozen=True)
class ControlConfig:
    dt: float = 0.2
    initial_setpoint: float = 25.0
    delta_setpoint: float = 1.0
    u_min: float = 0.0
    u_max: float = 1.0


@dataclass(frozen=True)
class PIDConfig:
    kp: float = 0.6
    ki: float = 0.04
    kd: float = 0.02


@dataclass(frozen=True)
class PlantConfig:
    tau: float = 30.0
    gain: float = 1.4
    env_temp: float = 22.0
    noise_std: float = 0.15
