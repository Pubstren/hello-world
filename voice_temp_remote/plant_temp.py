"""Temperature plant model treated as a black box for the controller."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import PlantConfig


@dataclass
class PlantState:
    temperature: float


class TemperaturePlant:
    def __init__(self, config: PlantConfig, initial_temp: float | None = None) -> None:
        self._config = config
        if initial_temp is None:
            initial_temp = config.env_temp
        self._state = PlantState(temperature=initial_temp)

    def step(self, u: float, dt: float, disturbance: float = 0.0) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        temp = self._state.temperature
        dtemp = dt * (
            -(temp - self._config.env_temp) / self._config.tau
            + self._config.gain * u
            + disturbance
        )
        temp = temp + dtemp
        self._state.temperature = temp
        meas = temp + np.random.normal(0.0, self._config.noise_std)
        return float(meas)

    def apply_random_disturbance(self, prob: float = 0.05, magnitude: float = -0.5) -> float:
        if np.random.rand() < prob:
            return magnitude
        return 0.0
