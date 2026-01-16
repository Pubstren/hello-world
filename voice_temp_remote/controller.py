"""PID controller with a simple mode/state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import ControlConfig, PIDConfig


class Mode(Enum):
    OFF = "off"
    ON = "on"
    HOLD = "hold"


@dataclass
class ControllerState:
    mode: Mode
    setpoint: float
    integral: float
    prev_error: float


class PIDController:
    def __init__(self, control_cfg: ControlConfig, pid_cfg: PIDConfig) -> None:
        self._cfg = control_cfg
        self._pid = pid_cfg
        self._state = ControllerState(
            mode=Mode.OFF,
            setpoint=control_cfg.initial_setpoint,
            integral=0.0,
            prev_error=0.0,
        )

    @property
    def state(self) -> ControllerState:
        return self._state

    def apply_command(self, command: str) -> None:
        if command == "on":
            self._state.mode = Mode.ON
        elif command == "off":
            self._state.mode = Mode.OFF
        elif command == "hold":
            self._state.mode = Mode.HOLD
        elif command == "up":
            self._state.setpoint += self._cfg.delta_setpoint
        elif command == "down":
            self._state.setpoint -= self._cfg.delta_setpoint

    def compute(self, measurement: float) -> float:
        if self._state.mode == Mode.OFF:
            self._state.integral = 0.0
            self._state.prev_error = 0.0
            return self._cfg.u_min

        error = self._state.setpoint - measurement
        derivative = (error - self._state.prev_error) / self._cfg.dt
        u = (
            self._pid.kp * error
            + self._pid.ki * self._state.integral
            + self._pid.kd * derivative
        )
        u_clamped = max(self._cfg.u_min, min(self._cfg.u_max, u))
        if u_clamped == u:
            self._state.integral += error * self._cfg.dt
        self._state.prev_error = error
        return u_clamped
